"""The network interface fallback (infra/netroute.py): mocked sockets / openers, a real loopback server, the hub adapter,
the downloader of the online installer, the Settings combo."""
from __future__ import annotations

import http.server
import json
import socket
import ssl
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from infra import netroute as nr
from infra.netroute import DEFAULT, NOPROXY, Route
from tests.test_studio import app, lib, make_studio  # noqa: F401  (fixtures)

HOP_A = ("wlan0", "192.168.1.20")
HOP_B = ("tun0", "10.8.0.2")


@pytest.fixture
def two_ifaces(monkeypatch):
    monkeypatch.setattr(nr, "local_addresses", lambda ipv6=True: [HOP_A, HOP_B])
    monkeypatch.setattr(nr, "_proxies_configured", lambda: False)


class FakeOpener:
    """Opener whose success depends on the route; records the order of attempts."""

    def __init__(self, ok, log, route, ctx=None):
        self.ok, self.log, self.route, self.ctx = ok, log, route, ctx

    def open(self, req, timeout=None):
        self.log.append(self.route)
        outcome = self.ok(self.route)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def install(monkeypatch, ok):
    log = []
    monkeypatch.setattr(nr, "_opener", lambda route, ctx, ct: FakeOpener(ok, log, route, ctx))
    return log


# ------------------------------------------------------------------------------------------------ the interface list
def test_local_addresses_skips_loopback_linklocal_down_and_puts_ipv4_first(monkeypatch):
    raw = [("lo", "127.0.0.1"), ("eth0", "fe80::1%eth0"), ("eth0", "fd00::5"), ("eth0", "192.168.0.7"), ("eth0", "169.254.3.4"),
           ("tun0", "10.8.0.2"), ("any", "0.0.0.0"), ("eth0", "192.168.0.7")]
    monkeypatch.setattr(nr, "_from_psutil", lambda: raw)
    assert nr.local_addresses() == [("eth0", "192.168.0.7"), ("tun0", "10.8.0.2"), ("eth0", "fd00::5")]
    assert nr.local_addresses(ipv6=False) == [("eth0", "192.168.0.7"), ("tun0", "10.8.0.2")]


def test_local_addresses_falls_back_when_a_source_fails(monkeypatch):
    def boom():
        raise RuntimeError("no psutil")

    monkeypatch.setattr(nr, "_from_psutil", boom)
    monkeypatch.setattr(nr, "_from_command", lambda *a, **k: [("eth9", "192.168.5.5")])
    monkeypatch.setattr(nr, "_from_windows", lambda: [("Ethernet 3", "192.168.5.5")])
    assert nr.local_addresses() == [("eth9", "192.168.5.5")] or nr.local_addresses() == [("Ethernet 3", "192.168.5.5")]


def test_real_machine_never_lists_loopback():
    for _, ip in nr.local_addresses():
        assert not ip.startswith("127.") and ip != "::1"


# ------------------------------------------------------------------------------------------------ preference
def test_preference_env_file_and_default(monkeypatch):
    assert nr.preference() == "auto"
    nr.set_preference("tun0")
    assert nr.preference() == "tun0" and (nr.state_dir() / "net_iface.txt").is_file()
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "DEFAULT")
    assert nr.preference() == "default"                          # the environment wins over the file
    monkeypatch.delenv("VOXPRINT_NET_IFACE")
    nr.set_preference("auto")
    assert nr.preference() == "auto" and not (nr.state_dir() / "net_iface.txt").exists()


def test_candidates_order_auto(two_ifaces):
    assert nr.candidates("huggingface.co") == [DEFAULT, Route("wlan0", "192.168.1.20", False), Route("tun0", "10.8.0.2", False)]
    nr.remember(Route("tun0", "10.8.0.2", False))
    assert nr.candidates("huggingface.co")[0] == Route("tun0", "10.8.0.2", False)
    assert nr.candidates("127.0.0.1") == [DEFAULT]                # loopback never hops


def test_candidates_with_a_proxy_also_try_without_it(two_ifaces, monkeypatch):
    monkeypatch.setattr(nr, "_proxies_configured", lambda: True)
    assert nr.candidates("github.com")[:2] == [DEFAULT, NOPROXY]


def test_candidates_specific_and_default_preferences(two_ifaces, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "TUN0")
    assert nr.candidates("github.com") == [Route("tun0", "10.8.0.2", False)]
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "172.16.0.9")
    assert nr.candidates("github.com") == [Route("172.16.0.9", "172.16.0.9", False)]
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "nosuch0")
    with pytest.raises(OSError, match="nosuch0"):
        nr.candidates("github.com")
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "default")
    assert nr.candidates("github.com") == [DEFAULT]


def test_remembered_route_is_dropped_when_its_address_is_gone(monkeypatch):
    monkeypatch.setattr(nr, "local_addresses", lambda ipv6=True: [HOP_A])
    nr.remember(Route("tun0", "10.8.0.2", False))
    nr.reset_memory()                                             # a new process reads the file ...
    assert nr.remembered() is None and not (nr.state_dir() / "net_route.json").exists()      # ... and finds the VPN gone


# ------------------------------------------------------------------------------------------------ urlopen
def test_default_route_failing_hops_to_the_next_interface_and_remembers(two_ifaces, monkeypatch):
    log = install(monkeypatch, lambda r: "RESPONSE" if r.ip == "10.8.0.2" else urllib.error.URLError(TimeoutError("timed out")))
    msgs = []
    monkeypatch.setattr(nr, "on_event", msgs.append)
    assert nr.urlopen("https://huggingface.co/x", 30) == "RESPONSE"
    assert [r.ip for r in log] == ["", "192.168.1.20", "10.8.0.2"]
    assert nr.remembered() == Route("tun0", "10.8.0.2", False)
    assert json.loads((nr.state_dir() / "net_route.json").read_text())["ip"] == "10.8.0.2"
    assert msgs == ["network route for huggingface.co: tun0 (10.8.0.2)"]          # short, no secrets
    log.clear()
    assert nr.urlopen("https://huggingface.co/y", 30) == "RESPONSE"             # next request (a resumed range request): first try
    assert [r.ip for r in log] == ["10.8.0.2"]


def test_remembered_route_that_stops_working_is_forgotten_and_search_restarts(two_ifaces, monkeypatch):
    nr.remember(Route("tun0", "10.8.0.2", False))
    state = {"vpn_up": False}

    def ok(r):
        if r == DEFAULT or (r.ip == "10.8.0.2" and not state["vpn_up"]):
            return OSError("unreachable")
        return "VIA " + r.ip

    log = install(monkeypatch, ok)
    assert nr.urlopen("https://github.com/a", 5) == "VIA 192.168.1.20"
    assert [r.ip for r in log] == ["10.8.0.2", "", "192.168.1.20"]
    assert nr.remembered() == Route("wlan0", "192.168.1.20", False)


def test_http_errors_are_answers_not_network_failures(two_ifaces, monkeypatch):
    err = urllib.error.HTTPError("https://x/y", 404, "nf", {}, None)
    log = install(monkeypatch, lambda r: err)
    with pytest.raises(urllib.error.HTTPError):
        nr.urlopen("https://github.com/missing", 5)
    assert len(log) == 1 and nr.remembered() is None


def test_certificate_errors_do_not_hop(two_ifaces, monkeypatch):
    log = install(monkeypatch, lambda r: urllib.error.URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")))
    monkeypatch.setattr(nr, "_certifi_context", lambda: None)
    with pytest.raises(urllib.error.URLError):
        nr.urlopen("https://github.com/a", 5)
    assert len(log) == 1


def test_everything_failing_raises_the_last_error(two_ifaces, monkeypatch):
    log = install(monkeypatch, lambda r: urllib.error.URLError(TimeoutError(r.ip or "default")))
    with pytest.raises(urllib.error.URLError, match="10.8.0.2"):
        nr.urlopen("https://github.com/a", 5)
    assert len(log) == 3 and nr.remembered() is None


def test_explicit_interface_is_used_alone(two_ifaces, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "wlan0")
    log = install(monkeypatch, lambda r: OSError("down"))
    with pytest.raises(OSError):
        nr.urlopen("https://github.com/a", 5)
    assert [r.ip for r in log] == ["192.168.1.20"]


# ------------------------------------------------------------------------------------------------ real sockets
@pytest.fixture
def server():
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            body = b"hello"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def test_loopback_uses_the_stdlib_opener(server, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("loopback must not build the source-address opener")

    monkeypatch.setattr(nr, "_opener", refuse)
    with nr.urlopen(f"http://127.0.0.1:{server}/", timeout=5) as r:
        assert r.read() == b"hello"


def test_source_address_and_short_connect_timeout_with_real_sockets(server, monkeypatch):
    """The default attempt 'times out' (mocked socket.create_connection); the hop binds the source address and succeeds."""
    calls = []
    real = socket.create_connection

    def fake(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        calls.append((address[0], timeout, source_address))
        if source_address is None:
            raise socket.timeout("timed out")
        return real(address, timeout, source_address)

    monkeypatch.setattr(socket, "create_connection", fake)
    monkeypatch.setattr(nr, "_is_loopback_host", lambda h: False)           # let the loopback test server be hopped to
    monkeypatch.setattr(nr, "local_addresses", lambda ipv6=True: [("lo-alias", "127.0.0.1")])
    monkeypatch.setattr(nr, "_proxies_configured", lambda: False)
    monkeypatch.setattr(nr, "_usable", lambda ip: True)
    with nr.urlopen(f"http://127.0.0.1:{server}/", timeout=30) as r:
        assert r.fp.raw._sock.gettimeout() == 30                          # the transfer keeps the caller's read timeout
        assert r.read() == b"hello"
    assert calls[0][1:] == (nr.CONNECT_TIMEOUT, None)                      # first: default route, SHORT connect timeout
    assert calls[1][2] == ("127.0.0.1", 0) and calls[1][1] <= nr.HOP_TIMEOUT
    assert nr.remembered() == Route("lo-alias", "127.0.0.1", False)


# ------------------------------------------------------------------------------------------------ huggingface_hub
def test_prepare_hf_picks_the_working_interface_and_installs_it(two_ifaces, monkeypatch):
    probes, installed = [], []
    monkeypatch.setattr(nr, "_probe", lambda host, port, route, timeout: probes.append(route.ip) or route.ip == "10.8.0.2")
    monkeypatch.setattr(nr, "_install_hf_adapter", installed.append)
    assert nr.prepare_hf() == "tun0 (10.8.0.2)"
    assert probes == ["", "192.168.1.20", "10.8.0.2"] and installed == ["10.8.0.2"]
    assert nr.remembered().ip == "10.8.0.2"
    probes.clear()
    assert nr.prepare_hf() == "tun0 (10.8.0.2)" and probes == []              # cached for a few minutes
    assert nr.prepare_hf(force=True) and probes[0] == "10.8.0.2"              # re-test: the remembered route first


def test_prepare_hf_leaves_a_working_default_alone(two_ifaces, monkeypatch):
    installed = []
    monkeypatch.setattr(nr, "_probe", lambda host, port, route, timeout: True)
    monkeypatch.setattr(nr, "_install_hf_adapter", installed.append)
    assert nr.prepare_hf() == "" and installed == [""]


def test_prepare_hf_does_nothing_with_a_proxy_or_the_default_preference(two_ifaces, monkeypatch):
    installed, probes = [], []
    monkeypatch.setattr(nr, "_probe", lambda *a: probes.append(a) or False)
    monkeypatch.setattr(nr, "_install_hf_adapter", installed.append)
    monkeypatch.setenv("VOXPRINT_NET_IFACE", "default")
    assert nr.prepare_hf(force=True) == "" and not probes and installed == [""]


def test_hub_session_binds_the_source_address(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import configure_http_backend
    from huggingface_hub.utils import get_session

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    try:
        nr._install_hf_adapter("10.8.0.2")
        pool = get_session().get_adapter("https://huggingface.co").poolmanager
        assert pool.connection_pool_kw["source_address"] == ("10.8.0.2", 0)
        assert __import__("os").environ["HF_HUB_DISABLE_XET"] == "1"          # the Rust downloader cannot bind
    finally:
        __import__("os").environ.pop("HF_HUB_DISABLE_XET", None)
        from huggingface_hub import constants

        constants.HF_HUB_DISABLE_XET = False
        configure_http_backend()
    assert "source_address" not in get_session().get_adapter("https://huggingface.co").poolmanager.connection_pool_kw


def test_ensure_model_prepares_the_route_only_for_the_real_hub(monkeypatch):
    from infra import model_downloader as md

    seen = []
    monkeypatch.setattr(nr, "prepare_hf", lambda *a, **k: seen.append(1) or "")
    md._prepare_network()
    assert seen == [1]


# ------------------------------------------------------------------------------------------------ installer downloader, Settings
def test_online_fetch_uses_the_hopper(tmp_path, two_ifaces, monkeypatch):
    from tools import online_fetch as of

    assert of._nr is not None
    log = install(monkeypatch, lambda r: OSError("x"))
    monkeypatch.setattr(nr, "candidates", lambda host: [DEFAULT])
    with pytest.raises(OSError):
        of._open("https://github.com/x/y.zip")
    assert len(log) == 1
    with pytest.raises(of.FetchError):
        of._open("http://example.com/insecure")                              # the HTTPS-only rule is unchanged


def test_single_file_linux_downloader_has_the_hopper_inlined(tmp_path):
    import runpy

    from tools import make_linux_package as mk

    mk.build(Path(__file__).resolve().parents[1], tmp_path / "o", "v0.1.0-beta")
    ns = runpy.run_path(str(tmp_path / "o" / "voxprint-fetch.py"), run_name="not_main")
    assert ns["_nr"] is not None and hasattr(ns["_nr"], "urlopen") and ns["_nr"].CONNECT_TIMEOUT == nr.CONNECT_TIMEOUT


def test_settings_dialog_network_interface_choice(app, lib, monkeypatch):
    from core import i18n

    i18n.set_language("en")
    monkeypatch.setattr(nr, "local_addresses", lambda ipv6=True: [HOP_A, HOP_B])
    s = make_studio(lib)
    try:
        d = s.settings_dialog()
        items = [d.cmb_net.itemData(i) for i in range(d.cmb_net.count())]
        assert items == ["auto", "default", "wlan0", "tun0"] and d.cmb_net.currentData() == "auto"
        assert d.lbl_net.text() == "Network interface"
        d.cmb_net.setCurrentIndex(items.index("tun0"))
        assert nr.preference() == "tun0"
        d.cmb_net.setCurrentIndex(0)
        assert nr.preference() == "auto"
    finally:
        s.shutdown()


# ------------------------------------------------------------------------------------------------ speed probe
def test_the_fastest_route_is_picked_by_a_ranged_get_and_sources_are_ranked(two_ifaces, monkeypatch):
    import io
    import time as _t

    from tools import online_fetch as of

    delay = {"": 0.25, "192.168.1.20": 0.02}                   # default slow, wlan0 fast, tun0 fails
    seen = []

    def open_fn(route, req, timeout):
        seen.append(req.get_header("Range"))
        if route.ip not in delay:
            raise OSError("unreachable")
        _t.sleep(delay[route.ip])
        return io.BytesIO(b"x" * 1024)

    nr.reset_memory()
    best = nr.pick_fastest("https://example.org/f.zip", routes=nr.candidates("example.org"), open_fn=open_fn)
    assert best == Route(*HOP_A, False) and nr.remembered() == best and set(seen) == {"bytes=0-262143"}
    assert nr.pick_fastest("https://example.org/f.zip", routes=[DEFAULT, Route(*HOP_B, False)],
                           open_fn=lambda r, q, t: (_ for _ in ()).throw(OSError("down"))) is None
    monkeypatch.setattr(of, "_SPEED", {"up.example": 1e5, "mirror.example": 9e6, "dead.example": 0.0})
    assert of.ranked(["https://dead.example/a", "https://up.example/a", "https://mirror.example/a"]) == [
        "https://mirror.example/a", "https://up.example/a", "https://dead.example/a"]
