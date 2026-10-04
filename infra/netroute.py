"""Network route fallback ("interface hopper") for every download of Voxprint.  Standard library only (psutil is used if present).

Problem: with a VPN or an unusual adapter the operating system's default route cannot reach a host (Hugging Face, GitHub)
although the internet works through another network interface.  What this module does:

1. A connection is tried the normal way first, with a SHORT connect timeout (``CONNECT_TIMEOUT``, 8 s; the read timeout of the
   caller is kept for the transfer itself).
2. If that fails at the network level (timeout, refused, unreachable, DNS) the local IPv4/IPv6 addresses are listed (loopback,
   link-local, down interfaces skipped) and the connection is retried from each of them in turn (``source_address`` of the
   socket), optionally also without the system proxy.  An HTTP answer (404, 403 ...) always counts as "the network works".
3. The route that worked is remembered (memory + ``<state>/net_route.json``) and tried FIRST next time - also for the resumed
   range requests of a download and for the mirror fallback - and forgotten when it fails (then the search starts again).

The user can force a choice: ``VOXPRINT_NET_IFACE`` or ``<state>/net_iface.txt`` (Settings -> Network interface):
``auto`` (default: the behaviour above), ``default`` (never hop: the OS decides) or an interface name / local IP address
(bind to it, never hop).  Nothing here detects a particular VPN product.

Entry points: :func:`urlopen` (urllib users), :func:`prepare_hf` (huggingface_hub / requests, which cannot bind per request:
the working source address is installed once as the hub's HTTP backend), :func:`local_addresses`, :func:`describe`.
"""
from __future__ import annotations

import http.client
import importlib
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("voxprint.net")

CONNECT_TIMEOUT = 8.0       # first attempt: how long to wait for a connection before trying other interfaces
HOP_TIMEOUT = 6.0           # every further attempt
MAX_HOPS = 8                # at most so many local addresses are tried per request
ENV_IFACE = "VOXPRINT_NET_IFACE"
PREF_FILE = "net_iface.txt"
ROUTE_FILE = "net_route.json"
HF_PROBE_TTL = 300.0

#: called with a short English message whenever the working route changes (the installer prints it; the app logs it)
on_event: Callable[[str], None] = lambda msg: log.info("%s", msg)
_state_dir: Optional[Path] = None


# ----------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Route:
    """How to connect: from which local address (``ip`` empty = the OS chooses) and with or without the system proxy."""
    name: str = "default"
    ip: str = ""
    proxy: bool = True

    def label(self) -> str:
        if not self.ip:
            return self.name
        return f"{self.name} ({self.ip})" if self.name != self.ip else self.ip


DEFAULT = Route()
NOPROXY = Route("default, no proxy", "", False)


def configure(state_dir: Optional[Path] = None) -> None:
    """Tell the module where the small state files live (the app passes ``paths.state_dir()``)."""
    global _state_dir
    _state_dir = Path(state_dir) if state_dir else None


def state_dir() -> Path:
    """The folder of ``net_iface.txt`` / ``net_route.json`` (same place as the app's state folder)."""
    if _state_dir is not None:
        return _state_dir
    env = os.environ.get("VOXPRINT_HOME")
    if env:
        return Path(env) / "state"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Voxprint" / "state"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    return (Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share") / "voxprint" / "state"


# ----------------------------------------------------------------------------------------------- interface list
def _usable(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    return not (a.is_loopback or a.is_link_local or a.is_unspecified or a.is_multicast or a.is_reserved)


def _from_psutil() -> List[Tuple[str, str]]:
    psutil = importlib.import_module("psutil")      # optional; loaded by name so PyInstaller does not bundle it into voxprint-fetch.exe

    out = []
    stats = psutil.net_if_stats()
    for name, addrs in psutil.net_if_addrs().items():
        st = stats.get(name)
        if st is not None and not st.isup:
            continue
        for a in addrs:
            if a.family in (socket.AF_INET, socket.AF_INET6):
                out.append((name, a.address.split("%")[0]))
    return out


def _from_command(cmd: List[str], pattern: str) -> List[Tuple[str, str]]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    out = []
    cur = ""
    for line in r.stdout.splitlines():
        m = re.match(pattern, line)
        if m and m.lastindex == 2:
            out.append((m.group(1), m.group(2)))
        elif re.match(r"^(\S+?):\s", line) and not line.startswith(" "):      # ifconfig: "en0: flags=..."
            cur = re.match(r"^(\S+?):", line).group(1)
        else:
            m2 = re.match(r"^\s+inet6?\s+([0-9a-fA-F:.]+)", line)
            if m2 and cur:
                out.append((cur, m2.group(1)))
    return out


def _from_windows() -> List[Tuple[str, str]]:
    """GetAdaptersAddresses through ctypes (adapter friendly name + unicast addresses of adapters that are up)."""
    import ctypes
    from ctypes import POINTER, Structure, byref, c_char_p, c_int, c_ubyte, c_uint, c_ulong, c_void_p, c_wchar_p, cast

    class SockAddr(Structure):
        _fields_ = [("family", ctypes.c_ushort), ("data", c_ubyte * 26)]

    class SocketAddress(Structure):
        _fields_ = [("sockaddr", POINTER(SockAddr)), ("length", c_int)]

    class Unicast(Structure):
        pass

    Unicast._fields_ = [("Length", c_ulong), ("Flags", c_ulong), ("Next", POINTER(Unicast)), ("Address", SocketAddress)]

    class Adapter(Structure):
        pass

    Adapter._fields_ = [("Length", c_ulong), ("IfIndex", c_ulong), ("Next", POINTER(Adapter)), ("AdapterName", c_char_p),
                        ("FirstUnicast", POINTER(Unicast)), ("FirstAnycast", c_void_p), ("FirstMulticast", c_void_p),
                        ("FirstDns", c_void_p), ("DnsSuffix", c_wchar_p), ("Description", c_wchar_p),
                        ("FriendlyName", c_wchar_p), ("PhysicalAddress", c_ubyte * 8), ("PhysicalAddressLength", c_ulong),
                        ("Flags", c_ulong), ("Mtu", c_ulong), ("IfType", c_ulong), ("OperStatus", c_uint)]

    fn = ctypes.windll.iphlpapi.GetAdaptersAddresses      # type: ignore[attr-defined]
    size = c_ulong(32768)
    for _ in range(4):
        buf = ctypes.create_string_buffer(size.value)
        rc = fn(0, 0x2 | 0x4 | 0x8, None, buf, byref(size))     # skip anycast, multicast, DNS servers
        if rc != 111:                                             # ERROR_BUFFER_OVERFLOW: grow and retry
            break
    if rc != 0:
        raise OSError(f"GetAdaptersAddresses failed: {rc}")
    out = []
    ad = cast(buf, POINTER(Adapter))
    while ad:
        a = ad.contents
        if a.OperStatus == 1 and a.IfType != 24:                  # IfOperStatusUp, not the software loopback
            u = a.FirstUnicast
            while u:
                sa = u.contents.Address.sockaddr
                if sa:
                    fam = sa.contents.family
                    raw = bytes(sa.contents.data)
                    if fam == socket.AF_INET:
                        out.append((a.FriendlyName or a.Description or "adapter", socket.inet_ntop(socket.AF_INET, raw[2:6])))
                    elif fam == socket.AF_INET6:
                        out.append((a.FriendlyName or a.Description or "adapter", socket.inet_ntop(socket.AF_INET6, raw[6:22])))
                u = u.contents.Next
        ad = a.Next
    return out


def _from_hostname() -> List[Tuple[str, str]]:
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            if info[0] in (socket.AF_INET, socket.AF_INET6):
                out.append((info[4][0], info[4][0]))
    except OSError:
        pass
    return out


def local_addresses(ipv6: bool = True) -> List[Tuple[str, str]]:
    """``[(interface name, ip)]`` of the usable local addresses: IPv4 first, then IPv6; no loopback / link-local / down."""
    found: List[Tuple[str, str]] = []
    sources: List[Callable[[], List[Tuple[str, str]]]] = [_from_psutil]
    if sys.platform == "win32":
        sources.append(_from_windows)
    elif sys.platform == "darwin":
        sources.append(lambda: _from_command(["ifconfig"], r"$^"))
    else:
        sources.append(lambda: _from_command(["ip", "-o", "addr", "show", "up"], r"^\d+:\s+(\S+)\s+inet6?\s+([0-9a-fA-F:.]+)"))
    sources.append(_from_hostname)
    for src in sources:
        try:
            found = src()
        except Exception as exc:  # noqa: BLE001 - every source is optional
            log.debug("address source failed: %s", exc)
            found = []
        if any(_usable(ip) for _, ip in found):
            break
    out, seen = [], set()
    for name, ip in found:
        ip = ip.split("%")[0]
        if not _usable(ip) or ip in seen:
            continue
        if ":" in ip and not ipv6:
            continue
        seen.add(ip)
        out.append((str(name), ip))
    out.sort(key=lambda t: ":" in t[1])          # IPv4 before IPv6 (stable)
    return out


# ----------------------------------------------------------------------------------------------- preference and memory
def preference() -> str:
    """``auto`` / ``default`` / an interface name or IP, from ``VOXPRINT_NET_IFACE`` or ``<state>/net_iface.txt``."""
    v = os.environ.get(ENV_IFACE, "").strip()
    if not v:
        try:
            v = (state_dir() / PREF_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            v = ""
    low = v.lower()
    if low in ("", "auto"):
        return "auto"
    if low in ("default", "system", "os"):
        return "default"
    return v


def set_preference(value: str) -> None:
    """Save the choice made in Settings (``auto`` removes the file)."""
    p = state_dir() / PREF_FILE
    v = (value or "").strip()
    if v.lower() in ("", "auto"):
        p.unlink(missing_ok=True)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(v + "\n", encoding="utf-8")


def route_for(pref: str) -> Optional[Route]:
    """The route that a specific choice (name or IP) means, or None if no such interface is up."""
    if pref in ("auto", "default"):
        return DEFAULT
    try:
        ipaddress.ip_address(pref)
        return Route(pref, pref, False)
    except ValueError:
        pass
    for name, ip in local_addresses():
        if name.lower() == pref.lower():
            return Route(name, ip, False)
    return None


_remembered: Optional[Route] = None
_loaded = False


def remembered() -> Optional[Route]:
    """The route that worked last time, if its address still exists on this computer."""
    global _remembered, _loaded
    if not _loaded:
        _loaded = True
        try:
            d = json.loads((state_dir() / ROUTE_FILE).read_text(encoding="utf-8"))
            _remembered = Route(str(d["name"]), str(d.get("ip", "")), bool(d.get("proxy", False)))
        except (OSError, ValueError, KeyError, TypeError):
            _remembered = None
    r = _remembered
    if r is not None and r.ip and r.ip not in {ip for _, ip in local_addresses()}:
        forget()
        return None
    return r


def remember(route: Route) -> None:
    global _remembered, _loaded
    _remembered, _loaded = route, True
    try:
        p = state_dir() / ROUTE_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"name": route.name, "ip": route.ip, "proxy": route.proxy, "time": int(time.time())}),
                     encoding="utf-8")
    except OSError:
        pass


def forget() -> None:
    global _remembered, _loaded
    _remembered, _loaded = None, True
    try:
        (state_dir() / ROUTE_FILE).unlink(missing_ok=True)
    except OSError:
        pass


def _proxies_configured() -> bool:
    try:
        return bool({k: v for k, v in urllib.request.getproxies().items() if k in ("http", "https") and v})
    except Exception:  # noqa: BLE001
        return False


def _is_loopback_host(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def candidates(host: str) -> List[Route]:
    """The routes to try for ``host``, in order, according to the preference."""
    pref = preference()
    if pref == "default" or _is_loopback_host(host):
        return [DEFAULT]
    if pref != "auto":
        r = route_for(pref)
        if r is None:
            raise OSError(f"network interface {pref!r} not found (Settings -> Network interface)")
        return [r]
    out: List[Route] = []
    mem = remembered()
    if mem is not None:
        out.append(mem)
    for r in [DEFAULT] + ([NOPROXY] if _proxies_configured() else []):
        if r not in out:
            out.append(r)
    for name, ip in local_addresses()[:MAX_HOPS]:
        r = Route(name, ip, False)
        if r not in out:
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------- urllib
def _bound(base: type, ip: str, connect_timeout: float):
    """An HTTP(S) connection class that binds the source address and uses a short timeout for connecting only."""
    class Conn(base):       # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            if ip:
                kw["source_address"] = (ip, 0)
            super().__init__(*a, **kw)

        def connect(self):
            wanted = self.timeout
            self.timeout = connect_timeout if wanted is None else min(connect_timeout, wanted)
            try:
                super().connect()
            finally:
                self.timeout = wanted
            if self.sock is not None and wanted is not None:
                self.sock.settimeout(wanted)
    return Conn


def _opener(route: Route, context: Optional[ssl.SSLContext], connect_timeout: float) -> urllib.request.OpenerDirector:
    ctx = context or ssl.create_default_context()

    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(_bound(http.client.HTTPSConnection, route.ip, connect_timeout), req, context=ctx)

    class HTTP(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(_bound(http.client.HTTPConnection, route.ip, connect_timeout), req)

    handlers: list = [HTTPS(), HTTP()]
    if not route.proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def _is_cert_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", None)
    return isinstance(exc, ssl.SSLCertVerificationError) or isinstance(reason, ssl.SSLCertVerificationError) \
        or "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _certifi_context() -> Optional[ssl.SSLContext]:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return None


_announced: Optional[Route] = None


def _announce(route: Route, host: str, hopped: bool) -> None:
    global _announced
    if route != _announced:
        _announced = route
        if hopped or route != DEFAULT:
            on_event(f"network route for {host}: {route.label()}")


def urlopen(req, timeout: float = 30.0, context: Optional[ssl.SSLContext] = None,
            connect_timeout: float = CONNECT_TIMEOUT):
    """Like :func:`urllib.request.urlopen`, but a connection that cannot be made is retried through the other interfaces.

    ``timeout`` is the read timeout of the transfer; connecting uses the short ``connect_timeout``.  Raises the last error."""
    if isinstance(req, str):
        req = urllib.request.Request(req)
    host = urllib.parse.urlparse(req.full_url).hostname or ""
    routes = candidates(host)
    last: Optional[BaseException] = None
    for i, route in enumerate(routes):
        ct = connect_timeout if i == 0 else min(HOP_TIMEOUT, connect_timeout)
        ctx = context
        for attempt in (0, 1):                  # the second one only after a certificate error: through certifi
            try:
                resp = _opener(route, ctx, ct).open(req, timeout=timeout)
            except urllib.error.HTTPError:
                if i or route != DEFAULT:
                    remember(route)             # the server answered through this route: it works
                raise
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                if attempt == 0 and _is_cert_error(exc) and (ctx2 := _certifi_context()) is not None:
                    ctx = ctx2
                    continue
                if _is_cert_error(exc):
                    raise                       # the route is fine, the certificate is not: other interfaces will not help
                last = exc
                break
            else:
                hopped = i > 0
                if route != DEFAULT or hopped:
                    remember(route)
                _announce(route, host, hopped)
                return resp
        if route == remembered() and route != DEFAULT:
            forget()                            # the remembered route failed: search again next time
    assert last is not None
    raise last


# ----------------------------------------------------------------------------------------------- huggingface_hub (requests)
#: The whole route search before a model download may take this long (seconds); the routes are probed in parallel.
PROBE_CAP = 5.0


def disable_xet() -> None:
    """Plain HTTP downloads for huggingface_hub: its Rust "xet" transfer path (hf_xet) talks to a separate CAS host that some
    networks block or throttle (downloads then sit at 0 bytes).  ``VOXPRINT_ALLOW_XET=1`` keeps it."""
    if os.environ.get("VOXPRINT_ALLOW_XET"):
        return
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    consts = sys.modules.get("huggingface_hub.constants")
    if consts is not None:
        try:
            consts.HF_HUB_DISABLE_XET = True
        except Exception:  # noqa: BLE001
            pass


def _first_reachable(host: str, routes: List[Route], cap: float = PROBE_CAP) -> Optional[Route]:
    """Probe the routes in parallel and return the best one that connects within ``cap`` seconds: the first route (the
    remembered / default one) if it works, otherwise the earliest alternative that does; ``None`` if none connects."""
    routes = routes[:MAX_HOPS]
    if not routes:
        return None
    cond = threading.Condition()
    results: dict = {}

    def work(i: int, r: Route) -> None:
        try:
            ok = bool(_probe(host, 443, r, cap))
        except Exception:  # noqa: BLE001
            ok = False
        with cond:
            results[i] = ok
            cond.notify_all()

    for i, r in enumerate(routes):
        threading.Thread(target=work, args=(i, r), daemon=True, name="vx-route-probe").start()
    deadline = time.monotonic() + cap
    with cond:
        while True:
            for i in range(len(routes)):          # the earliest route that works, once every earlier one has failed
                if i not in results:
                    break
                if results[i]:
                    return routes[i]
            else:
                return None                       # all failed
            left = deadline - time.monotonic()
            if left <= 0:
                break
            cond.wait(left)
        ok = [i for i in sorted(results) if results[i]]
        return routes[ok[0]] if ok else None


_hf_state: Tuple[float, str] = (float("-inf"), "")


def _probe(host: str, port: int, route: Route, timeout: float) -> bool:
    try:
        socket.create_connection((host, port), timeout, source_address=(route.ip, 0) if route.ip else None).close()
        return True
    except OSError:
        return False


def _install_hf_adapter(ip: str) -> None:
    """Make huggingface_hub's requests sessions bind to ``ip`` ("" = back to the default).  The Rust downloader (hf_xet)
    cannot bind, so it is switched off while a specific address is used."""
    # loaded by name: voxprint-fetch.exe (PyInstaller) must not pull huggingface_hub / requests / torch into the installer
    configure_http_backend = importlib.import_module("huggingface_hub").configure_http_backend

    if not ip:
        configure_http_backend()
        return
    requests = importlib.import_module("requests")
    HTTPAdapter = importlib.import_module("requests.adapters").HTTPAdapter

    class Bound(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
            pool_kwargs["source_address"] = (ip, 0)
            super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def factory() -> "requests.Session":
        s = requests.Session()
        s.mount("https://", Bound())
        s.mount("http://", Bound())
        return s

    configure_http_backend(backend_factory=factory)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        importlib.import_module("huggingface_hub.constants").HF_HUB_DISABLE_XET = True
    except Exception:  # noqa: BLE001
        pass


def prepare_hf(host: str = "huggingface.co", force: bool = False) -> str:
    """Before a huggingface_hub download: pick a route that reaches ``host`` and install it for the hub.  Returns its label
    (``""`` when nothing had to change).  Cheap when repeated (the answer is kept for 5 minutes)."""
    global _hf_state
    disable_xet()
    now = time.monotonic()
    if not force and now - _hf_state[0] < HF_PROBE_TTL:
        return _hf_state[1]
    label = ""
    try:
        pref = preference()
        if pref == "default" or _proxies_configured():       # requests honours the proxy itself
            _install_hf_adapter("")
        else:
            routes = candidates(host)
            chosen = _first_reachable(host, routes)
            if chosen is None or not chosen.ip:
                _install_hf_adapter("")
                if chosen is None and len(routes) > 1:
                    on_event(f"no network route reaches {host} (tried {len(routes)})")
            else:
                _install_hf_adapter(chosen.ip)
                remember(chosen)
                label = chosen.label()
                _announce(chosen, host, True)
    except Exception as exc:  # noqa: BLE001 - never block a download because of the hopper
        log.warning("network route selection failed: %s", exc)
    _hf_state = (now, label)
    return label


def reset_memory() -> None:
    """Forget everything kept in memory (tests)."""
    global _remembered, _loaded, _announced, _hf_state
    _remembered, _loaded, _announced, _hf_state = None, False, None, (float("-inf"), "")


def describe() -> str:
    """One line for the log / Settings: the preference and the remembered route."""
    r = remembered()
    return f"preference={preference()}; remembered={r.label() if r else '-'}"
