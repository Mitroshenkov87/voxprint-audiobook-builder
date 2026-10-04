"""HTTPS requests through the standard library with a fallback set of root certificates (certifi).

On a "fresh" Windows (especially Windows Server) the certificate store may lack some root certificates, so ``ssl``
answers ``CERTIFICATE_VERIFY_FAILED`` even though the site is reachable (downloading through ``requests``/certifi
works fine in the same situation).  We first try the normal context and, only on a certificate verification error,
retry exactly once with the certifi bundle.  Connections that cannot be made are retried through the other network
interfaces (:mod:`infra.netroute`).
"""
from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from typing import Optional

_certifi_ctx: Optional[ssl.SSLContext] = None


def _is_cert_error(exc: BaseException) -> bool:
    """Return True if ``exc`` (or its ``reason``) is a TLS certificate verification failure."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc)


def certifi_context() -> Optional[ssl.SSLContext]:
    """Return a (cached) SSL context that trusts the certifi bundle, or None when certifi is unavailable."""
    global _certifi_ctx
    if _certifi_ctx is None:
        try:
            import certifi

            _certifi_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001 - certifi is not installed: there is no fallback
            return None
    return _certifi_ctx


def urlopen(req, timeout: float = 10.0):
    """Like :func:`urllib.request.urlopen`, with two fallbacks (see :mod:`infra.netroute`): a connection that cannot be made
    is retried from the other network interfaces (VPN / odd adapters), and a certificate verification error is retried
    once through the certifi bundle."""
    from infra import netroute

    return netroute.urlopen(req, timeout=timeout)  # noqa: S310 - https only
