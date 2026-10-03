"""HTTPS-запросы стандартной библиотекой с запасным набором корневых сертификатов (certifi).

На «свежей» Windows (особенно Windows Server) в хранилище сертификатов может не быть нужных корневых
сертификатов, и ssl отвечает CERTIFICATE_VERIFY_FAILED, хотя сайт доступен (скачивание через requests/certifi при
этом работает). Сначала пробуем обычный контекст; при ошибке проверки сертификата - один повтор с certifi."""
from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from typing import Optional

_certifi_ctx: Optional[ssl.SSLContext] = None


def _is_cert_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc)


def certifi_context() -> Optional[ssl.SSLContext]:
    global _certifi_ctx
    if _certifi_ctx is None:
        try:
            import certifi

            _certifi_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001 - certifi не установлен: запасного пути нет
            return None
    return _certifi_ctx


def urlopen(req, timeout: float = 10.0):
    """Как urllib.request.urlopen, но с повтором через certifi при ошибке проверки сертификата."""
    try:
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - только https
    except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
        ctx = certifi_context() if _is_cert_error(exc) else None
        if ctx is None:
            raise
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)  # noqa: S310
