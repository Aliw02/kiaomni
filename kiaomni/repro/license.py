"""Temporary offline evaluation license gate; not tamper resistant."""
from __future__ import annotations
import hashlib
import hmac
import os
_EXPECTED = "96c896849cb11a1db81573da2884e1727d158a59441e96e67d607914a6df04f5"

class LicenseError(PermissionError):
    pass

def require_license(key: str | None = None) -> None:
    code = (key if key is not None else os.environ.get("KIAOMNI_LICENSE_KEY", "")).strip()
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if not code or not hmac.compare_digest(digest, _EXPECTED):
        raise LicenseError("Invalid KiaOmni evaluation license. Set KIAOMNI_LICENSE_KEY privately.")
