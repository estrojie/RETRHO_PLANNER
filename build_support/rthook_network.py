"""Runtime defaults needed by frozen astronomy/network applications."""
from __future__ import annotations

import os

try:
    import certifi

    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
except Exception:
    pass

os.environ.setdefault("MPLBACKEND", "QtAgg")
