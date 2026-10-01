"""Deny inherited non-loopback networking in Gate A subprocesses."""

from __future__ import annotations

import socket
from typing import Any

_original_connect = socket.socket.connect
_original_getaddrinfo = socket.getaddrinfo
_ALLOWED = {"127.0.0.1", "::1", "localhost"}


def _blocked_connect(instance: socket.socket, address: Any) -> Any:
    if isinstance(address, tuple) and address and str(address[0]) in _ALLOWED:
        return _original_connect(instance, address)
    if isinstance(address, str):
        return _original_connect(instance, address)
    raise OSError("external network disabled by Gate A")


def _blocked_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host is None or str(host) in _ALLOWED:
        return _original_getaddrinfo(host, *args, **kwargs)
    raise OSError("external DNS disabled by Gate A")


socket.socket.connect = _blocked_connect
socket.getaddrinfo = _blocked_getaddrinfo
