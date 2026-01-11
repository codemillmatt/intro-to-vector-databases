"""
Shared Pinecone utilities for consistent connection configuration.

This module centralizes the logic for connecting to Pinecone Local
in a DevContainer environment, handling the control-plane vs data-plane
port differences and Docker DNS quirks.
"""

import os
from urllib.parse import urlparse

from pinecone import Pinecone

# Default configuration (optimized for DevContainer environment)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "local")
PINECONE_HOST = os.getenv("PINECONE_HOST", "http://pinecone:5081")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bookstore")


def _dotted_hostname(hostname: str | None) -> str | None:
    """
    Return hostname with a trailing dot if needed.

    The Pinecone SDK validates custom hosts by requiring a dot (.) in the
    host string. Docker service names like "pinecone" don't include one,
    but "pinecone." resolves correctly in Docker DNS and satisfies the
    SDK check.
    """
    if not hostname:
        return hostname
    if hostname in {"localhost", "127.0.0.1"}:
        return hostname
    return f"{hostname}." if "." not in hostname else hostname


def get_pinecone_index_host() -> str:
    """
    Return the data-plane host for Pinecone index operations.

    Pinecone Local commonly exposes a separate data-plane port (5082).
    In some environments the SDK may resolve this as localhost, which
    fails inside containers. This helper allows overriding the index
    host via env var and auto-detects the common local setup.
    """
    override = os.getenv("PINECONE_INDEX_HOST")
    if override:
        return override

    parsed = urlparse(PINECONE_HOST)
    hostname = parsed.hostname
    port = parsed.port
    scheme = parsed.scheme or "http"

    # Pinecone Local uses 5081 for control plane, 5082 for data plane
    if hostname in {"pinecone", "localhost", "127.0.0.1"} and (port in {None, 5081}):
        dotted = _dotted_hostname(hostname)
        return f"{scheme}://{dotted}:5082"

    return ""


def get_pinecone_client() -> Pinecone:
    """Get a Pinecone client instance."""
    return Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)


def get_pinecone_index(pc: Pinecone | None = None, index_name: str = INDEX_NAME):
    """
    Get a Pinecone Index instance with properly configured host.

    Args:
        pc: Optional existing Pinecone client. If None, one will be created.
        index_name: The name of the index to connect to.

    Returns:
        A Pinecone Index object ready for queries/upserts.
    """
    if pc is None:
        pc = get_pinecone_client()

    index_host = get_pinecone_index_host()
    if index_host:
        return pc.Index(index_name, host=index_host)
    return pc.Index(index_name)
