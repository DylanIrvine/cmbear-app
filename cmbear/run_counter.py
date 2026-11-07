"""A shared, persistent run counter backed by Upstash Redis over HTTP.

Streamlit Community Cloud has an ephemeral filesystem, so a counter cannot live
in a local file: it would reset on every restart, redeploy or sleep. This keeps
the count in a serverless Redis database reached over its REST API. That
persists across restarts and increments atomically, so simultaneous runs are all
counted rather than racing over the same number.

Configure it with the database's REST URL and token, either in Streamlit secrets
(app Settings then Secrets on Community Cloud, or .streamlit/secrets.toml
locally):

    [upstash]
    url = "https://<name>.upstash.io"
    token = "<REST token>"

or as environment variables UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN
for local development. If neither is set, every function returns None and the app
simply shows no count instead of failing. Never commit the token: keep
.streamlit/secrets.toml in .gitignore.

The same database can serve several apps; each uses its own key (RUN_KEY).
"""

from __future__ import annotations

import os

import requests

RUN_KEY = "cmbear_run_count"
_TIMEOUT_SECONDS = 5


def _config():
    """Return (rest_url, token) from env or Streamlit secrets, or None."""
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not (url and token):
        try:
            import streamlit as st
            url = st.secrets["upstash"]["url"]
            token = st.secrets["upstash"]["token"]
        except Exception:
            return None
    if not (url and token):
        return None
    return url.rstrip("/"), token


def _command(*parts):
    """Run one Upstash REST command, returning its 'result'.

    Raises when the counter is not configured, so callers can tell 'no backend'
    (show nothing) apart from 'key not set yet' (a genuine count of zero).
    """
    config = _config()
    if config is None:
        raise RuntimeError("run counter is not configured")
    url, token = config
    endpoint = url + "/" + "/".join(str(p) for p in parts)
    response = requests.get(endpoint, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("result")


def increment_runs(key: str = RUN_KEY):
    """Atomically add one to the count and return the new total, or None."""
    try:
        result = _command("incr", key)
        return int(result) if result is not None else None
    except Exception:
        return None


def get_runs(key: str = RUN_KEY):
    """Return the current total (0 if the key is unset), or None if unavailable."""
    try:
        result = _command("get", key)
    except Exception:
        return None
    return int(result) if result not in (None, "") else 0
