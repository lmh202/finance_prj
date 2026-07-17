"""Shared helpers for the view modules."""

import streamlit as st

import api_client as api

BACKEND_DOWN_MSG = (
    "Backend not running — start it with `scripts/dev.ps1` or "
    "`uvicorn main:app --app-dir backend --port 8000`."
)


def call(fn, *args, **kwargs):
    """Run an api_client call, translating expected conditions into the
    page messages the in-process version used to show. Returns None when
    the page should stop rendering."""
    try:
        return fn(*args, **kwargs)
    except api.ApiUnavailable:
        st.error(BACKEND_DOWN_MSG)
    except api.ApiMarker as marker:
        if marker.detail == api.EMPTY_PORTFOLIO:
            st.info("Your portfolio is empty — build it on the Home page first.")
        else:
            st.error("Could not load price history — check your connection and retry.")
    return None


def portfolio_key() -> str:
    """Cache key that changes whenever the saved portfolio changes, so
    st.cache_data-wrapped engine calls refresh after edits on Home."""
    return api.get_portfolio().to_json()
