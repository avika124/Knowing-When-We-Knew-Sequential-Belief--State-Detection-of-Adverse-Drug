"""Thin client for the openFDA drug adverse event API (FAERS).

Only the ``count=receivedate`` aggregation is used: for a search expression it
returns the number of matching reports received on each day.  Responses are
cached on disk so that a replay never has to hit the API twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

API_URL = "https://api.fda.gov/drug/event.json"


class OpenFDAError(RuntimeError):
    pass


class OpenFDAClient:
    def __init__(self, cache_dir, api_key=None, pause=0.35, max_retries=6, timeout=90):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENFDA_API_KEY")
        self.pause = pause
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.n_requests = 0
        self.last_updated = None

    def _cache_path(self, search):
        # The cache key is the search expression alone; the aggregation is
        # always count=receivedate, so it need not be part of the key.  A cache
        # therefore belongs to one openFDA release: delete it to refetch after
        # an openFDA update.  The committed snapshot pins a single release, so
        # the replay is reproducible without touching the network at all.
        key = hashlib.sha1((search or "__all__").encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def count_by_receivedate(self, search=None):
        """Return ``{YYYYMMDD: n_reports}`` for reports matching ``search``."""
        path = self._cache_path(search)
        if path.exists():
            cached = json.loads(path.read_text())
            self.last_updated = cached.get("last_updated", self.last_updated)
            return cached["counts"]

        params = {"count": "receivedate"}
        if search:
            params["search"] = search
        if self.api_key:
            params["api_key"] = self.api_key

        payload = self._get(params)
        if payload is None:  # openFDA answers 404 when nothing matches
            counts, last_updated = {}, None
        else:
            counts = {r["time"]: r["count"] for r in payload["results"]}
            last_updated = payload.get("meta", {}).get("last_updated")
        self.last_updated = last_updated or self.last_updated
        path.write_text(
            json.dumps({"search": search, "last_updated": last_updated, "counts": counts})
        )
        return counts

    def _get(self, params):
        delay = 2.0
        for attempt in range(self.max_retries):
            time.sleep(self.pause)
            self.n_requests += 1
            try:
                resp = self.session.get(API_URL, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            raise OpenFDAError(f"openFDA returned {resp.status_code}: {resp.text[:300]}")
        raise OpenFDAError("openFDA request failed after retries (rate limit or outage)")
