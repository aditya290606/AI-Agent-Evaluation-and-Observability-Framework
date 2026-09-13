"""
HTTP and local transport for sending AgentPulse telemetry.

Features:
  - Non-blocking background worker queue.
  - Safe error handling: never crashes user code if backend is down.
  - API Key authentication (X-API-Key and Authorization: Bearer).
  - Direct local database fallback when backend is local or offline.
"""

import atexit
import json
import logging
import queue
import threading
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from agentpulse.config import get_config
from agentpulse.models import SpanData

logger = logging.getLogger("agentpulse.transport")


class TelemetryTransport:
    """Dispatches span telemetry to AgentPulse backend or local storage."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._is_dispatching: bool = False
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="agentpulse-transport-worker")
        self._worker.start()
        atexit.register(self.shutdown)

    def send_span(self, span: SpanData):
        """Enqueue span for transmission. Never blocks or raises."""
        config = get_config()
        if config.disabled:
            return

        try:
            self._queue.put_nowait(span)
        except queue.Full:
            logger.warning("[AgentPulse] Telemetry queue full; dropping span %s", span.span_id)
        except Exception as e:
            logger.debug("[AgentPulse] Error enqueuing span: %s", e)

    def _worker_loop(self):
        """Background thread dequeuing and transmitting spans."""
        while not self._stop_event.is_set():
            try:
                span = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._is_dispatching = True
                self._dispatch(span)
            except Exception as e:
                logger.debug("[AgentPulse] Error dispatching span: %s", e)
            finally:
                self._is_dispatching = False
                self._queue.task_done()

    def _dispatch(self, span: SpanData):
        """Send span over HTTP or directly to local database."""
        config = get_config()
        payload = span.to_dict()

        # 1. If configured as 'local', write directly to local DB
        if config.backend_url.lower() == "local":
            self._write_local_db(span)
            return

        # 2. Try HTTP transmission
        url = f"{config.backend_url}/v1/spans"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentPulse-Python-SDK/1.0",
        }
        if config.api_key:
            headers["X-API-Key"] = config.api_key
            headers["Authorization"] = f"Bearer {config.api_key}"

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=config.timeout) as resp:
                if resp.status < 300:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as net_err:
            logger.debug("[AgentPulse] HTTP dispatch to %s failed: %s", url, net_err)
            # Graceful local fallback: if backend is down, attempt local persistence if DB exists
            self._write_local_db(span)
        except Exception as ex:
            logger.debug("[AgentPulse] Unexpected dispatch error: %s", ex)
            self._write_local_db(span)

    def _write_local_db(self, span: SpanData):
        """Fallback to local database via unified ingestion engine."""
        try:
            from src.storage.ingestion import ingest_span_telemetry
            payload = span.to_dict()
            # Ensure metadata includes active agent config
            if "metadata" not in payload:
                payload["metadata"] = {}
            if "agent_name" not in payload["metadata"]:
                payload["metadata"]["agent_name"] = get_config().agent_name
            if "agent_version" not in payload["metadata"]:
                payload["metadata"]["agent_version"] = get_config().agent_version
            if "model" not in payload["metadata"]:
                payload["metadata"]["model"] = "claude-3-5-haiku"

            ingest_span_telemetry(payload, auto_evaluate=True)
        except ImportError:
            # Running standalone outside framework
            pass
        except Exception as e:
            logger.debug("[AgentPulse] Local ingestion failed: %s", e)

    def flush(self, timeout: float = 5.0):
        """Wait until all pending telemetry is sent or timeout expires."""
        start = time.time()
        while (not self._queue.empty() or self._is_dispatching) and (time.time() - start < timeout):
            time.sleep(0.05)

    def shutdown(self):
        """Flush and terminate background worker thread."""
        self.flush(timeout=1.0)
        self._stop_event.set()


_global_transport: Optional[TelemetryTransport] = None


def get_transport() -> TelemetryTransport:
    """Retrieve global telemetry transport."""
    global _global_transport
    if _global_transport is None:
        _global_transport = TelemetryTransport()
    return _global_transport
