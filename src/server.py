"""
AgentPulse Telemetry Ingestion HTTP Server.

Provides a fast, zero-dependency ingestion API for the AgentPulse SDK:
  - POST /v1/spans       : Ingest span telemetry from @observe
  - POST /v1/telemetry   : Alias for /v1/spans
  - GET  /v1/health      : Healthcheck endpoint
  - GET  /health         : Healthcheck endpoint

Authentication:
  Checks `X-API-Key` or `Authorization: Bearer <key>` against AGENTPULSE_API_KEY if configured.

Usage:
    python -m src.server --port 8000
"""

import argparse
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import os
import sys
import threading
import time
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, AgentRecord, AgentVersionRecord
from src.security.redactor import SecretRedactor
from src.security.guardrails import ResourceGuardrails


from src.storage.ingestion import (
    ingest_span_telemetry,
    validate_telemetry_payload,
    authenticate_api_key,
    IngestionValidationError,
)


class TelemetryRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP telemetry ingestion requests."""

    server_version = "AgentPulse-Ingest/1.0"

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authenticate(self) -> bool:
        # Check X-API-Key
        api_key = self.headers.get("X-API-Key")
        if not api_key:
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:].strip()
        return authenticate_api_key(api_key)

    def do_GET(self):
        if self.path in ("/health", "/v1/health", "/"):
            self._send_json(200, {
                "status": "healthy",
                "service": "AgentPulse Ingestion API",
                "version": "1.0.0",
                "time": time.time(),
            })
            return
        self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if not self._authenticate():
            self._send_json(401, {"error": "Unauthorized: Invalid or missing API key"})
            return

        if self.path not in ("/v1/spans", "/v1/telemetry"):
            self._send_json(404, {"error": "Endpoint not found"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        if content_len > 5 * 1024 * 1024:  # 5MB max payload limit
            self._send_json(413, {"error": "Payload exceeds maximum 5MB size limit"})
            return

        raw_body = self.rfile.read(content_len)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
            return

        try:
            result = ingest_span_telemetry(payload, auto_evaluate=True)
            self._send_json(200, result)
        except IngestionValidationError as val_err:
            self._send_json(400, {"error": f"Validation error: {str(val_err)}"})
        except Exception as e:
            self._send_json(500, {"error": f"Failed to persist span: {str(e)}"})

    def log_message(self, format, *args):
        """Suppress noisy default request logs; output only errors."""
        pass


def run_server(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    """Initialize DB and launch the Threading HTTP Server."""
    init_db()
    server = ThreadingHTTPServer((host, port), TelemetryRequestHandler)
    print(f"[AgentPulse] Telemetry Ingestion Server listening on http://{host}:{port}")
    return server


def start_background_server(port: int = 8000) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    """Helper to run the server in a daemon thread (useful for testing)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), TelemetryRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentPulse Ingestion Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Listening port")
    args = parser.parse_args()

    srv = run_server(host=args.host, port=args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        srv.shutdown()
