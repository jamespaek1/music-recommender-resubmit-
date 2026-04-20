"""
Structured logging for the Music Discovery Agent.

Logs every step of the agent pipeline so behavior is observable,
debuggable, and auditable. Logs to both console and a log file.
"""

import logging
import os
import json
from datetime import datetime


class AgentLogger:
    """Logs agent steps with structured metadata."""

    def __init__(self, log_dir: str = None, verbose: bool = True):
        self.verbose = verbose
        self.entries = []

        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"agent_{timestamp}.log")

        self.logger = logging.getLogger("MusicAgent")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)

        # Console handler (only if verbose)
        if verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

    def log_step(self, step: str, detail: str, data: dict = None):
        """Log a named pipeline step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "detail": detail,
            "data": data or {},
        }
        self.entries.append(entry)
        msg = f"[{step}] {detail}"
        if data:
            msg += f" | {json.dumps(data, default=str)}"
        self.logger.info(msg)

    def log_warning(self, step: str, message: str):
        """Log a warning during processing."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "detail": message,
            "level": "WARNING",
        }
        self.entries.append(entry)
        self.logger.warning(f"[{step}] {message}")

    def log_error(self, step: str, message: str):
        """Log an error."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "detail": message,
            "level": "ERROR",
        }
        self.entries.append(entry)
        self.logger.error(f"[{step}] {message}")

    def get_entries(self) -> list:
        """Return all log entries for this session."""
        return list(self.entries)

    def summary(self) -> str:
        """Return a brief summary of the logged steps."""
        steps = [e["step"] for e in self.entries]
        warnings = [e for e in self.entries if e.get("level") == "WARNING"]
        errors = [e for e in self.entries if e.get("level") == "ERROR"]
        return (
            f"Steps: {' → '.join(steps)} | "
            f"Warnings: {len(warnings)} | Errors: {len(errors)}"
        )
