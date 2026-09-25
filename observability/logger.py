"""
Structured logging of each turn to JSONL file.
One line per turn, easy to pipe to jq or ingest into ELK.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import structlog

logger = structlog.get_logger()


class TurnLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def log(
        self,
        turn_id: str,
        conversation_id: str,
        user_input: str,
        agent_reply: str,
        tool_calls: List[Dict[str, Any]],
        stages: Dict[str, float],
        total_ms: float,
        model_used: str,
    ) -> None:
        """Write one JSON line representing the turn."""
        entry = {
            "ts": datetime.now().isoformat(),
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "agent_reply": agent_reply,
            "tool_calls": tool_calls,
            "stages": stages,
            "total_ms": round(total_ms, 2),
            "model_used": model_used,
        }
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("turn_log_failed", error=str(e))