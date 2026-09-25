"""
Git helper functions for stash/pull/push/status/log.
"""
import subprocess
from pathlib import Path

REPO_PATH = Path(__file__).parent.parent.resolve()
REMOTE_NAME = "origin"
BRANCH = "main"


def _run(cmd: list[str], cwd=None) -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_PATH,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def git_status() -> str:
    out, err, rc = _run(["git", "status", "--short"])
    if rc != 0:
        return f"Git error: {err}"
    if not out.strip():
        return "Working tree clean"
    return out.strip()


def git_pull() -> str:
    _run(["git", "stash"])
    out, err, rc = _run(["git", "pull", REMOTE_NAME, BRANCH])
    if rc != 0:
        return f"Pull failed: {err}"
    _run(["git", "stash", "pop"])
    return f"Pulled successfully:\n{out}"


def git_push(message: str = "Agent update") -> str:
    _run(["git", "add", "-A"])
    out, err, rc = _run(["git", "commit", "-m", message])
    if rc != 0 and "nothing to commit" not in err.lower():
        return f"Commit failed: {err}"
    out, err, rc = _run(["git", "push", REMOTE_NAME, BRANCH])
    if rc != 0:
        return f"Push failed: {err}"
    return f"Pushed successfully:\n{out}"


def git_log(n=3) -> str:
    out, err, rc = _run(["git", "log", f"-n{n}", "--oneline"])
    if rc != 0:
        return f"Log error: {err}"
    return out.strip()