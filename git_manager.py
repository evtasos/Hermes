import subprocess
import os
from pathlib import Path

REPO_PATH = Path(__file__).parent.resolve()
REMOTE_NAME = "origin"
BRANCH = "main"


def _run(cmd: list[str], cwd=None) -> tuple[str, str, int]:
    """Run a git command, return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_PATH,
            capture_output=True,
            text=True,
            timeout=30
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
    """Pull latest code from remote."""
    # Stash any local changes first (to avoid merge conflicts)
    _run(["git", "stash"])
    
    out, err, rc = _run(["git", "pull", REMOTE_NAME, BRANCH])
    if rc != 0:
        return f"Pull failed: {err}"
    
    # Pop stash if we had local changes
    _run(["git", "stash", "pop"])
    
    return f"Pulled successfully:\n{out}"


def git_push(message: str = "Agent update") -> str:
    """Push local changes to remote."""
    # Add all changes
    _run(["git", "add", "-A"])
    
    # Commit
    out, err, rc = _run(["git", "commit", "-m", message])
    if rc != 0 and "nothing to commit" not in err.lower():
        return f"Commit failed: {err}"
    
    # Push
    out, err, rc = _run(["git", "push", REMOTE_NAME, BRANCH])
    if rc != 0:
        return f"Push failed: {err}"
    
    return f"Pushed successfully:\n{out}"


def git_log(n=3) -> str:
    """Show last N commits."""
    out, err, rc = _run(["git", "log", f"-n{n}", "--oneline"])
    if rc != 0:
        return f"Log error: {err}"
    return out.strip()
