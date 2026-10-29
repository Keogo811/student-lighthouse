"""Tests for the CLI."""
import subprocess

def test_cli_help() -> None:
    """Check that the CLI entry point responds."""
    result = subprocess.run(["lighthouse", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout