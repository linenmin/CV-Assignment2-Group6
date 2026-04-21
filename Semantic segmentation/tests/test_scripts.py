import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_analyze_dataset_script_runs_directly():
    script_path = PROJECT_ROOT / "scripts" / "analyze_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "project_root=" in result.stdout
    assert "dataset_root=" in result.stdout
