import shutil
from pathlib import Path
import subprocess


def test_quick_generate(tmp_path):
    out = tmp_path / "out"
    # Run the CLI
    cmd = ["python3", "-m", "secureai_dataset.setup_complete_dataset", "--out", str(out), "--quick", "--count", "3"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    # Check some files
    assert (out / "benign" / "images" / "portraits").exists()
    assert (out / "adversarial" / "scaling_attacks" / "bicubic").exists()
    assert (out / "documentation" / "dataset_stats.json").exists()
