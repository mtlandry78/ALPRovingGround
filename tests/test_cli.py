import subprocess
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_classify_with_fake_engine(tmp_path: Path) -> None:
    Image.new("RGB", (20, 10), color="white").save(tmp_path / "car1.jpg")
    csv_out = tmp_path / "out.csv"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "alprg.cli",
            "classify",
            str(tmp_path),
            "--engines",
            "fake",
            "--csv-out",
            str(csv_out),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "Using engines: fake" in result.stdout
    assert csv_out.exists()
    assert "filename,plate_text" in csv_out.read_text()


def test_cli_classify_empty_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        ["uv", "run", "python", "-m", "alprg.cli", "classify", str(tmp_path), "--engines", "fake"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert "No image files" in result.stdout


def test_cli_classify_unknown_engine(tmp_path: Path) -> None:
    Image.new("RGB", (10, 10)).save(tmp_path / "car1.jpg")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "alprg.cli",
            "classify",
            str(tmp_path),
            "--engines",
            "nonexistent",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 1
    assert "Unknown engine" in result.stdout
