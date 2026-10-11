"""Repository-local defaults for reproducible examples; independent of cwd."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_path(value: str | Path, config: Path) -> Path:
    """Resolve a YAML path against its containing directory, never cwd."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config.resolve().parent / path).resolve()
