from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_repo_root() -> Path:
    return get_project_root().parent


def get_dataset_root() -> Path:
    return get_repo_root() / "kul-computer-vision-ga-2-2026"
