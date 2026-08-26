import os
from pathlib import Path


def find_project_root() -> Path:
    """从当前目录向上查找项目根目录。"""
    current = Path.cwd().resolve()

    for directory in (current, *current.parents):
        if (
            (directory / ".git").exists()
            or (directory / "pyproject.toml").is_file()
        ):
            return directory

    return current


def get_data_dir() -> Path:
    """允许用户通过环境变量覆盖数据目录。"""
    configured = os.getenv("LLM_CLI_DATA_DIR")

    if configured:
        data_dir = Path(configured).expanduser().resolve()
    else:
        data_dir = find_project_root() / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir