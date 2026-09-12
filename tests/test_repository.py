from pathlib import Path


def test_learning_repository_has_expected_structure() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ai_agent_learning").is_dir()
