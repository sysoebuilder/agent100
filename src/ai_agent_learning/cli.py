import argparse
import asyncio

from ai_agent_learning.main import run_interactive


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent100",
        description="个人 LLM 命令行助手",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="agent100 0.1.1",
    )

    return parser


def main() -> None:
    parser = create_parser()
    parser.parse_args()

    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        print("\n已退出")