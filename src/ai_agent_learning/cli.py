import argparse
import asyncio

from ai_agent_learning.main import run_chat, run_config, run_taskplan


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent100",
        description="个人 LLM 命令行助手",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="agent100 0.3.0",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser(
        "config",
        help="选择提供商、输入模型名称并更新当前配置",
    )

    commands.add_parser(
        "chat",
        help="进入多轮会话",
    )

    taskplan_parser = commands.add_parser(
        "taskplan",
        help="生成任务计划",
    )

    taskplan_parser.add_argument(
        "goal",
        nargs="?",
        help="需要规划的任务目标",
    )

    return parser


def main() -> None:
    args = create_parser().parse_args()

    try:
        if args.command == "config":
            run_config()

        elif args.command == "chat":
            asyncio.run(run_chat())

        elif args.command == "taskplan":
            asyncio.run(run_taskplan(args.goal))

    except KeyboardInterrupt:
        print("\n已退出")
