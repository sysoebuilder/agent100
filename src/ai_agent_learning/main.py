import os
from getpass import getpass
from pathlib import Path
from dotenv import load_dotenv, set_key

from ai_agent_learning.conversation_service import ConversationService
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.session import create_session_name, save_session, select_session
from ai_agent_learning.paths import get_data_dir
from ai_agent_learning.taskplan import show_task_plan, validate_task_plan


API_KEY_ENV = "ARK_API_KEY"
REASONING_MODEL_ENV = "ARK_REASONING_MODEL"


def ask_required(
    prompt: str,
    *,
    secret: bool = False,
) -> str:
    reader = getpass if secret else input

    while True:
        try:
            value = reader(prompt).strip()
        except EOFError as error:
            raise SystemExit(
                "\n无法读取输入，配置已取消。"
            ) from error

        if value:
            return value

        print("输入不能为空，请重新输入。")


def save_env_value(
    env_file: Path,
    name: str,
    value: str,
) -> None:
    try:
        env_file.touch(exist_ok=True)

        set_key(
            str(env_file),
            name,
            value,
            quote_mode="always",
        )
    except OSError as error:
        raise SystemExit(
            f"无法保存配置文件：{env_file}"
        ) from error

    # 让本次运行立即使用刚输入的配置。
    os.environ[name] = value


def load_or_create_config() -> tuple[str, str]:
    env_file = get_data_dir() / ".env"

    # 系统环境变量优先，data/.env 只补充缺失配置。
    load_dotenv(
        env_file,
        override=False,
    )

    api_key = os.getenv(API_KEY_ENV, "").strip()
    reasoning_model = os.getenv(
        REASONING_MODEL_ENV,
        "",
    ).strip()

    if api_key and reasoning_model:
        return api_key, reasoning_model

    print("\n首次使用 llm-cli，需要完成模型配置。")
    print(f"配置将保存到：{env_file}")
    print("请勿将该文件提交到 Git。\n")

    if not api_key:
        api_key = ask_required(
            "请输入 ARK API Key（输入内容不会显示）：",
            secret=True,
        )
        save_env_value(
            env_file,
            API_KEY_ENV,
            api_key,
        )

    if not reasoning_model:
        reasoning_model = ask_required(
            "请输入 ARK 模型端点 ID：",
        )
        save_env_value(
            env_file,
            REASONING_MODEL_ENV,
            reasoning_model,
        )

    print("配置保存成功。\n")

    return api_key, reasoning_model
async def run_interactive():
    configure_logging()
    api_key, reasoning_id = load_or_create_config()

    TOKENIZER_MODEL = "doubao-seed-evolving"
    KEEP_RECENT = 6

    client = ArkModelClient(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    conversation_service = ConversationService(
        client=client,
        reasoning_model=reasoning_id,
        tokenizer_model=TOKENIZER_MODEL,
        keep_recent=KEEP_RECENT,
    )

    session = select_session()

    try:
        while(True):
            print("1. 普通对话")
            print("2. 任务计划")
            choice = input("请选择会话模式: ").strip()

            if choice == "quit" or choice == "exit":
                break

            if choice == "1":
                content = input("请输入消息: ").strip()
                if not content:
                    print("消息不能为空")
                    continue

                if(content == "exit" or content == "quit"):
                    break

                if not session.name.strip():
                    session.name = create_session_name(
                        content,
                        session_id=session.session_id,
                    )

                response = await conversation_service.chat(
                    session=session,
                    content=content,
                )
                
                print(f"助手: {response.message.content}")

            elif choice == "2":
                content = input("请输入任务目标: ").strip()
                if not content:
                    print("任务目标不能为空")
                    continue

                if(content == "exit" or content == "quit"):
                    break

                if not session.name.strip():
                    session.name = create_session_name(
                        content,
                        session_id=session.session_id,
                    )

                plan = await conversation_service.create_task_plan(
                    session=session,
                    content=content,
                )
                validate_task_plan(plan)
                show_task_plan(plan)
                
        if session.messages:
            save_session(session)

    finally:
        await client.close()
        
if __name__ == "__main__":
    from ai_agent_learning.cli import main

    main()


