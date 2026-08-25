import asyncio
import os

from dotenv import load_dotenv

from ai_agent_learning.conversation_service import ConversationService
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.session import create_session_name, save_session, select_session
from ai_agent_learning.TaskPlan import show_task_plan, validate_task_plan

REASONING_ID = "ep-20260814170727-bsp2n"
TOKENIZER_MODEL = "doubao-seed-evolving"
KEEP_RECENT = 6

async def main():
    configure_logging()
    load_dotenv()

    client = ArkModelClient(
        api_key=os.environ["ARK_API_KEY"],
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    conversation_service = ConversationService(
        client=client,
        reasoning_model=REASONING_ID,
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
    asyncio.run(main())


