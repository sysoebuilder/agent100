from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.schema import ModelRequest, ModelResponse, Message, Session
from ai_agent_learning.session import save_session, load_session, clear_session, create_session_name, select_session
import os
from dotenv import load_dotenv

REASONING_ID = "ep-20260814170727-bsp2n"
TOKENIZER_MODEL = "doubao-seed-evolving"
KEEP_RECENT = 6

load_dotenv()

client = ArkModelClient(
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

session = select_session()
messages = session.messages

while(True):
    content = input("请输入消息: ").strip()
    if not content:
        print("消息不能为空")
        continue

    if(content == "exit" or content == "quit"):
        break

    if not session.name.strip():
        session.name = create_session_name(content)
    
    messages.append(Message(role="user", content=content))

    if(client.is_context_limit(messages, TOKENIZER_MODEL)):
        old_messages = messages[:-KEEP_RECENT]
        recent_messages = messages[-KEEP_RECENT:]

        if old_messages:
            summary = client.compact_context(old_messages, REASONING_ID)
            messages[:] = [summary, *recent_messages]
    
    request = ModelRequest(
        messages=messages,
        model=REASONING_ID,
        previous_response_id=None,
    )

    response = client.create_response(request)
    messages.append(response.message)
    session.response_id = response.response_id
    
    print(f"助手: {response.message.content}")

if messages:
    file_path = save_session(session)
        


