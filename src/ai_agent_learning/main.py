from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.schema import ModelRequest, ModelResponse, Message, Session
from ai_agent_learning.session import save_session, load_session, clear_session, create_session_name, select_session
import os
from dotenv import load_dotenv

load_dotenv()

client = ArkModelClient(
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

session = select_session()
response_id = session.response_id or None
messages = session.messages

while(True):
    content = input("请输入消息: ").strip()
    
    if(content == "exit" or content == "quit"):
        break
    
    messages.append(Message(role="user", content=content))

    request = ModelRequest(
        messages=messages,
        model="ep-20260814170727-bsp2n",
        previous_response_id=response_id,
    )

    response = client.create_response(request)
    messages.append(response.message)
    response_id = response.response_id
    
    print(f"助手: {response.message.content}")

if messages:
    file_path = save_session(
        session=Session(
            name=create_session_name(messages[0].content),
            messages=messages,
            response_id=response_id,
        ),
    )
        


