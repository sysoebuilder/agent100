from ai_agent_learning.schema import ModelRequest, ModelResponse, Message, TaskPlan
from ai_agent_learning.retry import retry_async
from openai import AsyncOpenAI
import httpx

CONTEXT_LIMIT = 1_000_000

class ModelClient:
    def __init__(self):
        pass
    
    def create_response(self, requset: ModelRequest):
        pass

class ArkModelClient(ModelClient):
    def __init__(
            self,
            api_key: str,
            base_url: str,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                30.0,
                connect=5.0,
            ),
        )

    async def create_response(self, requset: ModelRequest) -> ModelResponse:
        response = await self._client.responses.create(
            model=requset.model,
            store=True,
            input=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in requset.messages
            ],
            )

        if(response.output_text is None):
            raise ValueError("response.output_text is None")
        
        return ModelResponse(
            message=Message(role="assistant", content=response.output_text),
            response_id=response.id,
        )

    async def create_task_plan(self, requset: ModelRequest) -> TaskPlan:
        response = await self._client.responses.parse(
            model=requset.model,
            store=True,
            input=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in requset.messages
            ],
            text_format=TaskPlan,
            )
        
        if(response.output_parsed is None):
            raise ValueError("response.output_parsed is None")

        return response.output_parsed

    async def tokenization(self, messages: list[Message], model: str) -> int:
        async def create_request() -> httpx.Response:
            response = await self._http_client.post(
                "tokenization",
                json={
                    "model": model,
                    "text": [
                        f"{message.role}: {message.content}"
                        for message in messages
                    ],
                },
            )
            response.raise_for_status()
            return response

        response = await retry_async(create_request, max_attempts=3)
        
        result = response.json()
        total_tokens = sum(
            item["total_tokens"]
            for item in result["data"]
        )
        return total_tokens


    async def is_context_limit(self, messages: list[Message], model: str) -> bool:
        total_tokens = await self.tokenization(messages, model)
        return total_tokens > CONTEXT_LIMIT*0.8

    async def compact_context(self, messages: list[Message], model: str) -> Message:
        compression_input = [
            {
                "role": "system",
                "content": (
                    "你的任务是压缩随后提供的历史对话。"
                    "请保留重要事实、用户偏好、已有结论、"
                    "未完成任务和必要细节。"
                    "不要回答历史问题，只输出摘要。"
                ),
            },
            *[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        ]

        response = await self._client.responses.create(
            model=model,
            store=True,
            input=compression_input, # type: ignore
            )
        if(response.output_text is None):
            raise ValueError("response.output_text is None")

        return Message(role="system", content=f"历史对话摘要: {response.output_text}")

    async def close(self) -> None:
        await self._http_client.aclose()
        await self._client.close()