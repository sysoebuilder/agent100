from ai_agent_learning.schema import ModelRequest, ModelResponse, Message
from openai import OpenAI
import httpx
import os

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
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def create_response(self, requset: ModelRequest) -> ModelResponse:
        response = self._client.responses.create(
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

    def tokenization(self, messages: list[Message], model: str) -> int:
        response = httpx.post(
            f"{self._base_url}/tokenization",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "text": [
                    f"{message.role}: {message.content}"
                    for message in messages
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        total_tokens = sum(
            item["total_tokens"]
            for item in result["data"]
        )
        return total_tokens


    def is_context_limit(self, messages: list[Message], model: str) -> bool:
        total_tokens = self.tokenization(messages, model)
        return total_tokens > CONTEXT_LIMIT*0.8

    def compact_context(self, messages: list[Message], model: str) -> Message:
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

        response = self._client.responses.create(
            model=model,
            store=True,
            input=compression_input, # type: ignore
            )
        if(response.output_text is None):
            raise ValueError("response.output_text is None")

        return Message(role="system", content=f"历史对话摘要: {response.output_text}")
        