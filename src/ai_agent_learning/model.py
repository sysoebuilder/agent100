from ai_agent_learning.schema import ModelRequest, ModelResponse, Message
from openai import OpenAI

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
