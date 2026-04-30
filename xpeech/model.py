from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models import Model
from abc import ABC, abstractmethod
from .utils.retry_util import create_retrying_client


class ModelWrapper(ABC):
    @property
    @abstractmethod
    def model(self) -> Model:
        pass


class OpenAIChatModelWrapper(ModelWrapper):
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self._model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                http_client=create_retrying_client(),
                base_url=base_url,
                api_key=api_key,
            ),
        )

    @property
    def model(self) -> OpenAIChatModel:
        return self._model
