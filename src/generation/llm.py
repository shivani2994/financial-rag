"""
The LLM interface generation runs against.

`LLMClient` is the seam: everything else in this module (prompting,
citation enforcement, the refusal gate) only ever calls `.generate()`. The
concrete implementation below runs a local model through Ollama, but
swapping in a hosted model later means writing one new class against this
same interface -- no changes anywhere else.
"""

from abc import ABC, abstractmethod

from langchain_ollama import OllamaLLM

from config.settings import settings


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's raw text completion for `prompt`."""


class OllamaLLMClient(LLMClient):
    def __init__(self, model: str, base_url: str):
        self._llm = OllamaLLM(model=model, base_url=base_url, temperature=0.0)

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt)


def get_llm_client() -> LLMClient:
    return OllamaLLMClient(model=settings.ollama_model, base_url=settings.ollama_base_url)
