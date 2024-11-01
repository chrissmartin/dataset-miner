from dataclasses import dataclass
from typing import Union, TYPE_CHECKING


@dataclass
class CliArgs:
    source: str
    model: str
    output: str = "mined_dataset.json"
    debug: bool = False
    use_groq: bool = False
    remove_empty_columns: bool = False
    verify: bool = False


if TYPE_CHECKING:
    from langchain_groq import ChatGroq
    from langchain_ollama import OllamaLLM

ChatModel = Union["ChatGroq", "OllamaLLM"]


__all__ = ["CliArgs", "ChatModel"]
