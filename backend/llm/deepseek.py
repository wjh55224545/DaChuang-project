from __future__ import annotations
from functools import lru_cache
from langchain_openai import ChatOpenAI
from backend.config import get_settings


@lru_cache
def get_llm(streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        streaming=streaming,
        temperature=0.7,
        max_tokens=2048,
    )
