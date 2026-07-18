from __future__ import annotations
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/psych.db"
    deepseek_api_key: str = "sk-your-deepseek-api-key"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # 智谱AI (GLM) 配置 — 用于反馈模块智能通知生成
    zhipu_api_key: str = "d36c6e6b755b453f8b9f52f6541c1136.Pr8uW3uxCXsc923J"
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4-flash"

    obs_endpoint: str = "https://obs.cn-north-4.myhuaweicloud.com"
    obs_bucket: str = "psych-monitor"
    obs_ak: str = ""
    obs_sk: str = ""
    upload_dir: str = "./data/uploads"
    camera_dir: str = "./data/camera"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
