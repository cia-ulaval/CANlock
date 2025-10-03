from json import load
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    def __init__(self):
        with open(".app_settings.json", "r", encoding="utf-8") as f:
            data = load(f)
        super().__init__(**data)

    folders: list[str] = Field(default_factory=lambda: [""])
    log_level: str = Field(default="INFO")
