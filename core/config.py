import os

from pydantic import BaseSettings


class Config(BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    JWT_SECRET_KEY: str = "fastapi"
    JWT_ALGORITHM: str = "HS256"
    SENTRY_SDN: str = None
    CELERY_BROKER_URL: str = "amqp://user:bitnami@localhost:5672/"
    CELERY_BACKEND_URL: str = "redis://:password123@localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    DB_USER: str = os.environ.get("DB_USER", "root")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "JCpulWbTp5")
    DB_HOST: str = os.environ.get("DB_HOST", "14.225.204.74")
    DB_PORT: str = os.environ.get("DB_PORT", "3306")
    DB_DATABASE: str = os.environ.get("DB_DATABASE", "med-be")


class DevelopmentConfig(Config):
    WRITER_DB_URL: str = "mysql+aiomysql://root:fastapi@db:3306/fastapi"
    READER_DB_URL: str = "mysql+aiomysql://root:fastapi@db:3306/fastapi"
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379


class LocalConfig(Config):
    WRITER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi"
    READER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi"


class ProductionConfig(Config):
    DEBUG: str = False
    WRITER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/prod"
    READER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/prod"


def get_config():
    env = os.getenv("ENV", "local")
    config_type = {
        "dev": DevelopmentConfig(),
        "local": LocalConfig(),
        "prod": ProductionConfig(),
    }
    return config_type[env]


config: Config = get_config()
