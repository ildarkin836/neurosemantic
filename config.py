from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NeuroSemantic"
    triton_port: int
    triton_ip: str
    log_level: str
    fastapi_port: int
    num_workers: int
    det_threshold: float
    iou_threshold: float

    class Config:
        env_file = "/opt/neurosemantic/.env"


config = Settings()
