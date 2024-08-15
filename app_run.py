from config import config

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=config.fastapi_port,
        reload=True,
        log_level=config.log_level,
        timeout_keep_alive=20,
        workers=config.num_workers,
    )
