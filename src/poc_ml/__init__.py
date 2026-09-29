def main() -> None:
    import uvicorn

    uvicorn.run("poc_ml.api:app", host="127.0.0.1", port=8100, reload=True)
