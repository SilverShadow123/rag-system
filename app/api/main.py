from fastapi import FastAPI

app = FastAPI(
    title="Intelligent RAG System",
    version="0.1.0",
    description="Document retrieval and grounded question answering",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "rag-system",
    }
