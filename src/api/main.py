from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent.graph import recommend
from src.agent.tools import recommend_by_title, recommend_by_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    query: str
    n: int = 10


class TitleRequest(BaseModel):
    title: str
    n: int = 10


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend/title")
def recommend_by_title_endpoint(req: TitleRequest):
    results = recommend_by_title(req.title, req.n)
    return {"recommendations": results.to_dict(orient="records")}


@app.post("/recommend/query")
def recommend_by_query_endpoint(req: QueryRequest):
    results = recommend_by_query(req.query, req.n)
    return {"recommendations": results.to_dict(orient="records")}


@app.post("/recommend/agent")
def recommend_agent_endpoint(req: QueryRequest):
    result = recommend(req.query, req.n)
    return result