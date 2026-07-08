from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.tools import recommend_by_title, recommend_by_query, rerank, get_rich_text, df

llm = ChatOllama(model="llama3.1:8b")


class AgentState(TypedDict):
    query: str
    intent: str
    title: str
    n: int
    recommendations: list
    explanation: str


def query_parser(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="""You are a movie recommendation assistant. 
        Given a user query, determine:
        1. intent: either 'by_title' (user mentions a specific movie) or 'by_query' (user describes what they want)
        2. title: if intent is by_title, extract the exact movie title, else return empty string
        
        Respond in this exact format:
        intent: by_title or by_query
        title: movie title or empty"""),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    lines = response.content.strip().split("\n")
    intent = lines[0].split(":")[-1].strip()
    title = lines[1].split(":")[-1].strip() if len(lines) > 1 else ""
    return {**state, "intent": intent, "title": title}


def vector_search(state: AgentState) -> AgentState:
    if state["intent"] == "by_title":
        results = recommend_by_title(state["title"], n=20)
    else:
        results = recommend_by_query(state["query"], n=20)
    return {**state, "recommendations": results.to_dict(orient="records")}


def re_ranker(state: AgentState) -> AgentState:
    recs = state["recommendations"]
    if not recs:
        return state
    # Build the query text for the cross-encoder
    if state["intent"] == "by_title":
        query_text = get_rich_text(state["title"])
    else:
        query_text = state["query"]
    # Attach rich_text to each candidate so the reranker can score them
    candidates = []
    for r in recs:
        m = df[df["title"] == r["title"]]
        rt = m.iloc[0]["rich_text"] if not m.empty else r["title"]
        candidates.append({**r, "rich_text": rt})
    reranked = rerank(query_text, candidates, top_n=state["n"])
    if reranked is None:
        # Cross-encoder unavailable (no GPU / insufficient RAM) — fall back to
        # Ollama generative reranking so the pipeline still runs anywhere.
        reranked = _ollama_rerank(query_text, candidates, state["n"])
    # strip rich_text before returning
    cleaned = [{k: v for k, v in c.items() if k != "rich_text"} for c in reranked]
    return {**state, "recommendations": cleaned}


def _ollama_rerank(query_text: str, candidates: list, n: int) -> list:
    """Fallback reranker: ask the stock Llama (via Ollama) to reorder the
    candidate titles by relevance to the query. Best-effort — if parsing the
    LLM output fails, returns the candidates in their original retrieval order."""
    titles = [c["title"] for c in candidates]
    messages = [
        SystemMessage(content=(
            "You are a movie reranker. Given a query and a numbered list of "
            "candidate movies, return the numbers reordered from most to least "
            "relevant, comma-separated, e.g. '3,1,4,2'. Return ONLY the numbers."
        )),
        HumanMessage(content=(
            f"Query: {query_text}\n\nCandidates:\n" +
            "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))
        )),
    ]
    try:
        resp = llm.invoke(messages).content.strip()
        idxs = [int(x) - 1 for x in resp.replace(" ", "").split(",")]
        seen, order = set(), []
        for i in idxs:
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                order.append(i)
        # append any candidates the LLM dropped, preserving retrieval order
        for i in range(len(candidates)):
            if i not in seen:
                order.append(i)
        return [candidates[i] for i in order[:n]]
    except Exception:
        return candidates[:n]


def explainer(state: AgentState) -> AgentState:
    recs = state["recommendations"][:5]
    titles = [r["title"] for r in recs]
    messages = [
        SystemMessage(content="You are a movie recommendation assistant. Given a user query and a list of recommended movies, write a brief 2-3 sentence explanation of why these movies were recommended."),
        HumanMessage(content=f"Query: {state['query']}\nRecommended movies: {', '.join(titles)}")
    ]
    response = llm.invoke(messages)
    return {**state, "explanation": response.content.strip()}


workflow = StateGraph(AgentState)
workflow.add_node("query_parser", query_parser)
workflow.add_node("vector_search", vector_search)
workflow.add_node("re_ranker", re_ranker)
workflow.add_node("explainer", explainer)

workflow.set_entry_point("query_parser")
workflow.add_edge("query_parser", "vector_search")
workflow.add_edge("vector_search", "re_ranker")
workflow.add_edge("re_ranker", "explainer")
workflow.add_edge("explainer", END)

graph = workflow.compile()


def recommend(query: str, n: int = 10) -> dict:
    result = graph.invoke({"query": query, "intent": "", "title": "", "n": n, "recommendations": [], "explanation": ""})
    return {
        "recommendations": result["recommendations"],
        "explanation": result["explanation"]
    }