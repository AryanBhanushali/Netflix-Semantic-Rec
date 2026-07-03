from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.tools import recommend_by_title, recommend_by_query

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
    recs = state["recommendations"][:20]
    titles = "\n".join([f"{i+1}. {r['title']} (rating: {r['vote_average']})" for i, r in enumerate(recs)])
    messages = [
        SystemMessage(content="""You are a movie recommendation re-ranker. Given a user query and a list of candidate movies, select the most relevant movies and return ONLY their numbers, comma-separated, in order of relevance. Nothing else."""),
        HumanMessage(content=f"Query: {state['query']}\n\nCandidates:\n{titles}")
    ]
    response = llm.invoke(messages)
    try:
        indices = [int(x.strip()) - 1 for x in response.content.strip().split(",")]
        reranked = [recs[i] for i in indices if 0 <= i < len(recs)]
        return {**state, "recommendations": reranked[:state["n"]]}
    except:
        return {**state, "recommendations": recs[:state["n"]]}


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