from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from phoenix.otel import register

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.database import init_db

from .models import MessageRequest, MessageResponse

load_dotenv()

tracer_provider = register(
    project_name="aboba",
    auto_instrument=False,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
HTTPXClientInstrumentor().instrument()
tracer = tracer_provider.get_tracer("aboba")

checkpointer = None
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, graph
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            db_url_for_checkpoint = database_url.replace("postgresql+psycopg://", "postgresql://")
            async with AsyncPostgresSaver.from_conn_string(db_url_for_checkpoint) as checkpointer:
                await checkpointer.setup()
                await init_db()
                graph = build_graph(checkpointer=checkpointer)
                yield
        except Exception as _:
            checkpointer = MemorySaver()
            graph = build_graph(checkpointer=checkpointer)
            yield
    else:
        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")
    try:
        use_db = bool(os.getenv("DATABASE_URL"))
        config = RunnableConfig(
            configurable={
                "thread_id": request.thread_id,
                "llm_api_base": os.getenv("OPENAI_API_BASE"),
                "llm_api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("MODEL"),
                "use_db": use_db,
            }
        )
        
        existing_state = await graph.aget_state(config=config)
        messages = existing_state.values.get("messages", [])
        input_state = {
            **existing_state.values,
            "messages": messages,
            "user_query": request.message,
        }
        
        result = await graph.ainvoke(AgentState(**input_state), config=config)
        messages = result.get("messages", [])
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        if isinstance(last_message.content, list):
            final_response = ""
            for part in last_message.content:
                if isinstance(part, str):
                    final_response += part
                elif isinstance(part, dict) and "text" in part:
                    final_response += part["text"]
        elif isinstance(last_message.content, str):
            final_response = last_message.content
        else:
            final_response = str(last_message.content)

        return MessageResponse(response=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
