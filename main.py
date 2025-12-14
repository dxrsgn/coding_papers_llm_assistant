import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from phoenix.otel import register

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.database import init_db

load_dotenv()

tracer_provider = register(
    project_name="aboba",
    auto_instrument=False,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
HTTPXClientInstrumentor().instrument()
tracer = tracer_provider.get_tracer("aboba")


# async cli app is not really needed
# but i made in that way for compatability with fastapi app
# and postgre db
async def async_input(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


async def run_chat_loop(app, config):
    while True:
        try:
            user_input = (await async_input("User: ")).strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        existing_state = await app.aget_state(config=config)
        messages = existing_state.values.get("messages", [])
        input_state = {
            **existing_state.values,
            "messages": messages,
            "user_query": user_input,
        }
        result = await app.ainvoke(AgentState(**input_state), config=config)
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

        print(final_response)


async def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    config = RunnableConfig(
        configurable={
            "thread_id": "main_thread",
            "llm_api_base": os.getenv("OPENAI_API_BASE"),
            "llm_api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("MODEL"),
        }
    )
    
    if database_url:
        try:
            db_url_for_checkpoint = database_url.replace("postgresql+psycopg://", "postgresql://")
            async with AsyncPostgresSaver.from_conn_string(db_url_for_checkpoint) as checkpointer:
                await checkpointer.setup()
                await init_db()
                app = build_graph(checkpointer=checkpointer)
                await run_chat_loop(app, config)
        except Exception as e:
            checkpointer = MemorySaver()
            app = build_graph(checkpointer=checkpointer)
            await run_chat_loop(app, config)
    else:
        checkpointer = MemorySaver()
        app = build_graph(checkpointer=checkpointer)
        await run_chat_loop(app, config)
    
    print("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())

