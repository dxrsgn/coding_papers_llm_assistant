from dotenv import load_dotenv
import os
import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.graph import build_graph
from src.state import AgentState

from phoenix.otel import register
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

tracer_provider = register(
    project_name="aboba",
    auto_instrument=False,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
HTTPXClientInstrumentor().instrument()
tracer = tracer_provider.get_tracer("aboba")


async def async_input(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


async def main() -> None:
    app = build_graph()
    config = RunnableConfig(
        configurable={
            "thread_id": "main_thread",
            "llm_api_base": os.getenv("OPENAI_API_BASE"),
            "llm_api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("MODEL"),
        }
    )
    while True:
        try:
            user_input = (await async_input("User: ")).strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        existing_state = app.get_state(config=config)
        # merge existing state (shared memort context) with new input
        messages = existing_state.values.get("messages", []) #+ [HumanMessage(content=user_input)]
        input_state =  {
            **existing_state.values,
            "messages": messages,
            "user_query": user_input,
        }
        result = await app.ainvoke(AgentState(**input_state), config=config)
        messages = result.get("messages", [])
        final_response = ""
        for message in reversed(messages):
            if message.type == "ai":
                final_response = message.content
                break
        print(final_response)
    print("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())

