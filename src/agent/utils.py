import re
import json
from typing import TypeVar, Type, Any, Generic
from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

T = TypeVar('T', bound=BaseModel)

def create_llm(reasoning=False, **kwargs):
    # TODO: find reasoning switch for models except kwargs
    # apparently passing "reasoning" kwarg into chatopenai client makes
    # it format responses from llm's server into openai's responses api format
    # which is incompatible with itmo's vllm instance and i guess many other vllm instances
    #if reasoning:
    #    conf = {"reasoning": {"enabled": True, "effort": "high"}}
    #else:
    #    conf = {"reasoning": {"enabled": False, "effort": "low"}}
    conf = {}
    return ChatOpenAI(**kwargs | conf, max_retries=3)

def normalize_message_content(msg: BaseMessage) -> BaseMessage:
    if isinstance(msg.content, list):
        text_parts = []
        for part in msg.content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif "text" in part:
                    text_parts.append(part["text"])
        normalized_content = "".join(text_parts) if text_parts else str(msg.content)
    else:
        normalized_content = str(msg.content) if msg.content else ""
    
    # some providers concat in text both reasoning and response
    response_match = re.search(r'<response>(.*?)</response>', normalized_content, re.DOTALL)
    if response_match:
        normalized_content = response_match.group(1).strip()
    
    if isinstance(msg, AIMessage):
        #if hasattr(msg, "tool_calls"):
        #    return AIMessage(content=None, tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else None)
        return AIMessage(content_blocks=[{"type": "text", "text": normalized_content}], tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else None)
    elif isinstance(msg, HumanMessage):
        return HumanMessage(content=normalized_content)
    elif isinstance(msg, SystemMessage):
        return SystemMessage(content=normalized_content)
    else:
        return msg

def clean_response(text: str) -> str:
    text = text.strip()
    text = text.replace('```json', '').replace('```', '')
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        return text[first_brace:last_brace+1]
    return text

def parse_with_retry(model_class: Type[T], raw_text: str) -> T:
    cleaned = clean_response(raw_text)
    try:
        data = json.loads(cleaned)
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError):
        pass
    
    try:
        data = json.loads(raw_text)
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError):
        pass
    
    raise ValueError(f"Failed to parse response after cleaning: {raw_text}")


# generic[T] and T as output typings is not neccesary, but it
# gives nice support for intellisense and type checking
class StructuredRetryRunnable(Runnable, Generic[T]):
    def __init__(
        self,
        llm: Runnable,
        model_class: Type[T],
        max_retries: int = 3
    ):
        super().__init__()
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=model_class)
        self.model_class = model_class
        self.max_retries = max_retries
    
    def invoke(self, input: Any, config: Any = None, **kwargs) -> T:
        # we dont really need sync, async would be enough
        raise NotImplementedError("This method is not implemented")
    
    async def ainvoke(self, input: Any, config: Any = None, **kwargs) -> T:
        messages = input if isinstance(input, list) else [input]
        
        #print(f"max retries: {self.max_retries}")
        for _ in range(self.max_retries):
            try:
                response = await self.llm.ainvoke(messages, config=config)
                # this monstrosity is too filter out reasoning messages, keep only text
                # otherwise if we would use just llm.with_structured_output, it would concat reasoning and text together
                if hasattr(response, 'content'):
                    if isinstance(response.content, list):
                        text_parts = []
                        for m in response.content:
                            if isinstance(m, str):
                                text_parts.append(m)
                            elif isinstance(m, dict):
                                if m.get("type") == "text":
                                    text = m.get("text") or m.get("content", [{"text": ""}])[0].get("text", "")
                                    text_parts.append(text)
                            else:
                                text_parts.append(str(m))
                        raw_text = "".join(text_parts)
                    else:
                        raw_text = str(response.content)
                else:
                    raw_text = str(response)
                
                try:
                    parsed = self.parser.parse(raw_text)
                    return parsed
                except (ValidationError, json.JSONDecodeError, ValueError):
                    cleaned_text = clean_response(raw_text)
                    parsed = parse_with_retry(self.model_class, cleaned_text)
                    return parsed
            except Exception as e:
                print(f"exception: {e}")
                pass
        
        #print(f"Couldnt parse")
        raise ValueError(f"Failed after max retries: {messages}")
