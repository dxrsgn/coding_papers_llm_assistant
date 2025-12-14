from pydantic import BaseModel


class MessageRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"


class MessageResponse(BaseModel):
    response: str
