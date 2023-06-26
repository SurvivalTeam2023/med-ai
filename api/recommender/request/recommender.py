from pydantic import BaseModel, Field


class RequestCommenderRequest(BaseModel):
    user_id: str = Field(..., description="User id")
