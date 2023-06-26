from pydantic import BaseModel, Field


class RecommenderResponse(BaseModel):
    response: str = Field(..., description="Response list data")
