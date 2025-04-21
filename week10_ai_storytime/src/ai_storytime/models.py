from pydantic import BaseModel, Field


class QuestionAnswer(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    answer: str = Field(..., description="The answer to the question.")
