from pydantic import BaseModel
from typing import List

class SentimentRequest(BaseModel):
    text: str
    customer_id: str

class SentimentResponse(BaseModel):
    label: str
    score: float
    reason: str
    customer_id: str

class MessageRequest(BaseModel):
    customer_name: str
    business_type: str
    business_name: str
    last_service: str
    recency_days: int
    churn_score: float

class MessageVariant(BaseModel):
    type: str
    message: str

class MessageResponse(BaseModel):
    variants: List[MessageVariant]
