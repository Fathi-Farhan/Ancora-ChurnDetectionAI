from fastapi import APIRouter
from schemas.ai_schemas import SentimentRequest, SentimentResponse

router = APIRouter(prefix="/api/v1", tags=["sentiment"])

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    # Placeholder handler mapping to README API format
    return SentimentResponse(
        label="NEGATIVE",
        score=0.87,
        reason="komplain pelayanan lambat dan kualitas tidak memuaskan",
        customer_id=request.customer_id
    )
