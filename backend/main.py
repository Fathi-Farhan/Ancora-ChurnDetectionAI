from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sentiment, messages

app = FastAPI(
    title="Ancora Churn Detection AI API",
    description="AI-Powered Customer Retention Platform for Indonesian MSMEs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router)
app.include_router(messages.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Ancora Churn Detection AI API"}

# Job endpoint placeholder
@app.post("/api/v1/jobs/churn-detection", tags=["jobs"])
async def run_churn_detection():
    return {"status": "success", "message": "Batch churn detection job triggered successfully"}
