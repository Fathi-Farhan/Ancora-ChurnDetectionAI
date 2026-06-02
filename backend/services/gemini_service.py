class GeminiService:
    async def analyze_sentiment(self, text: str) -> dict:
        # Placeholder for Gemini API sentiment analysis
        return {
            "label": "NEUTRAL",
            "score": 0.5,
            "reason": "Sentiment analysis placeholder",
        }

    async def generate_retention_messages(self, customer_name: str, business_type: str) -> list:
        # Placeholder for Gemini API message generation
        return [
            {"type": "gentle_reminder", "message": f"Halo {customer_name}! Kami merindukanmu."},
            {"type": "promo_offer", "message": f"Diskon khusus untuk {customer_name}!"},
            {"type": "personal_touch", "message": f"Apa kabar {customer_name}? Mampir yuk!"}
        ]
