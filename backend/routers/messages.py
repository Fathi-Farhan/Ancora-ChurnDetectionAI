from fastapi import APIRouter
from schemas.ai_schemas import MessageRequest, MessageResponse, MessageVariant

router = APIRouter(prefix="/api/v1", tags=["messages"])

@router.post("/messages/generate", response_model=MessageResponse)
async def generate_messages(request: MessageRequest):
    # Placeholder handler mapping to README API format
    return MessageResponse(
        variants=[
            MessageVariant(
                type="gentle_reminder", 
                message=f"Halo {request.customer_name}! Sudah sebulan lebih nih sejak terakhir {request.last_service} di sini. Rambut kamu masih oke kan? Kita tunggu kamu balik ya Kak 🌸"
            ),
            MessageVariant(
                type="promo_offer", 
                message=f"{request.customer_name}, spesial buat pelanggan setia — minggu ini ada diskon 20% untuk creambath. Mau booking slot-nya? Langsung balas aja ya!"
            ),
            MessageVariant(
                type="personal_touch", 
                message=f"{request.customer_name}, Bu Yani nanya nih — rambutmu gimana kabarnya? Udah lama banget ga keliatan. Kangen deh. Mampir yuk kapan-kapan!"
            )
        ]
    )
