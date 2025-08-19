"""
OnlyAi Support - Telegram Bot with RAG and Monitoring
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from bot import TelegramBot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OnlyAi Support Bot",
    description="Telegram bot with RAG and monitoring capabilities",
    version="1.0.0"
)

# Initialize bot
bot = TelegramBot()

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    model: str
    timestamp: str
    bot_status: str

@app.get("/")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="Online",
            model=os.getenv("OPENAI_MODEL", "gpt-5.0-thinking"),
            timestamp=datetime.utcnow().isoformat(),
            bot_status="Active"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/status")
async def bot_status():
    """Get detailed bot status."""
    return {
        "bot_status": "Active",
        "webhook_url": f"{os.getenv('WEBHOOK_BASE')}/webhook",
        "allowed_users": bot.allowed_user_ids,
        "owner_users": bot.owner_telegram_ids,
                        "monitoring_active": True,
                "rag_ready": True,  # RAG is now implemented
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Handle Telegram webhook updates."""
    try:
        # Parse the update
        update_data = await request.json()
        logger.info(f"Received webhook update: {update_data.get('update_id', 'unknown')}")
        
        # Process the update
        await bot.handle_update(update_data)
        
        return JSONResponse(content={"status": "ok"})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@app.post("/set-webhook")
async def set_webhook():
    """Set the Telegram webhook URL."""
    try:
        webhook_base = os.getenv("WEBHOOK_BASE")
        if not webhook_base:
            raise HTTPException(status_code=400, detail="WEBHOOK_BASE not configured")
        
        webhook_url = f"{webhook_base}/webhook"
        
        # Set webhook
        await bot.set_webhook(webhook_url)
        
        return JSONResponse(content={
            "status": "success",
            "webhook_url": webhook_url
        })
        
    except Exception as e:
        logger.error(f"Set webhook error: {e}")
        raise HTTPException(status_code=500, detail="Failed to set webhook")

@app.delete("/webhook")
async def delete_webhook():
    """Delete the Telegram webhook."""
    try:
        await bot.delete_webhook()
        return JSONResponse(content={"status": "success", "message": "Webhook deleted"})
        
    except Exception as e:
        logger.error(f"Delete webhook error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete webhook")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
