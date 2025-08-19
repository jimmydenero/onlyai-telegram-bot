#!/usr/bin/env python3
"""
FastAPI application for the Telegram bot
"""

import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json
from datetime import datetime

from bot import TelegramBot
from rag_service import RAGService

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OnlyAi Telegram Bot API")

# Initialize bot
bot = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize bot on startup."""
    global bot, rag_service
    try:
        bot = TelegramBot()
        rag_service = RAGService()
        logger.info("✅ Bot initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize bot: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "OnlyAi Telegram Bot API", "timestamp": datetime.utcnow().isoformat()}

@app.get("/status")
async def status():
    """Get bot status."""
    if not bot:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    return {
        "bot_status": "Active",
        "webhook_url": os.getenv("WEBHOOK_BASE", "https://your-ngrok-url.ngrok.io/webhook"),
        "allowed_users": bot.allowed_user_ids,
        "owner_users": bot.owner_telegram_ids,
        "monitoring_active": True,
        "rag_ready": rag_service is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/webhook")
async def webhook(request: Request):
    """Handle Telegram webhook."""
    try:
        update_data = await request.json()
        update_id = update_data.get("update_id", "unknown")
        logger.info(f"Received webhook update: {update_id}")
        
        if bot:
            await bot.handle_update(update_data)
            return {"status": "ok"}
        else:
            logger.error("Bot not initialized")
            return {"status": "error", "message": "Bot not initialized"}
            
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/set-webhook")
async def set_webhook():
    """Set webhook URL."""
    try:
        webhook_url = os.getenv("WEBHOOK_BASE", "") + "/webhook"
        if bot:
            await bot.set_webhook(webhook_url)
            return {"status": "ok", "webhook_url": webhook_url}
        else:
            return {"status": "error", "message": "Bot not initialized"}
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.delete("/webhook")
async def delete_webhook():
    """Delete webhook."""
    try:
        if bot:
            await bot.delete_webhook()
            return {"status": "ok"}
        else:
            return {"status": "error", "message": "Bot not initialized"}
    except Exception as e:
        logger.error(f"Failed to delete webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/debug/rag")
async def debug_rag():
    """Debug endpoint to test RAG service directly."""
    try:
        if not rag_service:
            return {"status": "error", "message": "RAG service not initialized"}
        
        # Test basic RAG functionality
        test_question = "What is OnlyAi?"
        user_id = 12345
        
        logger.info("Testing RAG service...")
        
        # Test configuration
        config_status = {
            "openai_api_key": "present" if os.getenv("OPENAI_API_KEY") else "missing",
            "openai_model": rag_service.openai_model,
            "embed_model": rag_service.embed_model
        }
        
        # Test knowledge base
        try:
            kb_results = rag_service.search_knowledge_base(test_question, limit=1)
            kb_status = f"Found {len(kb_results)} results"
        except Exception as e:
            kb_status = f"Error: {str(e)}"
        
        # Test messages
        try:
            messages = rag_service.get_relevant_messages(test_question, limit=1)
            messages_status = f"Found {len(messages)} messages"
        except Exception as e:
            messages_status = f"Error: {str(e)}"
        
        # Test OpenAI call with simple test
        try:
            answer = await rag_service.simple_test(test_question, user_id)
            openai_status = "Success"
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
        except Exception as e:
            openai_status = f"Error: {str(e)}"
            answer_preview = None
        
        return {
            "status": "ok",
            "config": config_status,
            "knowledge_base": kb_status,
            "messages": messages_status,
            "openai": openai_status,
            "answer_preview": answer_preview,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Debug RAG error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/debug/simple")
async def debug_simple():
    """Simple test endpoint to isolate OpenAI API call."""
    try:
        import openai
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "OpenAI API key missing"}
        
        # Simple test call
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "status": "success",
            "answer": answer,
            "model": "gpt-4",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/debug/kb")
async def debug_knowledge_base():
    """Debug endpoint to inspect knowledge base contents."""
    try:
        if not rag_service:
            return {"status": "error", "message": "RAG service not initialized"}
        
        import sqlite3
        from pathlib import Path
        
        kb_path = Path("./knowledge_base/knowledge_base.db")
        if not kb_path.exists():
            return {"status": "error", "message": "Knowledge base database not found"}
        
        conn = sqlite3.connect(kb_path)
        cursor = conn.cursor()
        
        # Get all documents
        cursor.execute("SELECT id, title, content, category, tags, source_type, source_url, created_at FROM documents ORDER BY created_at DESC")
        documents = cursor.fetchall()
        
        # Get document count
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        
        conn.close()
        
        # Format results
        docs = []
        for doc in documents:
            docs.append({
                "id": doc[0],
                "title": doc[1],
                "content_preview": doc[2][:200] + "..." if len(doc[2]) > 200 else doc[2],
                "content_length": len(doc[2]),
                "category": doc[3],
                "tags": doc[4],
                "source_type": doc[5],
                "source_url": doc[6],
                "created_at": doc[7]
            })
        
        return {
            "status": "ok",
            "total_documents": count,
            "documents": docs,
            "database_path": str(kb_path),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Debug KB error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/debug/add-test-doc")
async def debug_add_test_document():
    """Test endpoint to manually add a document to the knowledge base."""
    try:
        if not rag_service:
            return {"status": "error", "message": "RAG service not initialized"}
        
        # Test document
        test_title = "Test Document"
        test_content = "This is a test document to verify database insertion works."
        test_category = "test"
        test_tags = ["test", "debug"]
        
        # Try to add the document
        success = rag_service.add_document(
            title=test_title,
            content=test_content,
            category=test_category,
            tags=test_tags
        )
        
        if success:
            return {
                "status": "success",
                "message": "Test document added successfully",
                "title": test_title,
                "content_length": len(test_content),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to add test document",
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Debug add test doc error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
