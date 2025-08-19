#!/usr/bin/env python3
"""
Test script for RAG functionality
"""

import asyncio
from rag_service import RAGService

async def test_rag():
    """Test RAG functionality."""
    print("🤖 Testing RAG Service...")
    
    try:
        # Initialize RAG service
        rag = RAGService()
        print("✅ RAG service initialized")
        
        # Test question answering
        test_question = "What is OnlyAi?"
        print(f"\n🧪 Testing question: {test_question}")
        
        answer = await rag.answer_question(test_question, user_id=12345, chat_type="private")
        print(f"📝 Answer: {answer}")
        
        print("\n🎉 RAG test completed!")
        print("\n📋 To test in Telegram:")
        print("1. Send any question to the bot")
        print("2. The bot will use RAG to answer based on stored messages")
        print("3. Check /messages to see stored Q&A")
        
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_rag())
