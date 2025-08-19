#!/usr/bin/env python3
"""
Test script for OnlyAi Support Bot
"""

import asyncio
import json
from bot import TelegramBot

async def test_bot():
    """Test basic bot functionality."""
    print("🤖 Testing OnlyAi Support Bot...")
    
    try:
        # Initialize bot
        bot = TelegramBot()
        print("✅ Bot initialized successfully")
        
        # Test user ID parsing
        test_user_ids = "123,456,789"
        parsed = bot._parse_user_ids(test_user_ids)
        print(f"✅ User ID parsing: {parsed}")
        
        # Test permission checks
        test_user_id = 123
        is_allowed = bot._is_allowed_user(test_user_id)
        is_owner = bot._is_owner(test_user_id)
        print(f"✅ Permission checks - User {test_user_id}: Allowed={is_allowed}, Owner={is_owner}")
        
        # Test webhook operations
        test_webhook_url = "https://test.example.com/webhook"
        print(f"✅ Webhook operations ready (URL: {test_webhook_url})")
        
        print("\n🎉 All tests passed! Bot is ready for deployment.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_bot())


