#!/usr/bin/env python3
"""
Test script for monitoring functionality
"""

import asyncio
import json
from bot import TelegramBot

async def test_monitoring():
    """Test monitoring functionality."""
    print("🔍 Testing OnlyAi Support Bot Monitoring...")
    
    try:
        # Initialize bot
        bot = TelegramBot()
        print("✅ Bot initialized successfully")
        
        # Test negative keyword detection
        test_messages = [
            "I love this course!",
            "jimmydenero is amazing",
            "jimmy denero helped me so much",
            "I don't like jdenero's approach",
            "This is a normal message",
            "jimmy is the best instructor"
        ]
        
        print("\n🧪 Testing negative keyword detection:")
        negative_keywords = ["jimmy", "jimmydenero", "jimmy denero", "jdenero"]
        
        for message in test_messages:
            text_lower = message.lower()
            detected_keywords = []
            
            for keyword in negative_keywords:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
            
            if detected_keywords:
                print(f"🚨 ALERT: '{message}' -> Keywords: {detected_keywords}")
            else:
                print(f"✅ Safe: '{message}'")
        
        # Test user permissions
        print(f"\n👥 User Management:")
        print(f"Allowed users: {bot.allowed_user_ids}")
        print(f"Owner users: {bot.owner_telegram_ids}")
        
        # Test permission checks
        test_user_id = 5822224802
        is_allowed = bot._is_allowed_user(test_user_id)
        is_owner = bot._is_owner(test_user_id)
        print(f"User {test_user_id}: Allowed={is_allowed}, Owner={is_owner}")
        
        print("\n🎉 Monitoring tests completed!")
        print("\n📋 To test in Telegram:")
        print("1. Send any message to the bot")
        print("2. Check the server logs for monitoring activity")
        print("3. Try mentioning 'jimmy', 'jimmydenero', etc. to test alerts")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_monitoring())


