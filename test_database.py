#!/usr/bin/env python3
"""
Test script for database functionality
"""

import asyncio
from models import SessionLocal, Message, MonitoredGroup
from datetime import datetime

def test_database():
    """Test database functionality."""
    print("🔍 Testing Database Functionality...")
    
    try:
        db = SessionLocal()
        
        # Check if tables exist
        print("✅ Database connection successful")
        
        # Count messages
        message_count = db.query(Message).count()
        print(f"📊 Total messages stored: {message_count}")
        
        # Count monitored groups
        group_count = db.query(MonitoredGroup).filter(MonitoredGroup.is_active == True).count()
        print(f"🎯 Monitored groups: {group_count}")
        
        # Show recent messages
        recent_messages = db.query(Message).order_by(Message.timestamp.desc()).limit(5).all()
        if recent_messages:
            print("\n📝 Recent messages:")
            for msg in recent_messages:
                status = "🎯" if msg.is_monitored else "📝"
                print(f"  {status} {msg.chat_type} (ID: {msg.chat_id}) - User: {msg.user_id}")
                print(f"     Time: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     Text: {msg.text[:50]}{'...' if len(msg.text) > 50 else ''}")
                print()
        
        # Show monitored groups
        monitored_groups = db.query(MonitoredGroup).filter(MonitoredGroup.is_active == True).all()
        if monitored_groups:
            print("🎯 Monitored groups:")
            for group in monitored_groups:
                print(f"  • {group.chat_title} (ID: {group.chat_id})")
                print(f"    Added: {group.added_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("🎯 No groups are currently being monitored")
        
        print("\n🎉 Database test completed!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    test_database()
