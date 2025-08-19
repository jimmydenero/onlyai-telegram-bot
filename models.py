#!/usr/bin/env python3
"""
Database models for Telegram AI Agent
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class Message(Base):
    """Model for storing Telegram messages."""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, nullable=False)  # Telegram message ID
    chat_id = Column(BigInteger, nullable=False)     # Telegram chat ID
    user_id = Column(BigInteger, nullable=False)     # Telegram user ID
    username = Column(String(100), nullable=True)    # Telegram username
    first_name = Column(String(100), nullable=True)  # User's first name
    last_name = Column(String(100), nullable=True)   # User's last name
    chat_type = Column(String(20), nullable=False)   # 'private', 'group', 'supergroup', 'channel'
    text = Column(Text, nullable=False)              # Message text content
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_monitored = Column(Boolean, default=False)    # Whether this message is from monitored group
    is_owner = Column(Boolean, default=False)        # Whether sender is an owner
    
    def __repr__(self):
        return f"<Message(id={self.id}, chat_id={self.chat_id}, user_id={self.user_id}, text='{self.text[:50]}...')>"

class MonitoredGroup(Base):
    """Model for storing monitored group information."""
    __tablename__ = 'monitored_groups'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, unique=True, nullable=False)  # Telegram chat ID
    chat_title = Column(String(255), nullable=True)            # Group name
    added_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<MonitoredGroup(chat_id={self.chat_id}, title='{self.chat_title}')>"

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/telegram_ai_agent.db")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create tables
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
