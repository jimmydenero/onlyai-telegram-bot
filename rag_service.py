#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) service for the Telegram bot
"""

import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import sqlite3
from pathlib import Path
import re
import tempfile
import subprocess

# Optional imports with fallbacks
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyPDF2 not available - PDF processing disabled")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("python-docx not available - DOCX processing disabled")

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("BeautifulSoup not available - web scraping disabled")

from models import SessionLocal, Message

load_dotenv()
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")  # Use environment variable or default to gpt-4
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

class RAGService:
    """RAG service for answering questions using stored messages and knowledge base."""
    
    def __init__(self):
        self.openai_model = OPENAI_MODEL
        self.embed_model = EMBED_MODEL
        self.kb_dir = Path("./knowledge_base")
        self.kb_dir.mkdir(exist_ok=True)
        self.kb_db_path = self.kb_dir / "knowledge_base.db"
        self._init_knowledge_base()
        
    def _init_knowledge_base(self):
        """Initialize the knowledge base database."""
        try:
            conn = sqlite3.connect(self.kb_db_path)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    tags TEXT,
                    source_type TEXT DEFAULT 'manual',
                    source_url TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create embeddings table for semantic search
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    chunk_text TEXT NOT NULL,
                    embedding BLOB,
                    chunk_index INTEGER,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Knowledge base initialized")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
    
    def scrape_website(self, url: str) -> Optional[str]:
        """Scrape content from a website."""
        if not BEAUTIFULSOUP_AVAILABLE:
            logger.error("BeautifulSoup not available - web scraping disabled")
            return None
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else "Scraped Content"
            
            return title_text, text
            
        except Exception as e:
            logger.error(f"Error scraping website {url}: {e}")
            return None
    
    def process_pdf(self, file_path: str) -> Optional[str]:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not available - PDF processing disabled")
            return None
            
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return None
    
    def process_docx(self, file_path: str) -> Optional[str]:
        """Extract text from DOCX file."""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not available - DOCX processing disabled")
            return None
            
        try:
            doc = docx.Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return None
    
    def process_txt(self, file_path: str) -> Optional[str]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
                
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {e}")
            return None
    
    def process_video(self, file_path: str) -> Optional[str]:
        """Extract audio and transcribe video content."""
        try:
            # First, extract audio from video
            audio_path = file_path.replace('.mp4', '.wav').replace('.mov', '.wav').replace('.avi', '.wav')
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', file_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', 
                audio_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
            
            # Transcribe audio using OpenAI Whisper
            with open(audio_path, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file,
                    response_format="text"
                )
            
            # Clean up audio file
            os.remove(audio_path)
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error processing video {file_path}: {e}")
            return None
    
    def add_document_from_file(self, file_path: str, title: str = None, category: str = "general", tags: List[str] = None) -> bool:
        """Add a document from a file to the knowledge base."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Determine file type and process accordingly
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                content = self.process_pdf(str(file_path))
            elif file_extension == '.docx':
                content = self.process_docx(str(file_path))
            elif file_extension == '.txt':
                content = self.process_txt(str(file_path))
            elif file_extension in ['.mp4', '.mov', '.avi']:
                content = self.process_video(str(file_path))
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return False
            
            if not content:
                logger.error(f"Could not extract content from {file_path}")
                return False
            
            # Use filename as title if not provided
            if not title:
                title = file_path.stem
            
            # Add to knowledge base
            success = self.add_document(
                title=title,
                content=content,
                category=category,
                tags=tags,
                source_type="file",
                file_path=str(file_path)
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding document from file: {e}")
            return False
    
    def add_document_from_url(self, url: str, title: str = None, category: str = "general", tags: List[str] = None) -> bool:
        """Add a document from a URL to the knowledge base."""
        try:
            # Scrape website content
            result = self.scrape_website(url)
            
            if not result:
                logger.error(f"Could not scrape content from {url}")
                return False
            
            scraped_title, content = result
            
            # Use scraped title if not provided
            if not title:
                title = scraped_title
            
            # Add to knowledge base
            success = self.add_document(
                title=title,
                content=content,
                category=category,
                tags=tags,
                source_type="url",
                source_url=url
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding document from URL: {e}")
            return False
    
    def add_document(self, title: str, content: str, category: str = "general", tags: List[str] = None, 
                    source_type: str = "manual", source_url: str = None, file_path: str = None) -> bool:
        """Add a document to the knowledge base."""
        try:
            conn = sqlite3.connect(self.kb_db_path)
            cursor = conn.cursor()
            
            # Insert document
            cursor.execute('''
                INSERT INTO documents (title, content, category, tags, source_type, source_url, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (title, content, category, json.dumps(tags or []), source_type, source_url, file_path))
            
            document_id = cursor.lastrowid
            
            # Split content into chunks and store embeddings
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                # For now, store chunks without embeddings (can be enhanced later)
                cursor.execute('''
                    INSERT INTO embeddings (document_id, chunk_text, chunk_index)
                    VALUES (?, ?, ?)
                ''', (document_id, chunk, i))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Added document: {title} (source: {source_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for better retrieval."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def search_knowledge_base(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant documents."""
        try:
            conn = sqlite3.connect(self.kb_db_path)
            cursor = conn.cursor()
            
            # Check if source_type column exists
            cursor.execute("PRAGMA table_info(documents)")
            columns = [column[1] for column in cursor.fetchall()]
            has_source_type = 'source_type' in columns
            has_source_url = 'source_url' in columns
            
            # Simple keyword-based search (can be enhanced with semantic search)
            query_terms = query.lower().split()
            
            # Build query based on available columns
            if has_source_type and has_source_url:
                cursor.execute('''
                    SELECT d.id, d.title, d.content, d.category, d.tags, d.source_type, d.source_url
                    FROM documents d
                    WHERE LOWER(d.title) LIKE ? OR LOWER(d.content) LIKE ?
                    ORDER BY d.updated_at DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit * 3))
            else:
                # Fallback for older schema
                cursor.execute('''
                    SELECT d.id, d.title, d.content, d.category, d.tags
                    FROM documents d
                    WHERE LOWER(d.title) LIKE ? OR LOWER(d.content) LIKE ?
                    ORDER BY d.updated_at DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit * 3))
            
            results = []
            seen_docs = set()
            
            for row in cursor.fetchall():
                if has_source_type and has_source_url:
                    doc_id, title, content, category, tags, source_type, source_url = row
                else:
                    doc_id, title, content, category, tags = row
                    source_type = 'manual'
                    source_url = None
                
                if doc_id not in seen_docs and len(results) < limit:
                    # Score based on keyword matches
                    score = 0
                    content_lower = content.lower()
                    for term in query_terms:
                        if term in content_lower:
                            score += 1
                    
                    if score > 0:
                        results.append({
                            'id': doc_id,
                            'title': title,
                            'content': content[:500] + '...' if len(content) > 500 else content,
                            'category': category,
                            'tags': json.loads(tags) if tags else [],
                            'source_type': source_type,
                            'source_url': source_url,
                            'score': score
                        })
                        seen_docs.add(doc_id)
            
            conn.close()
            return sorted(results, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def get_relevant_messages(self, query: str, limit: int = 10, hours_back: int = 24) -> List[Message]:
        """Get relevant messages from the database based on semantic similarity."""
        try:
            # For now, use simple keyword matching and recency
            # TODO: Implement proper semantic search with embeddings
            
            db = SessionLocal()
            
            # Get messages from the last N hours
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Simple keyword-based retrieval
            query_lower = query.lower()
            relevant_messages = []
            
            # Get recent messages
            recent_messages = db.query(Message).filter(
                Message.timestamp >= cutoff_time,
                Message.text.isnot(None)
            ).order_by(Message.timestamp.desc()).limit(50).all()
            
            # Score messages based on keyword overlap
            scored_messages = []
            for msg in recent_messages:
                score = 0
                msg_lower = msg.text.lower()
                
                # Exact keyword matches
                for word in query_lower.split():
                    if word in msg_lower:
                        score += 1
                
                # Recency bonus
                hours_ago = (datetime.utcnow() - msg.timestamp).total_seconds() / 3600
                recency_bonus = max(0, 1 - (hours_ago / 24))  # Higher score for newer messages
                score += recency_bonus
                
                if score > 0:
                    scored_messages.append((msg, score))
            
            # Sort by score and return top results
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            relevant_messages = [msg for msg, score in scored_messages[:limit]]
            
            logger.info(f"Found {len(relevant_messages)} relevant messages for query: {query[:50]}...")
            return relevant_messages
            
        except Exception as e:
            logger.error(f"Error getting relevant messages: {e}")
            return []
        finally:
            db.close()
    
    def format_context(self, messages: List[Message]) -> str:
        """Format messages into context for the LLM."""
        if not messages:
            return "No relevant context found."
        
        context_parts = []
        for msg in messages:
            user_display = msg.username or f"{msg.first_name} {msg.last_name or ''}".strip() or f"User {msg.user_id}"
            timestamp = msg.timestamp.strftime("%H:%M")
            context_parts.append(f"[{timestamp}] {user_display}: {msg.text}")
        
        return "\n".join(context_parts)
    
    def format_kb_context(self, kb_results: List[Dict[str, Any]]) -> str:
        """Format knowledge base results into context."""
        if not kb_results:
            return ""
        
        context_parts = ["Knowledge Base:"]
        for result in kb_results:
            source_info = f" (from {result['source_type']})" if result.get('source_type') else ""
            context_parts.append(f"[{result['title']}{source_info}] {result['content']}")
        
        return "\n".join(context_parts)
    
    async def answer_question(self, question: str, user_id: int, chat_type: str = "private") -> str:
        """Answer a question using RAG."""
        try:
            logger.info(f"🤖 Processing question from user {user_id}: {question[:50]}...")
            
            # Check OpenAI configuration
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OpenAI API key not found")
                return "Sorry, I'm not properly configured. Please contact the administrator."
            
            # Simple system prompt for now
            system_prompt = """You are OnlyAi Support, the assistant for Jimmy Denero's "OnlyAi" (AI OnlyFans Management) community. 
            
            Provide helpful, practical answers about OnlyAi and AI topics. Be conversational and professional.
            
            If the user greets you, ask what they need help with.
            If you don't know something, say so and suggest where they might find the information.
            Keep answers concise and actionable."""
            
            # Simple user prompt
            user_prompt = f"Question: {question}\n\nProvide a helpful answer about OnlyAi and AI topics."
            
            # Call OpenAI
            logger.info("Calling OpenAI API...")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated answer for user {user_id}: {answer[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return f"Sorry, I'm having trouble processing your question. Error: {str(e)[:100]}"
    
    def store_question_answer(self, question: str, answer: str, user_id: int, chat_type: str):
        """Store Q&A for future reference."""
        try:
            db = SessionLocal()
            
            # Store the question
            question_msg = Message(
                message_id=0,  # Will be set by Telegram
                chat_id=user_id,
                user_id=user_id,
                username=None,  # Will be filled by message handler
                first_name=None,
                last_name=None,
                chat_type=chat_type,
                text=f"Q: {question}",
                is_monitored=False,
                is_owner=False
            )
            db.add(question_msg)
            
            # Store the answer
            answer_msg = Message(
                message_id=0,
                chat_id=user_id,
                user_id=0,  # Bot user ID
                username="OnlyAi Support",
                first_name="OnlyAi",
                last_name="Support",
                chat_type=chat_type,
                text=f"A: {answer}",
                is_monitored=False,
                is_owner=False
            )
            db.add(answer_msg)
            
            db.commit()
            logger.info(f"💾 Stored Q&A for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error storing Q&A: {e}")
        finally:
            db.close()

    async def simple_test(self, question: str, user_id: int) -> str:
        """Simple test method to isolate OpenAI API issues."""
        try:
            logger.info(f"🧪 Simple test for user {user_id}: {question}")
            
            # Check OpenAI configuration
            if not os.getenv("OPENAI_API_KEY"):
                return "OpenAI API key missing"
            
            # Simple OpenAI call
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"✅ Simple test successful: {answer[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Simple test failed: {e}")
            return f"Error: {str(e)}"
