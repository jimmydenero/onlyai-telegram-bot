"""
Telegram Bot for OnlyAi Support
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Update
from aiogram.filters import Command
from dotenv import load_dotenv

from models import SessionLocal, Message, MonitoredGroup
from rag_service import RAGService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self):
        """Initialize the Telegram bot."""
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment")
        
        # Initialize bot and dispatcher
        self.bot = Bot(token=self.token)
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        
        # Load configuration
        self.allowed_user_ids = self._parse_user_ids(os.getenv("ALLOWED_USER_IDS", ""))
        self.owner_telegram_ids = self._parse_user_ids(os.getenv("OWNER_TELEGRAM_IDS", ""))
        
        # Initialize RAG service
        self.rag_service = RAGService()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("TelegramBot initialized successfully")
    
    def _parse_user_ids(self, user_ids_str: str) -> List[int]:
        """Parse comma-separated user IDs string into list of integers."""
        if not user_ids_str:
            return []
        return [int(uid.strip()) for uid in user_ids_str.split(",") if uid.strip()]
    
    def _register_handlers(self):
        """Register all message handlers."""
        
        # Command handlers
        self.dp.message.register(self._handle_test, Command(commands=["test"]))
        self.dp.message.register(self._handle_start, Command(commands=["start"]))
        self.dp.message.register(self._handle_help, Command(commands=["help"]))
        self.dp.message.register(self._handle_status, Command(commands=["status"]))
        self.dp.message.register(self._handle_groups, Command(commands=["groups"]))
        self.dp.message.register(self._handle_monitor, Command(commands=["monitor"]))
        self.dp.message.register(self._handle_messages, Command(commands=["messages"]))
        self.dp.message.register(self._handle_group_messages, Command(commands=["group_messages"]))
        self.dp.message.register(self._handle_add_doc, Command(commands=["add_doc"]))
        self.dp.message.register(self._handle_search_docs, Command(commands=["search_docs"]))
        self.dp.message.register(self._handle_list_docs, Command(commands=["list_docs"]))
        self.dp.message.register(self._handle_upload_file, Command(commands=["upload_file"]))
        self.dp.message.register(self._handle_scrape_url, Command(commands=["scrape_url"]))
        
        # Message handlers - register forwarded message handler first
        self.dp.message.register(self._handle_forwarded_message, lambda m: m.forward_from is not None)
        self.dp.message.register(self._handle_message)
    
    async def handle_update(self, update_data: Dict[str, Any]):
        """Handle incoming Telegram update."""
        try:
            update = Update(**update_data)
            await self.dp.feed_update(self.bot, update)
        except Exception as e:
            logger.error(f"Error handling update: {e}")
    
    async def set_webhook(self, webhook_url: str):
        """Set the webhook URL."""
        try:
            await self.bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set to: {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")
            raise
    
    async def delete_webhook(self):
        """Delete the webhook."""
        try:
            await self.bot.delete_webhook()
            logger.info("Webhook deleted")
        except Exception as e:
            logger.error(f"Failed to delete webhook: {e}")
            raise
    
    def _is_allowed_user(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot."""
        if not self.allowed_user_ids:  # If empty, allow all users
            return True
        return user_id in self.allowed_user_ids
    
    def _is_owner(self, user_id: int) -> bool:
        """Check if user is an owner."""
        return user_id in self.owner_telegram_ids
    
    def _is_monitored_chat(self, chat_id: int) -> bool:
        """Check if a chat is being monitored."""
        try:
            db = SessionLocal()
            monitored = db.query(MonitoredGroup).filter(
                MonitoredGroup.chat_id == chat_id,
                MonitoredGroup.is_active == True
            ).first()
            return monitored is not None
        except Exception as e:
            logger.error(f"Error checking monitored chat: {e}")
            return False
        finally:
            db.close()
    
    def _store_message(self, message: types.Message, is_monitored: bool = False):
        """Store message in database."""
        db = None
        try:
            if not message.text:  # Only store text messages
                return
            
            # Don't store owner messages
            if self._is_owner(message.from_user.id):
                logger.info(f"🚫 Skipping owner message from {message.from_user.id}")
                return
            
            db = SessionLocal()
            db_message = Message(
                message_id=message.message_id,
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                username=message.from_user.username,
                first_name=message.from_user.first_name,
                last_name=message.from_user.last_name,
                chat_type=message.chat.type,
                text=message.text,
                is_monitored=is_monitored,
                is_owner=self._is_owner(message.from_user.id)
            )
            db.add(db_message)
            db.commit()
            logger.info(f"💾 Stored message {message.message_id} from {message.from_user.username or message.from_user.first_name} in {message.chat.type} chat {message.chat.id} (monitored: {is_monitored})")
        except Exception as e:
            logger.error(f"Error storing message: {e}")
        finally:
            if db:
                db.close()
    
    async def _handle_test(self, message: types.Message):
        """Handle /test command."""
        try:
            user_id = message.from_user.id
            chat_id = message.chat.id
            
            logger.info(f"Test command from user {user_id} in chat {chat_id}")
            
            # Check user permissions
            if not self._is_allowed_user(user_id):
                await message.reply("❌ You are not authorized to use this bot.")
                return
            
            # Return health status
            status_text = (
                "✅ **Bot Status**\n"
                f"• Model: {os.getenv('OPENAI_MODEL', 'gpt-5.0-thinking')}\n"
                f"• User ID: {user_id}\n"
                f"• Chat ID: {chat_id}\n"
                f"• Allowed: {'Yes' if self._is_allowed_user(user_id) else 'No'}\n"
                f"• Owner: {'Yes' if self._is_owner(user_id) else 'No'}\n"
                f"• Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            
            await message.reply(status_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in test handler: {e}")
            await message.reply("❌ Error processing test command.")
    
    async def _handle_start(self, message: types.Message):
        """Handle /start command."""
        try:
            user_id = message.from_user.id
            
            if not self._is_allowed_user(user_id):
                await message.reply(
                    "❌ You are not authorized to use this bot.\n"
                    "Please contact the administrator to get access."
                )
                return
            
            welcome_text = (
                "🤖 **OnlyAi Support Bot**\n\n"
                "I'm here to help you with questions about OnlyAi!\n\n"
                "**Commands:**\n"
                "• `/test` - Check bot status\n"
                "• `/status` - Detailed status report\n"
                "• `/help` - Show this help message\n\n"
                "**Owner Commands:**\n"
                "• `/groups` - View monitored groups\n"
                "• `/monitor` - Activate group monitoring (use in group)\n"
                "• `/messages` - View all stored messages\n"
                "• `/group_messages` - View only group messages\n\n"
                "**How to use:**\n"
                "• Ask me questions directly\n"
                "• Mention me in group chats: @onlyaisupportbot\n"
                "• I'll provide answers based on our knowledge base and chat history"
            )
            
            await message.reply(welcome_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in start handler: {e}")
            await message.reply("❌ Error processing start command.")
    
    async def _handle_help(self, message: types.Message):
        """Handle /help command."""
        await self._handle_start(message)  # Same as start for now
    
    async def _handle_status(self, message: types.Message):
        """Handle /status command."""
        try:
            user_id = message.from_user.id
            chat_id = message.chat.id
            chat_type = message.chat.type
            
            logger.info(f"Status command from user {user_id} in {chat_type} chat {chat_id}")
            
            # Check user permissions
            if not self._is_allowed_user(user_id):
                await message.reply("❌ You are not authorized to use this bot.")
                return
            
            # Return detailed status
            status_text = (
                "📊 **Bot Status Report**\n\n"
                f"• **Chat Type:** {chat_type}\n"
                f"• **Chat ID:** {chat_id}\n"
                f"• **User ID:** {user_id}\n"
                f"• **Allowed:** {'Yes' if self._is_allowed_user(user_id) else 'No'}\n"
                f"• **Owner:** {'Yes' if self._is_owner(user_id) else 'No'}\n"
                f"• **Monitoring:** Active\n"
                f"• **RAG:** Ready ✅\n\n"
                f"**To monitor group messages:**\n"
                f"1. Add bot to group\n"
                f"2. Make bot admin\n"
                f"3. Disable privacy mode in @BotFather\n\n"
                f"**Current time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            
            await message.reply(status_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in status handler: {e}")
            await message.reply("❌ Error processing status command.")
    
    async def _handle_groups(self, message: types.Message):
        """Handle /groups command."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Groups command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can view group information.")
                return
            
            # Get monitored groups from database
            try:
                db = SessionLocal()
                monitored_groups = db.query(MonitoredGroup).filter(
                    MonitoredGroup.is_active == True
                ).all()
                
                if monitored_groups:
                    groups_list = []
                    for group in monitored_groups:
                        groups_list.append(f"• {group.chat_title} (ID: {group.chat_id})")
                    
                    groups_text = (
                        "📋 **Group Monitoring Status**\n\n"
                        "**Currently monitoring:**\n" + 
                        "\n".join(groups_list) + "\n\n"
                        "**To add more groups:**\n"
                        "1. Add @onlyaisupportbot to your group\n"
                        "2. Make the bot an admin\n"
                        "3. Ensure privacy mode is disabled\n"
                        "4. Send /monitor in the group to activate monitoring\n\n"
                        "**Privacy mode check:**\n"
                        "Go to @BotFather → /setprivacy → Select your bot → Disable"
                    )
                else:
                    groups_text = (
                        "📋 **Group Monitoring Status**\n\n"
                        "**Currently monitoring:**\n"
                        "• No groups configured yet\n\n"
                        "**To add group monitoring:**\n"
                        "1. Add @onlyaisupportbot to your group\n"
                        "2. Make the bot an admin\n"
                        "3. Ensure privacy mode is disabled\n"
                        "4. Send /monitor in the group to activate monitoring\n\n"
                        "**Privacy mode check:**\n"
                        "Go to @BotFather → /setprivacy → Select your bot → Disable"
                    )
            except Exception as e:
                logger.error(f"Error getting monitored groups: {e}")
                groups_text = "❌ Error retrieving group information."
            finally:
                db.close()
            
            await message.reply(groups_text, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in groups handler: {e}")
            await message.reply("❌ Error processing groups command.")
    
    async def _handle_monitor(self, message: types.Message):
        """Handle /monitor command to set up group monitoring."""
        try:
            user_id = message.from_user.id
            chat_id = message.chat.id
            chat_type = message.chat.type
            
            logger.info(f"Monitor command from user {user_id} in {chat_type} chat {chat_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can set up group monitoring.")
                return
            
            # Only allow in groups/supergroups
            if chat_type not in ["group", "supergroup"]:
                await message.reply("❌ This command can only be used in groups.")
                return
            
            # Check if already monitoring this group
            if self._is_monitored_chat(chat_id):
                await message.reply(f"✅ This group is already being monitored (Chat ID: {chat_id})")
                return
            
            # Add group to monitored list
            try:
                db = SessionLocal()
                monitored_group = MonitoredGroup(
                    chat_id=chat_id,
                    chat_title=message.chat.title
                )
                db.add(monitored_group)
                db.commit()
                
                await message.reply(
                    f"✅ **Group monitoring activated!**\n\n"
                    f"• **Group:** {message.chat.title}\n"
                    f"• **Chat ID:** {chat_id}\n"
                    f"• **Type:** {chat_type}\n\n"
                    f"All text messages in this group will now be stored in the database."
                )
                logger.info(f"🎯 Started monitoring group {chat_id} ({message.chat.title})")
                
            except Exception as e:
                logger.error(f"Error adding monitored group: {e}")
                await message.reply("❌ Error setting up group monitoring.")
            finally:
                db.close()
            
        except Exception as e:
            logger.error(f"Error in monitor handler: {e}")
            await message.reply("❌ Error processing monitor command.")
    
    async def _handle_messages(self, message: types.Message):
        """Handle /messages command to view stored messages."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Messages command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can view stored messages.")
                return
            
            # Get recent messages from database
            try:
                db = SessionLocal()
                recent_messages = db.query(Message).order_by(
                    Message.timestamp.desc()
                ).limit(10).all()
                
                if recent_messages:
                    messages_list = []
                    for msg in recent_messages:
                        status = "🎯" if msg.is_monitored else "📝"
                        user_display = msg.username or f"{msg.first_name} {msg.last_name or ''}".strip() or f"User {msg.user_id}"
                        
                        # Truncate text and remove problematic characters
                        safe_text = msg.text[:100].replace('\n', ' ').replace('\r', ' ')
                        if len(msg.text) > 100:
                            safe_text += "..."
                        
                        messages_list.append(
                            f"{status} {msg.chat_type} (ID: {msg.chat_id})\n"
                            f"User: {user_display} | {msg.timestamp.strftime('%H:%M:%S')}\n"
                            f"Text: {safe_text}"
                        )
                    
                    messages_text = (
                        "📊 Recent Stored Messages\n\n" +
                        "\n\n".join(messages_list) +
                        f"\n\nTotal stored: {db.query(Message).count()} messages"
                    )
                else:
                    messages_text = "📊 No messages stored yet."
                    
            except Exception as e:
                logger.error(f"Error getting messages: {e}")
                messages_text = "❌ Error retrieving messages."
            finally:
                db.close()
            
            await message.reply(messages_text)
            
        except Exception as e:
            logger.error(f"Error in messages handler: {e}")
            await message.reply("❌ Error processing messages command.")
    
    async def _handle_group_messages(self, message: types.Message):
        """Handle /group_messages command to view only group messages."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Group messages command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can view group messages.")
                return
            
            # Get recent group messages from database
            try:
                db = SessionLocal()
                group_messages = db.query(Message).filter(
                    Message.chat_type.in_(["group", "supergroup"]),
                    Message.is_monitored == True
                ).order_by(Message.timestamp.desc()).limit(10).all()
                
                if group_messages:
                    messages_list = []
                    for msg in group_messages:
                        user_display = msg.username or f"{msg.first_name} {msg.last_name or ''}".strip() or f"User {msg.user_id}"
                        
                        # Truncate text and remove problematic characters
                        safe_text = msg.text[:100].replace('\n', ' ').replace('\r', ' ')
                        if len(msg.text) > 100:
                            safe_text += "..."
                        
                        messages_list.append(
                            f"🎯 {msg.chat_type} (ID: {msg.chat_id})\n"
                            f"User: {user_display} | {msg.timestamp.strftime('%H:%M:%S')}\n"
                            f"Text: {safe_text}"
                        )
                    
                    messages_text = (
                        "👥 Recent Group Messages\n\n" +
                        "\n\n".join(messages_list) +
                        f"\n\nTotal group messages: {db.query(Message).filter(Message.chat_type.in_(['group', 'supergroup'])).count()}"
                    )
                else:
                    messages_text = "👥 No group messages stored yet."
                    
            except Exception as e:
                logger.error(f"Error getting group messages: {e}")
                messages_text = "❌ Error retrieving group messages."
            finally:
                db.close()
            
            await message.reply(messages_text)
            
        except Exception as e:
            logger.error(f"Error in group messages handler: {e}")
            await message.reply("❌ Error processing group messages command.")
    
    async def _handle_add_doc(self, message: types.Message):
        """Handle /add_doc command to add documents to knowledge base."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Add document command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can add documents to the knowledge base.")
                return
            
            # Check if message has text after command
            command_text = message.text.strip()
            if command_text == "/add_doc":
                await message.reply(
                    "📝 To add a document, use:\n"
                    "/add_doc <title> | <content>\n\n"
                    "Example:\n"
                    "/add_doc OnlyAi Setup Guide | This guide covers the complete setup process for OnlyAi..."
                )
                return
            
            # Parse title and content
            try:
                parts = command_text.split(" | ", 1)
                if len(parts) != 2:
                    await message.reply("❌ Invalid format. Use: /add_doc <title> | <content>")
                    return
                
                title = parts[0].replace("/add_doc", "").strip()
                content = parts[1].strip()
                
                if not title or not content:
                    await message.reply("❌ Title and content cannot be empty.")
                    return
                
                # Add document to knowledge base
                success = self.rag_service.add_document(title, content)
                
                if success:
                    await message.reply(f"✅ Document '{title}' added to knowledge base successfully!")
                else:
                    await message.reply("❌ Failed to add document to knowledge base.")
                    
            except Exception as e:
                logger.error(f"Error parsing add_doc command: {e}")
                await message.reply("❌ Error parsing command. Use: /add_doc <title> | <content>")
                
        except Exception as e:
            logger.error(f"Error in add_doc handler: {e}")
            await message.reply("❌ Error processing add_doc command.")
    
    async def _handle_search_docs(self, message: types.Message):
        """Handle /search_docs command to search knowledge base."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Search docs command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can search the knowledge base.")
                return
            
            # Get search query
            query = message.text.replace("/search_docs", "").strip()
            if not query:
                await message.reply("❌ Please provide a search query: /search_docs <query>")
                return
            
            # Search knowledge base
            results = self.rag_service.search_knowledge_base(query, limit=5)
            
            if results:
                response = f"🔍 Search results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    response += f"{i}. **{result['title']}**\n"
                    response += f"   Category: {result['category']}\n"
                    response += f"   Content: {result['content'][:200]}...\n\n"
            else:
                response = f"🔍 No documents found for '{query}'"
            
            await message.reply(response)
            
        except Exception as e:
            logger.error(f"Error in search_docs handler: {e}")
            await message.reply("❌ Error processing search_docs command.")
    
    async def _handle_list_docs(self, message: types.Message):
        """Handle /list_docs command to list all documents."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"List docs command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can list knowledge base documents.")
                return
            
            # Get all documents
            try:
                import sqlite3
                from pathlib import Path
                
                kb_db_path = Path("./knowledge_base/knowledge_base.db")
                if not kb_db_path.exists():
                    await message.reply("📚 No knowledge base found.")
                    return
                
                conn = sqlite3.connect(kb_db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT title, category, created_at, 
                           (SELECT COUNT(*) FROM embeddings WHERE document_id = documents.id) as chunks
                    FROM documents 
                    ORDER BY created_at DESC
                ''')
                
                docs = cursor.fetchall()
                conn.close()
                
                if docs:
                    response = "📚 Knowledge Base Documents:\n\n"
                    for i, (title, category, created_at, chunks) in enumerate(docs, 1):
                        response += f"{i}. **{title}**\n"
                        response += f"   Category: {category}\n"
                        response += f"   Chunks: {chunks}\n"
                        response += f"   Added: {created_at}\n\n"
                else:
                    response = "📚 No documents in knowledge base."
                
                await message.reply(response)
                
            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                await message.reply("❌ Error retrieving documents.")
                
        except Exception as e:
            logger.error(f"Error in list_docs handler: {e}")
            await message.reply("❌ Error processing list_docs command.")
    
    async def _handle_upload_file(self, message: types.Message):
        """Handle /upload_file command to upload files to knowledge base."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Upload file command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can upload files to the knowledge base.")
                return
            
            # Check if message has a document
            if not message.document:
                await message.reply(
                    "📁 To upload a file, send the file with caption:\n"
                    "/upload_file <title> | <category> | <tags>\n\n"
                    "Example:\n"
                    "/upload_file OnlyAi SOP | setup | guide,manual\n\n"
                    "Supported formats: PDF, DOCX, TXT, MP4, MOV, AVI"
                )
                return
            
            # Get file info
            file_info = message.document
            file_name = file_info.file_name
            
            # Check file type
            allowed_extensions = ['.pdf', '.docx', '.txt', '.mp4', '.mov', '.avi']
            file_extension = Path(file_name).suffix.lower()
            
            if file_extension not in allowed_extensions:
                await message.reply(f"❌ Unsupported file type: {file_extension}\nSupported: {', '.join(allowed_extensions)}")
                return
            
            # Parse caption for metadata
            caption = message.caption or ""
            parts = caption.split(" | ")
            
            title = parts[0].replace("/upload_file", "").strip() if len(parts) > 0 else file_name
            category = parts[1].strip() if len(parts) > 1 else "general"
            tags = [tag.strip() for tag in parts[2].split(",")] if len(parts) > 2 else []
            
            # Download file
            try:
                file_path = f"./uploads/{file_name}"
                os.makedirs("./uploads", exist_ok=True)
                
                await message.bot.download(file_info, file_path)
                
                # Process and add to knowledge base
                success = self.rag_service.add_document_from_file(
                    file_path=file_path,
                    title=title,
                    category=category,
                    tags=tags
                )
                
                if success:
                    await message.reply(f"✅ File '{title}' uploaded and processed successfully!")
                else:
                    await message.reply("❌ Failed to process file. Check the file format and try again.")
                
                # Clean up downloaded file
                os.remove(file_path)
                
            except Exception as e:
                logger.error(f"Error downloading/processing file: {e}")
                await message.reply("❌ Error processing file. Please try again.")
                
        except Exception as e:
            logger.error(f"Error in upload_file handler: {e}")
            await message.reply("❌ Error processing upload_file command.")
    
    async def _handle_scrape_url(self, message: types.Message):
        """Handle /scrape_url command to scrape websites and add to knowledge base."""
        try:
            user_id = message.from_user.id
            
            logger.info(f"Scrape URL command from user {user_id}")
            
            # Check if user is owner
            if not self._is_owner(user_id):
                await message.reply("❌ Only owners can scrape websites for the knowledge base.")
                return
            
            # Check if message has text after command
            command_text = message.text.strip()
            if command_text == "/scrape_url":
                await message.reply(
                    "🌐 To scrape a website, use:\n"
                    "/scrape_url <url> | <title> | <category> | <tags>\n\n"
                    "Example:\n"
                    "/scrape_url https://fanvue.com/terms | Fanvue Terms | legal | terms,policy"
                )
                return
            
            # Parse URL and metadata
            try:
                parts = command_text.split(" | ")
                if len(parts) < 1:
                    await message.reply("❌ Invalid format. Use: /scrape_url <url> | <title> | <category> | <tags>")
                    return
                
                url = parts[0].replace("/scrape_url", "").strip()
                title = parts[1].strip() if len(parts) > 1 else None
                category = parts[2].strip() if len(parts) > 2 else "general"
                tags = [tag.strip() for tag in parts[3].split(",")] if len(parts) > 3 else []
                
                if not url:
                    await message.reply("❌ URL cannot be empty.")
                    return
                
                # Validate URL
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                # Scrape and add to knowledge base
                success = self.rag_service.add_document_from_url(
                    url=url,
                    title=title,
                    category=category,
                    tags=tags
                )
                
                if success:
                    await message.reply(f"✅ Website scraped and added to knowledge base successfully!")
                else:
                    await message.reply("❌ Failed to scrape website. Check the URL and try again.")
                    
            except Exception as e:
                logger.error(f"Error parsing scrape_url command: {e}")
                await message.reply("❌ Error parsing command. Use: /scrape_url <url> | <title> | <category> | <tags>")
                
        except Exception as e:
            logger.error(f"Error in scrape_url handler: {e}")
            await message.reply("❌ Error processing scrape_url command.")
    
    async def _handle_forwarded_message(self, message: types.Message):
        """Handle forwarded messages for owner utilities."""
        try:
            user_id = message.from_user.id
            chat_id = message.chat.id
            
            # Only process in owner's DM
            if not self._is_owner(user_id) or chat_id != user_id:
                return
            
            # Check if this is a forwarded message
            if not message.forward_from:
                return
            
            forwarded_user_id = message.forward_from.id
            forwarded_username = message.forward_from.username or "Unknown"
            
            logger.info(f"Owner {user_id} forwarded message from user {forwarded_user_id}")
            
            # Add user to allowed list
            if forwarded_user_id not in self.allowed_user_ids:
                self.allowed_user_ids.append(forwarded_user_id)
                # TODO: Persist to database
                
                await message.reply(
                    f"✅ Added user {forwarded_username} (ID: {forwarded_user_id}) to allowed users."
                )
            else:
                await message.reply(
                    f"ℹ️ User {forwarded_username} (ID: {forwarded_user_id}) is already allowed."
                )
                
        except Exception as e:
            logger.error(f"Error in forwarded message handler: {e}")
    
    async def _handle_message(self, message: types.Message):
        """Handle regular messages."""
        try:
            user_id = message.from_user.id
            chat_id = message.chat.id
            chat_type = message.chat.type
            text = message.text
            
            if not text:
                return
            
            logger.info(f"📨 Message from user {user_id} in {chat_type} chat {chat_id}: {text[:50]}...")
            
            # Check if this is a monitored chat
            is_monitored = self._is_monitored_chat(chat_id)
            
            # Store text messages in database (owners are filtered out in _store_message)
            self._store_message(message, is_monitored=is_monitored)
            
            # For group messages, only track - don't respond
            if chat_type in ["group", "supergroup"]:
                logger.info(f"👥 Group message tracked in {chat_type} chat {chat_id}: {text[:100]}...")
                return
            
            # For private messages, check permissions and respond
            if not self._is_allowed_user(user_id):
                logger.info(f"❌ Unauthorized user {user_id} tried to send message")
                await message.reply("❌ You are not authorized to use this bot.")
                return
            
            # Check for negative mentions of "jimmydenero" or variants
            negative_keywords = ["jimmy", "jimmydenero", "jimmy denero", "jdenero"]
            text_lower = text.lower()
            
            for keyword in negative_keywords:
                if keyword in text_lower:
                    logger.warning(f"🚨 Potential negative mention detected: '{keyword}' in message from user {user_id}")
                    # TODO: Implement sentiment analysis and alerting
            
            # Use RAG to answer the question
            try:
                answer = await self.rag_service.answer_question(text, user_id, chat_type)
                
                # Store the Q&A for future reference
                self.rag_service.store_question_answer(text, answer, user_id, chat_type)
                
                await message.reply(answer)
                
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                await message.reply(
                    "🤖 I received your message! I'm having trouble processing it right now.\n\n"
                    "📊 Message monitoring is active - I'm tracking all messages for insights."
                )
            
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
    
    async def close(self):
        """Clean up bot resources."""
        try:
            await self.bot.session.close()
            logger.info("Bot session closed")
        except Exception as e:
            logger.error(f"Error closing bot session: {e}")
