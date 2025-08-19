# OnlyAi Telegram Bot

An intelligent Telegram bot for Jimmy Denero's "OnlyAi" (AI OnlyFans Management) community. The bot provides AI-powered support using GPT-4, knowledge base management, and group monitoring capabilities.

## 🚀 Features

- **AI-Powered Support**: Uses GPT-4 for intelligent responses
- **Knowledge Base Management**: Add, search, and manage documents
- **File Processing**: Support for PDF, DOCX, TXT, and video files
- **Web Scraping**: Extract content from websites
- **Group Monitoring**: Monitor and analyze group conversations
- **Owner Controls**: Special permissions for bot owners
- **RAG System**: Retrieval-Augmented Generation for context-aware responses

## 🛠️ Commands

### Basic Commands
- `/start` - Start the bot
- `/help` - Show help information
- `/status` - Check bot status
- `/test` - Test bot functionality

### Knowledge Base Commands
- `/add_doc <title> | <content> | <category> | <tags>` - Add manual document
- `/upload_file` - Upload and process files (PDF, DOCX, TXT, videos)
- `/scrape_url <url> | <title> | <category> | <tags>` - Scrape website content
- `/search_docs <query>` - Search knowledge base
- `/list_docs` - List all documents

### Monitoring Commands
- `/groups` - List monitored groups
- `/monitor` - Manage group monitoring
- `/messages` - View recent messages
- `/group_messages` - View group messages

## 🚀 Quick Deploy on Railway

### Option 1: Deploy from GitHub (Recommended)

1. **Fork this repository** to your GitHub account
2. **Go to [Railway](https://railway.app)** and sign in
3. **Click "New Project"** → "Deploy from GitHub repo"
4. **Select your forked repository**
5. **Add Environment Variables** (see Configuration section below)
6. **Deploy!** Railway will automatically build and deploy your bot

### Option 2: Deploy from Local

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize and deploy**:
   ```bash
   railway init
   railway up
   ```

## ⚙️ Configuration

### Required Environment Variables

Set these in your Railway project settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4
EMBED_MODEL=text-embedding-3-small

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
ALLOWED_USER_IDS=your_user_id
OWNER_TELEGRAM_IDS=your_user_id

# Server Configuration
PORT=8080
HOST=0.0.0.0

# Database Configuration
DATABASE_URL=sqlite:///./data/telegram_ai_agent.db

# File Upload Configuration
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=pdf,docx,txt,md

# Debug Configuration
DEBUG=False
```

### Getting Your Telegram Bot Token

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow the instructions
3. Copy the token provided

### Getting Your User ID

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Copy your user ID from the response

## 🔧 Local Development

### Prerequisites

- Python 3.9+
- pip
- ngrok (for webhook testing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/telegram-ai-agent.git
   cd telegram-ai-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your configuration
   ```

4. **Start ngrok** (for webhook testing):
   ```bash
   ngrok http 8080
   ```

5. **Run the bot**:
   ```bash
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

6. **Set webhook** (replace with your ngrok URL):
   ```bash
   curl -X POST "https://api.telegram.org/botYOUR_BOT_TOKEN/setWebhook" \
        -d "url=https://your-ngrok-url.ngrok.io/webhook"
   ```

## 📁 Project Structure

```
telegram-ai-agent/
├── main.py              # FastAPI application entry point
├── bot.py               # Telegram bot implementation
├── rag_service.py       # RAG (Retrieval-Augmented Generation) service
├── models.py            # Database models
├── requirements.txt     # Python dependencies
├── railway.json         # Railway deployment configuration
├── Procfile            # Railway process file
├── runtime.txt         # Python runtime specification
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- Use environment variables for all sensitive configuration
- Regularly rotate your API keys

## 🐛 Troubleshooting

### Common Issues

1. **Bot not responding**: Check webhook URL and bot token
2. **Database errors**: Ensure database directory exists and is writable
3. **File upload failures**: Check file size limits and allowed extensions
4. **Knowledge base errors**: Verify database schema is correct

### Logs

Check Railway logs for detailed error information:
```bash
railway logs
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For support, please contact the bot owner or create an issue in this repository.


