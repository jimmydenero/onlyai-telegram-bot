#!/bin/bash

echo "🚀 Deploying OnlyAi Support Bot..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create it with your configuration."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install -r requirements.txt

# Test the bot
echo "🧪 Testing bot functionality..."
python3 test_bot.py

if [ $? -eq 0 ]; then
    echo "✅ Bot test passed!"
else
    echo "❌ Bot test failed!"
    exit 1
fi

# Start the bot
echo "🤖 Starting OnlyAi Support Bot..."
echo "📡 Bot will be available at: http://localhost:8080"
echo "🔗 Health check: http://localhost:8080/"
echo "🌐 Webhook endpoint: http://localhost:8080/webhook"
echo ""
echo "Press Ctrl+C to stop the bot"
echo ""

python3 main.py


