import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Minimal bot is working!')

def main():
    token = os.getenv('TELEGRAM_TOKEN')
    logger.info(f"Token: {'*'*len(token) if token else 'NOT FOUND'}")
    
    if not token:
        logger.error("TELEGRAM_TOKEN not found in environment")
        return
        
    try:
        app = Application.builder().token(token).build()
        app.add_handler(CommandHandler("start", start))
        logger.info("Starting bot...")
        app.run_polling()
    except Exception as e:
        logger.error(f"Bot failed: {e}")

if __name__ == '__main__':
    main()
