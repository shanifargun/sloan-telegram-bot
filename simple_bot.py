import os
import time
import logging
from telegram_api import TelegramAPI
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleSloanBot:
    def __init__(self):
        self.api = TelegramAPI()
        self.last_update_id = 0
        
    def handle_updates(self):
        """Main loop to handle incoming updates."""
        logger.info("Starting bot...")
        
        while True:
            try:
                updates = self.api.get_updates(offset=self.last_update_id + 1)
                
                if not updates.get('ok', False):
                    time.sleep(1)
                    continue
                
                for update in updates.get('result', []):
                    self.last_update_id = update['update_id']
                    
                    # Handle different update types
                    if 'message' in update:
                        self.handle_message(update['message'])
                    elif 'callback_query' in update:
                        self.handle_callback(update['callback_query'])
                    
            except Exception as e:
                logger.error(f"Error handling updates: {e}")
                time.sleep(5)
    
    def handle_message(self, message: dict):
        """Handle incoming messages."""
        chat_id = message['chat']['id']
        text = message.get('text', '')
        
        if text.startswith('/start'):
            self.api.send_message(
                chat_id,
                "Welcome to Sloan Social Reminder Bot!\n"
                "Use /today to see today's birthdays and connection suggestions.",
                parse_mode='Markdown'
            )
        elif text.startswith('/today'):
            # TODO: Implement today's birthdays and suggestions
            self.api.send_message(
                chat_id,
                "🎉 Today's birthdays:\n• John Doe\n• Jane Smith\n\n"
                "🤝 People you might want to connect with:",
                parse_mode='Markdown'
            )
    
    def handle_callback(self, callback_query: dict):
        """Handle callback queries from inline buttons."""
        self.api.answer_callback_query(callback_query['id'])
        
        # TODO: Implement callback handling for connection levels
        chat_id = callback_query['message']['chat']['id']
        message_id = callback_query['message']['message_id']
        
        self.api.edit_message_text(
            chat_id,
            message_id,
            "✅ Connection level updated!"
        )

if __name__ == '__main__':
    bot = SimpleSloanBot()
    bot.handle_updates()
