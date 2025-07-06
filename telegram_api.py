import os
import requests
from typing import Optional, Dict, Any

class TelegramAPI:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('TELEGRAM_TOKEN')
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
    def send_message(self, chat_id: int, text: str, parse_mode: str = None) -> bool:
        """Send a message to a chat."""
        payload = {
            'chat_id': chat_id,
            'text': text
        }
        if parse_mode:
            payload['parse_mode'] = parse_mode
            
        response = requests.post(f"{self.base_url}/sendMessage", json=payload)
        return response.status_code == 200
    
    def set_webhook(self, url: str) -> bool:
        """Set a webhook URL for receiving updates."""
        response = requests.post(f"{self.base_url}/setWebhook", json={'url': url})
        return response.status_code == 200
    
    def get_updates(self, offset: Optional[int] = None) -> Dict[str, Any]:
        """Get new updates from Telegram."""
        params = {'timeout': 30}
        if offset:
            params['offset'] = offset
            
        response = requests.get(f"{self.base_url}/getUpdates", params=params)
        return response.json() if response.status_code == 200 else {}
    
    def answer_callback_query(self, callback_query_id: str, text: str = None) -> bool:
        """Answer a callback query (for inline buttons)."""
        payload = {'callback_query_id': callback_query_id}
        if text:
            payload['text'] = text
            
        response = requests.post(f"{self.base_url}/answerCallbackQuery", json=payload)
        return response.status_code == 200
    
    def edit_message_text(self, chat_id: int, message_id: int, text: str, 
                         reply_markup: Optional[Dict] = None) -> bool:
        """Edit an existing message."""
        payload = {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': text
        }
        if reply_markup:
            payload['reply_markup'] = reply_markup
            
        response = requests.post(f"{self.base_url}/editMessageText", json=payload)
        return response.status_code == 200
