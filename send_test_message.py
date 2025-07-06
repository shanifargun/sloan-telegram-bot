import os
import asyncio
from telegram import Bot
from dotenv import load_dotenv

load_dotenv()

async def send_test_message():
    """Automatically detect and respond to user messages"""
    try:
        bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
        
        print("Waiting for you to message @SloanFellowsBot...")
        
        # Wait for new messages (max 30 seconds)
        for _ in range(30):
            updates = await bot.get_updates()
            if updates:
                user_id = updates[-1].message.from_user.id
                username = updates[-1].message.from_user.username
                first_name = updates[-1].message.from_user.first_name
                
                test_msg = (
                    f"👋 Hello {first_name}!\n"
                    "✅ Your SloanFellowsBot is working!\n\n"
                    f"User ID: {user_id}\n"
                    f"Username: @{username}"
                )
                
                await bot.send_message(
                    chat_id=user_id,
                    text=test_msg
                )
                print(f"Success! Sent test message to {first_name} (@{username})")
                return
            
            await asyncio.sleep(1)
        
        print("Timeout: No messages received in 30 seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(send_test_message())
