import os
import csv
import random
import logging
from datetime import datetime as dt
import pytz
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
import asyncio

# --- CONFIG ---
TOKEN = '7898515154:AAE2SJYkWCMoQ1S-3yZ6LW95kh8DtMgDxn0'
CSV_PATH = os.path.join(os.path.dirname(__file__), 'SFMBA Directory.csv')
CHAT_ID_FILE = 'chat_id.txt'
TIMEZONE = 'America/New_York'  # EST/EDT
DAILY_HOUR = 9  # 9:00 AM

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- CSV Handling ---
class ContactManager:
    def __init__(self, csv_path):
        self.contacts = []
        self.csv_path = csv_path
        self._load_contacts()

    def _load_contacts(self):
        try:
            with open(self.csv_path, newline='', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
                reader = csv.DictReader(f)
                
                # Clean column names and find matches
                clean_headers = [h.strip().replace('\ufeff', '') for h in reader.fieldnames]
                name_col = next((h for h in clean_headers if h.lower() == 'name'), None)
                
                if not name_col:
                    logging.error(f"Name column not found. Available columns: {clean_headers}")
                    return
                    
                for row in reader:
                    name = row.get(name_col, '').strip()
                    if not name:
                        continue
                        
                    contact = {
                        'Name': name,
                        'Private Email': row.get('Personal Email', '').strip(),
                        'MIT Email': row.get('MIT Email', '').strip(),
                        'Birthday': f"{row.get('Birthday Month', '').strip()}/{row.get('Birthday Day', '').strip()}",
                        'Connection': row.get('Connection', 'Normal').strip()
                    }
                    self.contacts.append(contact)
                    
                logging.info(f"Loaded {len(self.contacts)} contacts")
                
        except Exception as e:
            logging.error(f"Error loading contacts: {str(e)}", exc_info=True)

    def save_contacts(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'Private Email', 'MIT Email', 'Birthday', 'Connection'])
            writer.writeheader()
            writer.writerows(self.contacts)

    def get_today_birthdays(self):
        today = dt.now(pytz.timezone(TIMEZONE)).strftime('%m/%d')
        return [
            c for c in self.contacts 
            if c.get('Birthday', '').endswith(today)
        ]

    def get_suggestions(self, count=2):
        valid = [c for c in self.contacts if c.get('Connection') != 'Stop']
        if not valid:
            return []
            
        weights = [WEIGHTS.get(c.get('Connection', 'Normal'), 1) for c in valid]
        return random.choices(valid, weights=weights, k=min(count, len(valid)))

# --- Logic ---
WEIGHTS = {'Close': 3, 'Normal': 1, 'Stop': 0}
CONNECTION_LEVELS = {'Close', 'Normal', 'Stop'}

contact_manager = ContactManager(CSV_PATH)

# --- Google Sheets Integration ---
# Uncomment to enable:
def sync_with_google_sheets():
    """Sync contacts with Google Sheets"""
    # Requires gspread package
    # Would need Google API credentials
    pass

# --- Messaging ---
def format_daily_message(birthdays, suggestions):
    msg = "🎂 *Today's Birthdays* 🎂\n\n"
    if birthdays:
        for c in birthdays:
            name = c.get('Name', '').strip() or 'Name not available'
            msg += f"🌟 *{name}*\n"
            if c.get('Private Email', '').strip():
                msg += f"• Personal: {c['Private Email'].strip()}\n"
            if c.get('MIT Email', '').strip():
                msg += f"• MIT: {c['MIT Email'].strip()}\n"
            msg += "\n"
    else:
        msg += "No birthdays today!\n\n"
    
    msg += "👋 *Connection Suggestions* 👋\n\n"
    if suggestions:
        for i, c in enumerate(suggestions, 1):
            name = c.get('Name', '').strip() or 'Name not available'
            connection = c.get('Connection', 'Normal').strip()
            msg += f"{i}. *{name}* ({connection})\n"
            if c.get('Private Email', '').strip():
                msg += f"• Contact: {c['Private Email'].strip()}\n"
            msg += "\n"
    else:
        msg += "No suggestions today.\n"
    
    # Simplified update instructions
    msg += "\n🔧 *Update Connections*\n"
    msg += "To update, send:\n"
    msg += "Firstname - Level\n"
    msg += "Example:\n"
    msg += "John - Close\n"
    msg += "Jane - Stop"
    
    return msg

# --- Bot and Dispatcher ---
bot = Bot(token=TOKEN)
dp = Dispatcher()

# --- Telegram Handlers (aiogram style) ---
@dp.message(Command("start"))
async def start(message: Message):
    await message.answer("Welcome! I'll help you keep in touch with your Sloan network.")
    chat_id = message.chat.id
    with open(CHAT_ID_FILE, 'w') as f:
        f.write(str(chat_id))
    await message.answer("Chat ID saved for daily reminders.")

async def send_daily_message():
    birthdays = contact_manager.get_today_birthdays()
    suggestions = contact_manager.get_suggestions()
    msg = format_daily_message(birthdays, suggestions)
    with open(CHAT_ID_FILE) as f:
        chat_id = int(f.read().strip())
    await bot.send_message(chat_id=chat_id, text=msg)

@dp.message(Command("today"))
async def today(message: Message):
    await send_daily_message()
    await message.answer("Today's message sent!")

@dp.message(Command("test"))
async def test_contacts(message: Message):
    """Test command to verify contact loading"""
    if not contact_manager.contacts:
        await message.answer("⚠️ No contacts loaded! Check CSV file and logs.")
        return
        
    sample = random.sample(contact_manager.contacts, min(5, len(contact_manager.contacts)))
    response = "✅ Loaded contacts test:\n\n"
    
    for i, contact in enumerate(sample, 1):
        response += f"{i}. {contact['Name']} ({contact['Connection']})"
        if contact['Private Email']:
            response += f" - {contact['Private Email']}"
        response += "\n"
    
    response += f"\nTotal contacts: {len(contact_manager.contacts)}"
    await message.answer(response)

@dp.message(Command("debug"))
async def debug_info(message: Message):
    """Show debug information about loaded contacts"""
    today = dt.now(pytz.timezone(TIMEZONE)).strftime('%m/%d')
    
    response = (
        f"🛠️ Debug Information\n"
        f"• Current date: {today}\n"
        f"• Loaded contacts: {len(contact_manager.contacts)}\n"
        f"• Sample contacts:\n"
    )
    
    for i, contact in enumerate(contact_manager.contacts[:3], 1):
        response += (
            f"{i}. {contact['Name']} "
            f"(Birthday: {contact.get('Birthday', 'N/A')}, "
            f"Connection: {contact.get('Connection', 'N/A')})\n"
        )
    
    await message.answer(response)

@dp.message()
async def handle_reply(message: Message):
    """Handle relationship updates with simple format: Firstname - Level"""
    if not message.text:
        return

    try:
        updates = []
        for line in message.text.strip().split('\n'):
            if '-' in line:
                name_part, level = map(str.strip, line.split('-', 1))
                if name_part and level in CONNECTION_LEVELS:
                    updates.append((name_part.lower(), level))

        if not updates:
            await message.answer("⚠️ No valid updates found. Use format:\nFirstname - Level\nExample:\nJohn - Close")
            return

        updated = []
        for name_part, new_level in updates:
            for contact in contact_manager.contacts:
                contact_first = contact['Name'].split()[0].lower()
                if (name_part == contact['Name'].lower() or 
                    name_part == contact_first):
                    if contact.get('Connection') != new_level:
                        contact['Connection'] = new_level
                        updated.append(f"{contact['Name']} → {new_level}")
                    break

        if updated:
            contact_manager.save_contacts()
            response = "✅ Updated:\n" + "\n".join(updated)
            await message.answer(response)
        else:
            await message.answer("⚠️ No matches found. Try using just the first name.")

    except Exception as e:
        logging.error(f"Update error: {e}", exc_info=True)
        await message.answer("❌ Error processing updates. Please check format and try again.")

# --- Deployment ---
# For PythonAnywhere:
# 1. Upload bot.py and CSV to your account
# 2. Create a new 'Always-On' task
# 3. Set command: python3 bot.py
# 4. The CSV will persist in your files

# --- Alternative: AWS Lambda ---
# Would require:
# - Storing CSV in S3
# - Setting up CloudWatch trigger
# - More complex setup

# Current implementation works for both local and cloud

# --- Scheduler ---
async def scheduler():
    while True:
        now = dt.now(pytz.timezone(TIMEZONE))
        if now.hour == DAILY_HOUR and now.minute == 0:
            await send_daily_message()
        await asyncio.sleep(60)

# --- Main Entrypoint ---
async def main():
    # Start the scheduler as a background task
    asyncio.create_task(scheduler())
    
    # Check for existing chat_id and send message if available
    try:
        with open(CHAT_ID_FILE) as f:
            chat_id = int(f.read().strip())
            await send_daily_message()
            logging.info("Sent initial daily message")
    except FileNotFoundError:
        logging.info("Waiting for /start command to initialize chat")
    except Exception as e:
        logging.error(f"Startup message error: {e}")
    
    # Start polling
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
