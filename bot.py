import os
import logging
import random
import csv
import datetime
from datetime import date, time
from typing import List, Dict, Tuple, Optional, Any, Union
from dotenv import load_dotenv
import sys
import imghdr_compat
sys.modules['imghdr'] = imghdr_compat
import asyncio
import pytz
import signal

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, CallbackContext, ContextTypes
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
BIRTHDAY_FORMAT = "%m/%d"
CSV_FILE = "SFMBA Directory.csv"
CONNECTION_LEVELS = ["Close", "Normal", "Stop"]
WEIGHTS = {"Close": 3, "Normal": 1, "Stop": 0}

class SloanBot:
    def __init__(self):
        self.people = []
        self.user_connections = {}  # Initialize user_connections dictionary
        self.csv_file = "SFMBA Directory.csv"
        self.connections_file = "user_connections.csv"
        
        # Load data with error handling
        try:
            self.load_data()
            self.load_user_connections()
        except Exception as e:
            logger.error(f"Error initializing SloanBot: {e}")
            raise

    def load_data(self):
        """Load people data from the CSV file."""
        if not os.path.exists(self.csv_file):
            error_msg = f"CSV file not found: {self.csv_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.people = []  # Reset the people list
                
                for row in reader:
                    # Clean and format the data
                    person = {
                        'Name': row.get('Name', '').strip(),
                        'Personal Email': row.get('Personal Email', '').strip(),
                        'MIT Email': row.get('MIT Email', '').strip(),
                        'Birthday Month': row.get('Birthday Month', '').strip(),
                        'Birthday Day': row.get('Birthday Day', '').strip()
                    }
                    if person['Name']:  # Only add if name is not empty
                        self.people.append(person)
                        
            if not self.people:
                logger.warning("No data was loaded from the CSV file")
            else:
                logger.info(f"Successfully loaded {len(self.people)} people from {self.csv_file}")
                
        except csv.Error as e:
            error_msg = f"CSV parsing error in {self.csv_file}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error loading {self.csv_file}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
    def load_user_connections(self):
        """Load user connection preferences from CSV file."""
        if not os.path.exists(self.connections_file):
            logger.info(f"No existing connections file found at {self.connections_file}. Starting with empty connections.")
            return
            
        try:
            with open(self.connections_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.user_connections = {}  # Reset the connections
                
                for row in reader:
                    name = row.get('Name', '').strip()
                    level = row.get('Connection_Level', '').strip()
                    if name and level:  # Only add if both name and level are not empty
                        self.user_connections[name] = level
                            
                logger.info(f"Successfully loaded {len(self.user_connections)} user connections from {self.connections_file}")
                
        except csv.Error as e:
            error_msg = f"CSV parsing error in {self.connections_file}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error loading {self.connections_file}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
    def save_user_connections(self):
        """Save user connection preferences to CSV.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        if not self.user_connections:
            logger.warning("No user connections to save")
            return False
            
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.connections_file)), exist_ok=True)
            
            with open(self.connections_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['Name', 'Connection_Level'])
                # Write data
                for name, level in sorted(self.user_connections.items()):
                    writer.writerow([name, level])
                    
            logger.info(f"Successfully saved {len(self.user_connections)} user connections to {self.connections_file}")
            return True
            
        except PermissionError as e:
            error_msg = f"Permission denied when trying to write to {self.connections_file}: {e}"
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error saving user connections to {self.connections_file}: {e}"
            logger.error(error_msg)
            return False
            
    def set_connection_level(self, name: str, level: str) -> bool:
        """Set connection level for a person.
        
        Args:
            name: The name of the person to update
            level: The connection level to set ('Close', 'Normal', or 'Stop')
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if not name or not isinstance(name, str):
            logger.error(f"Invalid name provided: {name}")
            return False
            
        normalized_level = level.strip().capitalize() if level else ''
        if normalized_level not in ['Close', 'Normal', 'Stop']:
            logger.error(f"Invalid connection level: {level}")
            return False
            
        # Check if the name exists in our data
        if not any(p['Name'] == name for p in self.people):
            logger.warning(f"Name not found in directory: {name}")
            return False
            
        try:
            self.user_connections[name] = normalized_level
            
            # Save the updated connections
            if not self.save_user_connections():
                logger.error(f"Failed to save connection level for {name}")
                return False
                
            logger.info(f"Successfully set connection level for {name} to {normalized_level}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting connection level for {name}: {e}")
            return False
        
    def get_todays_birthdays(self) -> List[Dict]:
        """Get list of people with birthdays today."""
        today = datetime.now()
        month, day = today.month, today.day
        
        return [
            p for p in self.people 
            if int(p.get('Birthday Month', 0)) == month 
            and int(p.get('Birthday Day', 0)) == day
        ]
        
    def get_suggestions(self, count: int = 2) -> List[Dict]:
        """Get random suggestions based on connection levels."""
        if not self.people:
            return []
            
        # Assign weights based on connection level
        weights = []
        for person in self.people:
            name = person['Name']
            level = self.user_connections.get(name, 'Normal')
            
            if level == 'Close':
                weight = 10
            elif level == 'Normal':
                weight = 5
            else:  # 'Stop'
                weight = 0
                
            # Add some randomness
            weight += random.random() * 2
            weights.append(weight)
            
        # Normalize weights
        total = sum(weights)
        if total == 0:
            return []
            
        weights = [w/total for w in weights]
        
        # Select random people based on weights
        return random.choices(
            population=self.people,
            weights=weights,
            k=min(count, len(self.people))
        )
        
    def format_message(self, birthdays: List[Dict], suggestions: List[Dict]) -> Tuple[str, InlineKeyboardMarkup]:
        """Format the message with birthdays and suggestions."""
        message = ""
        
        # Add birthdays section
        if birthdays:
            message += "🎂 *Birthdays Today* 🎂\n\n"
            for person in birthdays:
                name = person.get('Name', 'Unknown')
                personal_email = person.get('Personal Email', '')
                mit_email = person.get('MIT Email', '')
                message += f"• {name} is celebrating today! 🎉\n"
                if personal_email or mit_email:
                    message += "   "
                    if personal_email:
                        message += f"📧 {personal_email}  "
                    if mit_email:
                        message += f"📧 {mit_email}"
                    message += "\n"
            message += "\n"
                
        # Add suggestions section
        if suggestions:
            message += "\n👋 *Suggestions for today:*"
            keyboard_buttons = []
            
            for i, person in enumerate(suggestions, 1):
                name = person.get('Name', 'Unknown')
                level = person.get('Connection_Level', 'Normal')
                email = person.get('Personal Email', person.get('MIT Email', 'No email'))
                
                msg = f"\n{i}. *{name}* ({level})\n   📧 {email}\n"
                message += msg
                
                # Add buttons for updating connection level
                row = []
                for conn_level in CONNECTION_LEVELS:
                    callback_data = f"update_{name}_{conn_level}"
                    row.append(
                        InlineKeyboardButton(
                            f"{name[:5]}...: {conn_level}",
                            callback_data=callback_data
                        )
                    )
                keyboard_buttons.append(row)
            
            if keyboard_buttons:
                reply_markup = InlineKeyboardMarkup(keyboard_buttons)
        
        # Add instructions if we have any interactive elements
        if reply_markup:
            message += (
                "\n_To update someone's connection level, click the buttons below._"
            )
        
        return message, reply_markup

    async def daily_reminder(self, context: ContextTypes.DEFAULT_TYPE):
        """Send daily 9AM reminder"""
        birthdays = self.get_todays_birthdays()
        suggestions = self.get_suggestions()
        
        message = "🎉 Today's Birthdays:\n"
        message += "\n".join([f"• {p['Name']}" for p in birthdays]) if birthdays else "No birthdays today"
        message += "\n\n🤝 Connection Suggestions:\n"
        message += "\n".join([f"• {s}" for s in suggestions[:3]])  # Top 3 suggestions
        
        # Send to all users who have interacted with the bot
        for user_id in self.get_active_users():
            await context.bot.send_message(
                chat_id=user_id,
                text=message
            )

    def run(self):
        """Run the bot with scheduled jobs"""
        self.token = os.getenv('TELEGRAM_TOKEN')
        application = Application.builder().token(self.token).build()
        
        # Schedule daily message (9AM EST = 13:00 UTC)
        job_queue = application.job_queue
        job_queue.run_daily(
            self.daily_reminder,
            time=time(hour=13, minute=0, tzinfo=pytz.UTC),
            name="daily_reminder"
        )
        
        # Add signal handlers for proper shutdown on Render
        loop = asyncio.get_event_loop()
        for signal in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(
                signal,
                lambda: asyncio.create_task(application.shutdown())
            )
        
        try:
            application.run_polling()
        finally:
            # Ensure CSV is saved on shutdown
            self.save_user_connections()

bot = SloanBot()

# Command handlers
def start(update: Update, context: CallbackContext) -> None:
    """Send a welcome message when the command /start is issued."""
    welcome_message = (
        "👋 Welcome to the Sloan Social Reminder Bot!\n\n"
        "I'll help you keep in touch with your Sloan network by:\n"
        "• Sending daily birthday reminders 🎂\n"
        "• Suggesting people to connect with 🤝\n"
        "• Helping you maintain your connections 📅\n\n"
        "Use /help to see all available commands."
    )
    
    update.message.reply_text(welcome_message, parse_mode='Markdown')

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *Sloan Social Reminder Bot Help*\n\n"
        "*Available commands:*\n"
        "/start - Start the bot and see welcome message\n"
        "/help - Show this help message\n"
        "/today - Get today's birthdays and suggestions\n"
        "/setlevel [name] [Close/Normal/Stop] - Set connection level for someone\n"
        "/schedule [HH:MM] - Set your daily reminder time (24h format, EST)\n\n"
        "*How to use:*\n"
        "1. Use /today to see who has a birthday today.\n"
        "2. Update connection levels using the buttons.\n"
        "3. Set a daily reminder with /schedule.\n"
        "4. Use /setlevel to manually update connections.\n\n"
        "The bot will suggest people based on your connection levels!"
    )
    update.message.reply_text(help_text, parse_mode='Markdown')

def today(update: Update, context: CallbackContext) -> None:
    """Send today's birthdays and suggestions."""
    # Get today's birthdays
    birthdays = bot.get_todays_birthdays()
    
    if birthdays:
        # Format birthday message
        bday_names = [person['Name'] for person in birthdays]
        bday_message = f"🎉 *Today's Birthdays:*\n" + "\n".join(f"• {name}" for name in bday_names)
    else:
        bday_message = "No birthdays today! 🎂"
    
    # Get connection suggestions
    suggestions = bot.get_suggestions(count=2)
    
    if suggestions:
        # Format suggestions with buttons
        suggestion_text = "\n\n🤝 *People you might want to connect with:*"
        
        # Send the birthday message first
        update.message.reply_text(bday_message, parse_mode='Markdown')
        
        # Then send suggestions with buttons
        for person in suggestions:
            name = person['Name']
            level = person.get('Connection_Level', 'Normal')
            email = person.get('Personal Email', person.get('MIT Email', 'No email'))
            
            # Create inline keyboard for connection levels
            keyboard = [
                [
                    InlineKeyboardButton("👋 Normal", callback_data=f"level_{name}_Normal"),
                    InlineKeyboardButton("❤️ Close", callback_data=f"level_{name}_Close"),
                    InlineKeyboardButton("✋ Stop", callback_data=f"level_{name}_Stop")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send each suggestion with buttons
            update.message.reply_text(
                f"*{name}* - Last contact: {level}\nHow well do you know them?",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
    else:
        # If no suggestions, just send the birthday message
        update.message.reply_text(bday_message, parse_mode='Markdown')

def set_level(update: Update, context: CallbackContext) -> None:
    """Manually set connection level for a person."""
    try:
        if not context.args or len(context.args) < 2:
            update.message.reply_text(
                "Usage: /setlevel [Name] [Close/Normal/Stop]\n"
                "Example: /setlevel John Close"
            )
            return
            
        # Join all arguments except the last one as the name (to handle names with spaces)
        name = ' '.join(context.args[:-1]).strip()
        level = context.args[-1].strip().capitalize()
        
        if level not in ['Close', 'Normal', 'Stop']:
            update.message.reply_text(
                "Invalid connection level. Please use: Close, Normal, or Stop"
            )
            return
            
        if bot.set_connection_level(name, level):
            update.message.reply_text(
                f"✅ Updated connection level for *{name}* to *{level}*",
                parse_mode='Markdown'
            )
        else:
            update.message.reply_text(
                f"❌ Couldn't update {name}. Name not found or invalid level."
            )
            
    except Exception as e:
        logger.error(f"Error in set_level: {e}")
        update.message.reply_text("❌ An error occurred. Please try again.")

def button_callback(update: Update, context: CallbackContext) -> None:
    """Handle button callbacks for updating connection levels."""
    query = update.callback_query
    query.answer()
    
    try:
        # Parse the callback data
        parts = query.data.split('_')
        if len(parts) != 3:
            logger.error(f"Invalid callback data: {query.data}")
            return
            
        _, name, level = parts
        
        # Check if the name exists in our data
        name_exists = any(p['Name'] == name for p in bot.people)
        
        if name_exists:
            # Update the connection level
            if bot.set_connection_level(name, level):
                query.edit_message_text(
                    text=f"✅ Updated connection level for *{name}* to *{level}*\n\n{query.message.text}",
                    parse_mode='Markdown',
                    reply_markup=query.message.reply_markup
                )
                return
        
        # If we get here, there was an error or name not found
        query.edit_message_text(
            text=f"❌ Couldn't update {name}. Name not found or invalid level.\n\n{query.message.text}",
            parse_mode='Markdown',
            reply_markup=query.message.reply_markup
        )
            
    except Exception as e:
        logger.error(f"Error in button_callback: {e}")
        try:
            query.edit_message_text(
                text=f"❌ An error occurred. Please try again.\n\n{query.message.text}",
                parse_mode='Markdown',
                reply_markup=query.message.reply_markup
            )
        except Exception as e2:
            logger.error(f"Error updating message: {e2}")

def send_daily_update(context: CallbackContext) -> None:
    """Send the daily update to the user."""
    try:
        # Get the chat ID from context
        chat_id = context.job.context
        
        # Get today's birthdays
        birthdays = bot.get_todays_birthdays()
        
        if birthdays:
            # Format birthday message
            bday_names = [person['Name'] for person in birthdays]
            bday_message = f"🎉 *Today's Birthdays:*\n" + "\n".join(f"• {name}" for name in bday_names)
        else:
            bday_message = "No birthdays today! 🎂"
        
        # Get connection suggestions
        suggestions = bot.get_suggestions(count=2)
        
        if suggestions:
            # Format suggestions with buttons
            for person in suggestions:
                name = person['Name']
                last_contact = person.get('Last_Contact', 'Never')
                
                # Create inline keyboard for connection levels
                keyboard = [
                    [
                        InlineKeyboardButton("👋 Normal", callback_data=f"level_{name}_Normal"),
                        InlineKeyboardButton("❤️ Close", callback_data=f"level_{name}_Close"),
                        InlineKeyboardButton("✋ Stop", callback_data=f"level_{name}_Stop")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Send each suggestion with buttons
                context.bot.send_message(
                    chat_id=chat_id,
                    text=f"*{name}* - Last contact: {last_contact}\nHow well do you know them?",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
        
        # Always send the birthday message
        context.bot.send_message(
            chat_id=chat_id,
            text=bday_message,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"Critical error in send_daily_update: {e}")

def set_schedule(update: Update, context: CallbackContext) -> None:
    """Set up the daily schedule for the user."""
    try:
        chat_id = update.effective_chat.id
        
        # Remove any existing job for this chat
        current_jobs = context.job_queue.get_jobs_by_name(str(chat_id))
        for job in current_jobs:
            job.schedule_removal()
        
        # Set default time to 9:00 AM EST (converted to UTC)
        target_time = time(13, 0)  # 9 AM EST is 1 PM UTC
        
        # Check if user provided a specific time
        if context.args:
            try:
                user_time = context.args[0]
                hours, minutes = map(int, user_time.split(':'))
                target_time = time(hours, minutes)
            except (ValueError, IndexError):
                update.message.reply_text(
                    "Invalid time format. Please use HH:MM (24-hour format).\n"
                    "Example: /schedule 09:00"
                )
                return
        
        # Schedule the daily update
        context.job_queue.run_daily(
            send_daily_update,
            time=target_time,
            days=(0, 1, 2, 3, 4, 5, 6),
            context=chat_id,
            name=str(chat_id)
        )
        
        update.message.reply_text(
            f"✅ Daily reminders scheduled for {target_time.strftime('%H:%M')} UTC"
        )
        
    except Exception as e:
        logger.error(f"Error in set_schedule: {e}")
        update.message.reply_text("❌ An error occurred while setting the schedule.")

def main() -> None:
    """Start the bot."""
    # Create the Application without job queue first
    application = Application.builder()\
        .token(os.getenv('TELEGRAM_TOKEN'))\
        .build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Set up daily job (9AM EST = 1PM UTC)
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_daily(
            send_daily_update,
            time=datetime.time(hour=13, minute=0),
            days=tuple(range(7))  # All days of week
        )
    
    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    if not os.getenv('TELEGRAM_TOKEN'):
        logger.error("TELEGRAM_TOKEN environment variable not set")
        print("Error: TELEGRAM_TOKEN environment variable is required")
        exit(1)
    
    main()
