# Sloan Social Reminder Bot

A Telegram bot designed to help Sloan Fellows stay connected by sending daily reminders about birthdays and suggesting people to reach out to.

## Features

- 🎂 Daily birthday reminders
- 👋 Smart suggestions for who to connect with
- 🔄 Easy connection level updates (Close/Normal/Stop)
- ⏰ Scheduled daily messages at 9:00 AM EST
- 📱 Interactive buttons for quick updates

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sloan-messaging-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Add your Telegram bot token to the `.env` file

4. **Prepare your CSV file**
   - Place your `SFMBA Directory.csv` file in the project directory
   - Ensure it has the required columns: `Name`, `Personal Email`, `MIT Email`, `Birthday Month`, `Birthday Day`

## Running the Bot

### Local Development

```bash
python bot.py
```

### Production Deployment

For production, consider deploying to a cloud provider like:

1. **Render**
   - Create a new Web Service
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `python bot.py`
   - Add the `TELEGRAM_TOKEN` environment variable

2. **Railway**
   - Create a new project
   - Add the `TELEGRAM_TOKEN` environment variable
   - Deploy from GitHub

3. **Google Cloud Run**
   - Containerize the app with Docker
   - Deploy to Cloud Run
   - Set up a Cloud Scheduler to keep it warm

## Usage

### Commands

- `/start` - Welcome message and bot introduction
- `/help` - Show help information
- `/today` - Get today's birthdays and suggestions
- `/setlevel Name Level` - Manually set connection level (Close/Normal/Stop)
- `/schedule` - Set up daily reminders at 9:00 AM EST

### Updating Connection Levels

1. When you receive the daily message, you'll see buttons below each suggestion
2. Click the appropriate button (Close, Normal, Stop) to update the connection level
3. The bot will confirm your update

## Data Storage

- Connection levels are stored in `user_connections.csv`
- The original CSV file is never modified
- Back up this file if you're redeploying the bot

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT
