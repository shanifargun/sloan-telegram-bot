# Sloan Bot Deployment Guide

## Render.com Setup (Recommended)
1. Sign up at [render.com](https://render.com)
2. Create new 'Background Worker' service
3. Connect your GitHub repository
4. Configure:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python bot.py`
5. Set environment variables:
   - `TELEGRAM_TOKEN` (from BotFather)

## Required Files
- `SFMBA Directory.csv` (upload via Render dashboard)
- `.env` (for local testing only)

## Daily Reminders
- Configured for 9AM EST (13:00 UTC)
- Runs automatically via polling (no cron needed)

## Backup
- `user_connections.csv` will persist in Render's ephemeral storage

## Free Tier Limits
- 750 hours/month (enough for 24/7 operation)
- Auto-sleeps after inactivity (bot stays awake via polling)
