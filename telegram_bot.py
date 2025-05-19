import os
import logging
import schedule
import time
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from portfoliomanager import InvestmentAdvisor, format_currency, format_percentage

# Load your bot token from environment or hardcoded (replace before pushing to GitHub)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7888039321:AAGea6o89KNsrTIzQIVpn4nnsfu-2DKJCTo")

advisor = InvestmentAdvisor(initial_capital=50000)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I’m StockAdvisorAI — your GenAI-powered investment assistant.\n\n"
        "Available commands:\n"
        "/report - Get portfolio + market report"
    )

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = advisor.generate_report()
    stats = report['portfolio_summary']
    message = (
        f"📊 Portfolio Report\n"
        f"Date: {report['date']}\n"
        f"Portfolio Value: {format_currency(stats['current_value'])}\n"
        f"Returns: {format_percentage(stats['returns_pct'])}\n"
        f"Cash: {format_currency(stats['cash'])}\n"
        f"Direction: {report['market_outlook']['market_direction']}\n"
        f"Top Sectors: {', '.join([x[0] for x in report['market_outlook']['top_sectors']])}"
    )
    await update.message.reply_text(message)

def run_scheduler(app):
    # Optional: Send daily updates at 9:15 AM IST
    async def send_daily_report():
        chat_id = os.getenv("ADMIN_CHAT_ID")  # Set your chat ID for private messages
        if chat_id:
            report = advisor.generate_report()
            message = f"📢 Daily Summary:\nValue: {format_currency(report['portfolio_summary']['current_value'])}"
            await app.bot.send_message(chat_id=chat_id, text=message)

    schedule.every().day.at("09:15").do(lambda: app.create_task(send_daily_report()))

    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report))
    print("🤖 Bot is running...")

    # Optional background job
    # import threading
    # threading.Thread(target=run_scheduler, args=(app,), daemon=True).start()

    app.run_polling()

if __name__ == "__main__":
    main()
