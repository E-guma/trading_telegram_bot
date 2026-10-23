import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, constants
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
from . import analysis
from . import lang  # The LLM layer

logging.basicConfig(level=logging.INFO)

# Multi-timeframe cache: 300 (5m), 900 (15m), 3600 (1h)
_cache = {
    300: {"t": 0.0, "df": None},
    900: {"t": 0.0, "df": None},
    3600: {"t": 0.0, "df": None},
}


async def get_candles(granularity: int = 900):
    """Fetches data based on the requested timeframe and updates its specific cache."""
    c = _cache.get(granularity, {"t": 0.0, "df": None})
    if c["df"] is None or time.time() - c["t"] > 60:
        c["df"] = await analysis.fetch_candles(granularity=granularity)
        c["t"] = time.time()
        _cache[granularity] = c
    return c["df"]


# --- UI Keyboards ---
def get_main_menu():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 5 Min", callback_data="analyze_300"),
                InlineKeyboardButton("📊 15 Min", callback_data="analyze_900"),
                InlineKeyboardButton("📊 1 Hour", callback_data="analyze_3600"),
            ],
            [InlineKeyboardButton("📈 Trend & Backtest (15m)", callback_data="trend")],
            [InlineKeyboardButton("🧪 Run Walk-Forward", callback_data="backtest")],
        ]
    )


def get_refresh_button(action_type: str):
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Refresh Data", callback_data=action_type)],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")],
        ]
    )


# --- Initialization ---
async def post_init(app: Application):
    """Initializes the blue menu button next to the chat bar."""
    await app.bot.set_my_commands(
        [
            ("start", "Open the main control panel"),
            ("analyze", "Get current 15-min outlook & indicators"),
            ("trend", "Analyze with historical accuracy overlay"),
            ("backtest", "Run rigorous walk-forward test"),
            ("risk", "Calculate position sizing"),
            ("subscribe", "Get automated reports every 15 min"),
            ("unsubscribe", "Stop automated reports"),
        ]
    )


# --- Handlers ---
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>V100 (1s) Trading Terminal</b>\n\n"
        "Select an action below or use the menu to navigate.\n"
        "<i>Expect ~50% baseline accuracy: the index is RNG-generated. Not financial advice.</i>"
    )
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text, parse_mode="HTML", reply_markup=get_main_menu()
        )
    else:
        await update.message.reply_text(
            text, parse_mode="HTML", reply_markup=get_main_menu()
        )


async def handle_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Routes button clicks to the appropriate logic."""
    query = update.callback_query
    await query.answer()  # Stops the button from flashing

    # Send typing indicator
    await ctx.bot.send_chat_action(
        chat_id=query.message.chat_id, action=constants.ChatAction.TYPING
    )

    if query.data.startswith("analyze_"):
        granularity = int(query.data.split("_")[1])
        await _run_analyze(query.message, granularity)
    elif query.data == "trend":
        await _run_trend(query.message)
    elif query.data == "backtest":
        await _run_backtest(query.message)
    elif query.data == "menu":
        await start(update, ctx)


# --- Core Logic Functions ---
async def _run_analyze(message, granularity: int = 900):
    tf_labels = {300: "5-Min", 900: "15-Min", 3600: "1-Hour"}
    label = tf_labels.get(granularity, "Custom")

    # Determine if we are editing an inline message or creating a new reply
    if not message.reply_markup:
        msg = await message.reply_text(f"Fetching {label} live data...")
    else:
        msg = await message.edit_text(f"Fetching {label} live data...")

    try:
        df = await get_candles(granularity)
        a = await asyncio.to_thread(analysis.analyse, df, label)
        await msg.edit_text(
            analysis.format_report(a),
            parse_mode="HTML",
            reply_markup=get_refresh_button(f"analyze_{granularity}"),
        )
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


async def analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING
    )
    await _run_analyze(update.message, 900)


async def _run_trend(message):
    msg = (
        await message.reply_text("Fetching data and backtesting...")
        if not message.reply_markup
        else await message.edit_text("Fetching data and backtesting...")
    )
    try:
        df = await get_candles(900)
        a = await asyncio.to_thread(analysis.analyse, df, "15-Min")
        bt = await asyncio.to_thread(analysis.walk_forward, df)
        await msg.edit_text(
            analysis.format_report(a, bt),
            parse_mode="HTML",
            reply_markup=get_refresh_button("trend"),
        )
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


async def trend(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING
    )
    await _run_trend(update.message)


async def _run_backtest(message):
    msg = (
        await message.reply_text("Running walk-forward test...")
        if not message.reply_markup
        else await message.edit_text("Running walk-forward test...")
    )
    try:
        df = await get_candles(900)
        bt = await asyncio.to_thread(analysis.walk_forward, df)
        folds = ", ".join(f"{x:.1%}" for x in bt["folds"])
        text = (
            f"<b>Walk-forward ({len(bt['folds'])} folds, n={bt['n_test']})</b>\n"
            f"Accuracy: <code>{bt['accuracy']:.1%} ± {bt['ci']:.1%}</code>\n"
            f"Majority-class baseline: {bt['baseline']:.1%}\n"
            f"Per fold: {folds}"
        )
        await msg.edit_text(
            text, parse_mode="HTML", reply_markup=get_refresh_button("backtest")
        )
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


async def backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING
    )
    await _run_backtest(update.message)


async def risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        balance, pct, stop = (float(x) for x in ctx.args[:3])
        amount = balance * pct / 100
        await update.message.reply_text(
            f"<b>Risk Profile</b>\n"
            f"Risk amount: <code>${amount:.2f}</code>\n"
            f"Stake per point of stop: <code>${amount / stop:.4f}</code>\n\n"
            "<i>(Check your broker's contract specs; payout per point varies.)</i>",
            parse_mode="HTML",
        )
    except (ValueError, ZeroDivisionError):
        await update.message.reply_text(
            "Usage: <code>/risk 1000 1 25</code> (balance, risk %, stop distance in points)",
            parse_mode="HTML",
        )


async def _push(ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = ctx.job.chat_id
    try:
        df = await get_candles(900)
        a = await asyncio.to_thread(analysis.analyse, df, "15-Min")
        await ctx.bot.send_message(
            chat_id, analysis.format_report(a), parse_mode="HTML"
        )
    except Exception as e:
        await ctx.bot.send_message(chat_id, f"Scheduled update failed: {e}")


async def subscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    for j in ctx.job_queue.get_jobs_by_name(str(chat_id)):
        j.schedule_removal()
    ctx.job_queue.run_repeating(
        _push, interval=900, first=5, chat_id=chat_id, name=str(chat_id)
    )
    await update.message.reply_text(
        "✅ Subscribed: Report scheduled every 15 minutes. Use /unsubscribe to stop."
    )


async def unsubscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    jobs = ctx.job_queue.get_jobs_by_name(str(update.effective_chat.id))
    for j in jobs:
        j.schedule_removal()
    await update.message.reply_text(
        "🛑 Unsubscribed." if jobs else "No active subscription found."
    )


# --- Conversational LLM Layer ---
async def chat_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Fallback handler for natural language chatting using Gemini."""
    user_text = update.message.text
    chat_id = update.effective_chat.id

    # Show a typing indicator to the user while Gemini thinks
    await ctx.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # Grab the latest 15-minute cached data so the LLM knows current market conditions
    current_data = {}
    if _cache[900]["df"] is not None:
        try:
            current_data = await asyncio.to_thread(
                analysis.analyse, _cache[900]["df"], "15-Min"
            )
        except Exception:
            pass  # Failsafe if the cache is empty or invalid

    try:
        reply = await lang.chat_with_data(user_text, current_data)
        # Parse mode is HTML to render Gemini's formatting
        await update.message.reply_text(reply, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ AI connection error: {e}")


def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    render_url = "https://fx-derivbot.onrender.com"

    # Initialize Application with the post_init hook
    app = Application.builder().token(token).post_init(post_init).build()

    for name, fn in [
        ("start", start),
        ("analyze", analyze),
        ("trend", trend),
        ("backtest", backtest),
        ("risk", risk),
        ("subscribe", subscribe),
        ("unsubscribe", unsubscribe),
    ]:
        app.add_handler(CommandHandler(name, fn))

    # Catch button clicks
    app.add_handler(CallbackQueryHandler(handle_button))

    # Catch standard text and pass it to Gemini
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    port = int(os.environ.get("PORT", 10000))
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path="webhook",
        webhook_url=f"{render_url}/webhook",
    )


if __name__ == "__main__":
    main()
