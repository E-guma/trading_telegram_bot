import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import analysis

load_dotenv()
logging.basicConfig(level=logging.INFO)

_cache = {"t": 0.0, "df": None}


async def get_candles():
    if _cache["df"] is None or time.time() - _cache["t"] > 60:
        _cache["df"] = await analysis.fetch_candles()
        _cache["t"] = time.time()
    return _cache["df"]


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "V100 (1s) analysis bot\n\n"
        "/analyze - indicators + 15-min model lean\n"
        "/trend - same as /analyze, with out-of-sample accuracy\n"
        "/backtest - walk-forward accuracy only\n"
        "/risk <balance> <risk%> <stop_distance> - position sizing\n"
        "/subscribe - report every 15 min\n"
        "/unsubscribe\n\n"
        "Expect ~50% accuracy: the index is RNG-generated. Not financial advice."
    )


async def analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Fetching data...")
    try:
        df = await get_candles()
        a = await asyncio.to_thread(analysis.analyse, df)
        await msg.edit_text(analysis.format_report(a))
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


async def trend(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Fetching data and backtesting...")
    try:
        df = await get_candles()
        a = await asyncio.to_thread(analysis.analyse, df)
        bt = await asyncio.to_thread(analysis.walk_forward, df)
        await msg.edit_text(analysis.format_report(a, bt))
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


async def backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Running walk-forward test...")
    try:
        df = await get_candles()
        bt = await asyncio.to_thread(analysis.walk_forward, df)
        folds = ", ".join(f"{x:.1%}" for x in bt["folds"])
        await msg.edit_text(
            f"Walk-forward ({len(bt['folds'])} folds, n={bt['n_test']})\n"
            f"Accuracy: {bt['accuracy']:.1%} ± {bt['ci']:.1%}\n"
            f"Majority-class baseline: {bt['baseline']:.1%}\n"
            f"Per fold: {folds}"
        )
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


async def risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        balance, pct, stop = (float(x) for x in ctx.args[:3])
        amount = balance * pct / 100
        await update.message.reply_text(
            f"Risk amount: {amount:.2f}\n"
            f"Stake per point of stop: {amount / stop:.4f}\n"
            "(Check your broker's contract specs; payout per point varies.)"
        )
    except (ValueError, ZeroDivisionError):
        await update.message.reply_text("Usage: /risk 1000 1 25  (balance, risk %, stop distance in points)")


async def _push(ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = ctx.job.chat_id
    try:
        df = await analysis.fetch_candles()
        a = await asyncio.to_thread(analysis.analyse, df)
        await ctx.bot.send_message(chat_id, analysis.format_report(a))
    except Exception as e:
        await ctx.bot.send_message(chat_id, f"Scheduled update failed: {e}")


async def subscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    for j in ctx.job_queue.get_jobs_by_name(str(chat_id)):
        j.schedule_removal()
    ctx.job_queue.run_repeating(_push, interval=900, first=5, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text("Subscribed: report every 15 minutes. /unsubscribe to stop.")


async def unsubscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    jobs = ctx.job_queue.get_jobs_by_name(str(update.effective_chat.id))
    for j in jobs:
        j.schedule_removal()
    await update.message.reply_text("Unsubscribed." if jobs else "No active subscription.")


def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    for name, fn in [
        ("start", start), ("analyze", analyze), ("trend", trend), ("backtest", backtest),
        ("risk", risk), ("subscribe", subscribe), ("unsubscribe", unsubscribe),
    ]:
        app.add_handler(CommandHandler(name, fn))
    app.run_polling()


if __name__ == "__main__":
    main()
