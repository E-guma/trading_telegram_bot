import json

from google import genai
from google.genai import types

# Initialize the modern client (it automatically reads the GEMINI_API_KEY environment variable)
client = genai.Client()


async def chat_with_data(user_query: str, market_data: dict) -> str:
    """Answers natural language questions using the latest cached market data."""

    prompt = (
        f"User question: {user_query}\n\n"
        f"Live Market Data Context:\n{json.dumps(market_data, indent=2)}"
    )

    # Asynchronous call using the new SDK syntax and the active Lite model
    response = await client.aio.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.5,
            system_instruction=(
                "You are a helpful quantitative trading assistant tracking the Volatility 100 (1s) Index. "
                "You are responding directly to users as a Telegram bot. "
                "You must format all your responses using standard HTML tags supported by Telegram "
                "(e.g., <b> for bold, <i> for italic, <code> for monospace). "
                "Use appropriate emojis to add visual interactivity to your responses. "
                "Answer user questions based on the provided live market data context. Be concise, direct, "
                "and never give explicit financial advice."
            ),
        ),
    )

    return response.text
