import json
import os
from typing import Any

from openai import APIConnectionError, APITimeoutError, RateLimitError

from openai_client import create_client


def get_weather(city: str) -> dict[str, Any]:
    fake_data = {
        "seattle": {"city": "Seattle", "temperature_c": 18, "condition": "Cloudy"},
        "minneapolis": {"city": "Minneapolis", "temperature_c": 22, "condition": "Sunny"},
        "bangalore": {"city": "Bangalore", "temperature_c": 24, "condition": "Light Rain"},
    }
    key = city.strip().lower()
    return fake_data.get(
        key,
        {"city": city, "temperature_c": 20, "condition": "Partly Cloudy"},
    )


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, for example Seattle.",
                        }
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with memory across turns. "
                "If the user asks about weather, call get_weather."
            ),
        }
    ]

    print("Chatbot started. Type 'quit' or 'exit' to stop.")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Ending chat.")
            break

        messages.append({"role": "user", "content": user_text})

        try:
            while True:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                )

                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls or []

                if not tool_calls:
                    assistant_text = assistant_message.content or ""
                    print(f"Assistant: {assistant_text}")
                    messages.append({"role": "assistant", "content": assistant_text})
                    break

                messages.append(assistant_message.model_dump(exclude_none=True))

                for tool_call in tool_calls:
                    if tool_call.function.name != "get_weather":
                        continue

                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    city = str(args.get("city", "Unknown"))
                    tool_result = get_weather(city)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

        except RateLimitError:
            print("Assistant: Rate limit reached. Please wait a moment and try again.")
            messages.pop()
            continue
        except APITimeoutError:
            print("Assistant: Request timed out. Please retry.")
            messages.pop()
            continue
        except APIConnectionError:
            print("Assistant: Network/connection error. Check connectivity and try again.")
            messages.pop()
            continue
        except Exception as exc:
            print(f"Assistant: Unexpected error: {exc}")
            messages.pop()
            continue


if __name__ == "__main__":
    main()
