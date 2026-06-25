import json
import os
from typing import Any

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

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. For weather questions, call the get_weather tool.",
        },
        {"role": "user", "content": "What is the weather in Seattle today?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
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

    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    first_message = first.choices[0].message
    print("First model response (before tool execution):")
    print(first_message.model_dump())

    tool_calls = first_message.tool_calls or []
    if not tool_calls:
        print("Model did not request a tool call.")
        return

    messages.append(first_message.model_dump(exclude_none=True))

    for tool_call in tool_calls:
        if tool_call.function.name != "get_weather":
            continue

        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        city = str(args.get("city", "Unknown"))
        tool_result = get_weather(city)

        print("\nTool call requested:")
        print({
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": args,
        })
        print("Tool result:")
        print(tool_result)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result),
            }
        )

    final = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0,
    )

    final_text = final.choices[0].message.content or ""
    print("\nFinal assistant answer:")
    print(final_text)


if __name__ == "__main__":
    main()
