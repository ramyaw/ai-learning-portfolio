import os

from openai import APIConnectionError, APITimeoutError, RateLimitError

from openai_client import create_client


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep answers concise.",
        }
    ]

    print("Simple chat loop started. Type 'quit' or 'exit' to stop.")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Ending chat.")
            break

        messages.append({"role": "user", "content": user_text})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
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

        assistant_text = response.choices[0].message.content or ""
        print(f"Assistant: {assistant_text}")

        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
