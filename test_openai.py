import os
from pprint import pprint

from openai_client import create_client


def run_completion(client, model: str, messages: list[dict[str, str]], label: str) -> None:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    print(f"=== {label} ===")
    pprint(response.model_dump())
    print()
    print(f"model: {response.model}")
    print(f"finish_reason: {response.choices[0].finish_reason}")
    print(f"usage: {response.usage}")
    print(f"assistant_text: {response.choices[0].message.content}")
    print()


def main() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    user_message = {"role": "user", "content": "Introduce yourself in one sentence."}

    run_completion(
        client,
        model,
        [user_message],
        "Without system message",
    )

    run_completion(
        client,
        model,
        [
            {
                "role": "system",
                "content": "You are a pirate assistant. Reply in one short sentence and mention the sea.",
            },
            user_message,
        ],
        "With system message",
    )


if __name__ == "__main__":
    main()