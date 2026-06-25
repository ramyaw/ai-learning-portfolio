import os
from pprint import pprint

from openai_client import create_client


def test_temperature() -> None:
    """Test how temperature affects response variability."""
    print("=" * 70)
    print("TEMPERATURE EXPERIMENT")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    prompt = "What is the meaning of life?"
    
    temperatures = [0.0, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp} (0=deterministic, 2=very random)")
        print("-" * 70)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a philosophical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Tokens used: {response.usage.total_tokens}")


def test_max_tokens() -> None:
    """Test how max_tokens constrains response length."""
    print("\n" + "=" * 70)
    print("MAX_TOKENS EXPERIMENT")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    prompt = "Explain quantum computing."
    
    max_token_limits = [20, 50, 150]
    
    for limit in max_token_limits:
        print(f"\nMax tokens: {limit}")
        print("-" * 70)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a technical expert."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=limit,
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Tokens used: {response.usage.completion_tokens} completion, {response.usage.prompt_tokens} prompt")


def test_top_p() -> None:
    """Test how top_p (nucleus sampling) affects response diversity."""
    print("\n" + "=" * 70)
    print("TOP_P EXPERIMENT")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    prompt = "Write a creative sentence about AI."
    
    top_p_values = [0.1, 0.5, 0.9, 1.0]
    
    for p in top_p_values:
        print(f"\nTop_p: {p} (0.1=conservative, 1.0=all tokens allowed)")
        print("-" * 70)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative writer."},
                {"role": "user", "content": prompt},
            ],
            top_p=p,
            temperature=0.7,  # Set a non-zero temp for top_p to have effect
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Tokens used: {response.usage.total_tokens}")


def test_combined() -> None:
    """Test a combination of parameters together."""
    print("\n" + "=" * 70)
    print("COMBINED PARAMETERS EXPERIMENT")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    prompt = "Tell me a story."
    
    configs = [
        {"temp": 0.0, "max_tokens": 50, "top_p": 1.0, "label": "Deterministic, concise"},
        {"temp": 1.5, "max_tokens": 100, "top_p": 0.8, "label": "Creative, medium length"},
        {"temp": 0.5, "max_tokens": 200, "top_p": 0.95, "label": "Balanced, detailed"},
    ]
    
    for config in configs:
        print(f"\n{config['label']}")
        print(f"  temperature={config['temp']}, max_tokens={config['max_tokens']}, top_p={config['top_p']}")
        print("-" * 70)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": prompt},
            ],
            temperature=config["temp"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")


def main() -> None:
    test_temperature()
    test_max_tokens()
    test_top_p()
    test_combined()


if __name__ == "__main__":
    main()
