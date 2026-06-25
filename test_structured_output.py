import json
import os
from pprint import pprint

from openai_client import create_client


def test_json_in_prompt() -> None:
    """Request JSON output by asking in the prompt and parse it."""
    print("=" * 70)
    print("STRUCTURED OUTPUT: JSON IN PROMPT")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    
    prompt = """Generate a user profile in valid JSON format with these fields:
- name (string)
- age (integer)
- email (string)
- interests (array of strings)

Return ONLY the JSON, no other text."""
    
    print(f"\nPrompt: {prompt}\n")
    print("-" * 70)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that returns data as valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # Keep it deterministic
    )
    
    response_text = response.choices[0].message.content
    print(f"Raw response:\n{response_text}\n")
    
    try:
        parsed = json.loads(response_text)
        print("Parsed JSON:")
        pprint(parsed)
        print(f"\nExtracted fields:")
        print(f"  Name: {parsed.get('name')}")
        print(f"  Age: {parsed.get('age')}")
        print(f"  Email: {parsed.get('email')}")
        print(f"  Interests: {parsed.get('interests')}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")


def test_json_response_format() -> None:
    """Request JSON using response_format parameter (if supported by model)."""
    print("\n" + "=" * 70)
    print("STRUCTURED OUTPUT: response_format='json_object'")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    
    prompt = """Generate a product review in JSON format with these fields:
- product_name (string)
- rating (number 1-5)
- pros (array of strings)
- cons (array of strings)
- summary (string)"""
    
    print(f"\nPrompt: {prompt}\n")
    print("-" * 70)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a product reviewer that outputs structured JSON data."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        response_text = response.choices[0].message.content
        print(f"Raw response:\n{response_text}\n")
        
        parsed = json.loads(response_text)
        print("Parsed JSON:")
        pprint(parsed)
        print(f"\nExtracted fields:")
        print(f"  Product: {parsed.get('product_name')}")
        print(f"  Rating: {parsed.get('rating')}/5")
        print(f"  Pros: {parsed.get('pros')}")
        print(f"  Cons: {parsed.get('cons')}")
        print(f"  Summary: {parsed.get('summary')}")
    except Exception as e:
        print(f"Error with response_format parameter: {e}")
        print("(This model version may not support response_format, falling back to prompt-based approach)")


def test_complex_structure() -> None:
    """Parse a more complex nested JSON structure."""
    print("\n" + "=" * 70)
    print("STRUCTURED OUTPUT: COMPLEX NESTED JSON")
    print("=" * 70)
    
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = create_client()
    
    prompt = """Generate a sample API response for a weather API in JSON format with:
{
  "location": {
    "city": string,
    "country": string
  },
  "current": {
    "temperature": number (in Celsius),
    "condition": string
  },
  "forecast": [
    {
      "day": string,
      "high": number,
      "low": number,
      "condition": string
    }
  ]
}

Return ONLY valid JSON."""
    
    print(f"\nPrompt: {prompt}\n")
    print("-" * 70)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a weather API that returns structured JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    
    response_text = response.choices[0].message.content
    print(f"Raw response:\n{response_text}\n")
    
    try:
        parsed = json.loads(response_text)
        print("Parsed JSON (full structure):")
        pprint(parsed)
        
        print(f"\nExtracted values:")
        print(f"  Location: {parsed['location']['city']}, {parsed['location']['country']}")
        print(f"  Current: {parsed['current']['temperature']}°C, {parsed['current']['condition']}")
        print(f"  Forecast:")
        for day in parsed.get('forecast', []):
            print(f"    {day['day']}: High {day['high']}°C, Low {day['low']}°C ({day['condition']})")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing or extracting JSON: {e}")


def main() -> None:
    test_json_in_prompt()
    test_json_response_format()
    test_complex_structure()


if __name__ == "__main__":
    main()
