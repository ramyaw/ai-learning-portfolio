import os
from pathlib import Path
import ssl

from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI

try:
    import truststore
except ImportError:
    truststore = None


def create_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    ca_bundle = os.getenv("OPENAI_CA_BUNDLE", "").strip()

    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in .env before running this script.")

    client_kwargs = {"api_key": api_key}
    if truststore is not None:
        client_kwargs["http_client"] = DefaultHttpxClient(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        )
    elif ca_bundle:
        bundle_path = Path(ca_bundle)
        if not bundle_path.is_file():
            raise FileNotFoundError(f"OPENAI_CA_BUNDLE not found: {bundle_path}")
        client_kwargs["http_client"] = DefaultHttpxClient(verify=str(bundle_path))

    return OpenAI(**client_kwargs)