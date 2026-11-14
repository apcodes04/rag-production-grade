"""Which models can this API key actually call?  python scripts/list_models.py

Model ids are retired on a schedule (gemini-2.0-flash was shut down on
2026-06-01), so never trust a model name from a tutorial, including this one.
Ask the API.
"""
import _bootstrap  # noqa: F401

from rag.settings import get_settings


def main() -> None:
    from google import genai
    settings = get_settings()
    if not settings.gemini_api_key:
        print("No GEMINI_API_KEY set.")
        return
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    print(f"{'model':45} {'in/out tokens':>16}  supported actions")
    for model in client.models.list():
        actions = ",".join(getattr(model, "supported_actions", None) or [])
        limits = f"{getattr(model, 'input_token_limit', '?')}/{getattr(model, 'output_token_limit', '?')}"
        print(f"{model.name:45} {limits:>16}  {actions}")
    print(f"\nconfigured: llm={settings.llm_model}  embedding={settings.embedding_model}")


if __name__ == "__main__":
    main()
