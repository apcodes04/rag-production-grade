"""One real call to each Gemini endpoint, to prove the key and models work.

Run this before blaming the pipeline: it separates "my API setup is wrong"
from "my retrieval is wrong".  python scripts/smoke_live.py
"""
import _bootstrap  # noqa: F401

from rag.factory import build_embedder, build_llm
from rag.logging_setup import setup_logging
from rag.settings import get_settings


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    print(f"provider={settings.provider} embedding={settings.embedding_model} "
          f"dim={settings.embedding_dim} llm={settings.llm_model}")

    embedder = build_embedder(settings)
    vectors = embedder.embed_documents(["hello world", "refund policy"])
    query = embedder.embed_query("how do refunds work")
    print(f"embeddings ok: shape={vectors.shape}, norm={float((vectors[0] ** 2).sum()) ** 0.5:.4f}, "
          f"cosine(query, refund doc)={float(vectors[1] @ query):.4f}")

    llm = build_llm(settings)
    # If this raises NOT_FOUND, the model id is retired or not available on your
    # key: run `python scripts/list_models.py` and pin one that is listed.
    result = llm.generate("Answer in exactly three words.", "What is RAG?",
                          max_output_tokens=settings.llm_max_output_tokens, temperature=0.0)
    print(f"llm ok: finish={result.finish_reason} tokens_in={result.prompt_tokens} "
          f"tokens_out={result.output_tokens} thinking_tokens={result.thinking_tokens}\n"
          f"  {result.text!r}")
    if not result.text:
        print("  EMPTY ANSWER: the model spent the whole budget thinking. Raise "
              "LLM_MAX_OUTPUT_TOKENS or set LLM_THINKING_LEVEL=minimal in .env")

    # An untested failover is not a failover. A listed model can still answer
    # 404 for this account, and you would only find out during an outage.
    if settings.llm_fallback_model:
        from rag.providers.gemini import GeminiLLM
        backup = GeminiLLM(settings.gemini_api_key.get_secret_value(),
                           settings.llm_fallback_model, max_retries=0,
                           timeout_s=settings.llm_timeout_s,
                           thinking_level=settings.llm_thinking_level)
        try:
            check = backup.generate("Answer in one word.", "What is RAG?",
                                    max_output_tokens=settings.llm_max_output_tokens,
                                    temperature=0.0)
            print(f"fallback ok: {settings.llm_fallback_model} -> {check.text.strip()[:40]!r}")
        except Exception as exc:                 # noqa: BLE001
            print(f"FALLBACK BROKEN: {settings.llm_fallback_model} -> "
                  f"{str(exc).split('{')[0].strip()[:90]}\n  Run: python scripts/probe_models.py")


if __name__ == "__main__":
    main()
