# Bring your own engine

LMM does not ship a model and does not want to pick one for you. It needs an
engine to turn a retrieved record into a sentence — and any of these will do:

=== "Local (default)"

    ```bash
    # nothing to set — a local Qwen2.5-3B-Instruct on your own machine
    export LMM_MODEL_PATH=/path/to/any-instruct-model   # ...or any HF model
    ```

=== "OpenAI"

    ```bash
    export LMM_BACKEND=openai
    export OPENAI_API_KEY=sk-...
    export LMM_MODEL=gpt-4o-mini
    ```

=== "OpenRouter / Groq / vLLM / Ollama"

    ```bash
    export LMM_BACKEND=openai
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or your server
    export OPENAI_API_KEY=...
    export LMM_MODEL=meta-llama/llama-3.3-70b-instruct
    ```

=== "Azure OpenAI"

    ```bash
    export LMM_BACKEND=azure          # its own client: a deployment name
    ```                               # is a shape only Azure has

=== "llama.cpp"

    ```bash
    export LMM_BACKEND=gguf           # int4 — the fast option on cheap hardware
    ```

**Nothing about the architecture changes with the engine, and that is the
point.** The graph, the gate and the derivation are plain Python; they never
call the engine and never learn which one is running. What the engine
decides is how a record is *worded* — not whether it may be spoken. So a
weaker engine costs you fluency, never provenance; an unsupported claim is
dropped by the same rule whichever model produced it.

Keys are read from the environment or a `.env` outside git. On-premise
deployments point `OPENAI_BASE_URL` at their own vLLM/Ollama server and no
document ever leaves the building.
