"""Ollama inference backend for Phase 1.

Upstream PrefEval builds *raw* chat-template strings for Llama/Mistral (that is what
Bedrock received), so we send them through Ollama's /api/generate with raw=True:
the prompt text reaching the model is exactly what the upstream code produced.
"""

import json
import logging
import time

import requests

log = logging.getLogger("phase1.backend")

# Literal BOS strings upstream writes at the start of raw prompts (Llama-3, Mistral). Ollama adds BOS
# itself for both (verified: Mistral "hello" -> 3 tokens, "<s>hello" -> 4), so the literal is stripped.
LEADING_BOS = ("<|begin_of_text|>", "<s>")
# Bump whenever prompt normalisation / decoding changes; stored in every result record so results
# from different backend behaviour are never silently mixed (aggregate.py filters on it).
#   1: initial (double BOS, untrimmed self-critic prompts)  -- invalid, discarded
#   2: single BOS + trailing-whitespace trim (same rule for Llama-3 and Mistral; Mistral first run under rev 2)
BACKEND_REV = 2


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached after all retries."""


class OllamaBackend:
    def __init__(self, tag, num_ctx, host="http://localhost:11434", seed=41, temperature=0.0,
                 keep_alive="30m", max_retries=5, timeout=600):
        self.tag = tag
        self.num_ctx = num_ctx
        self.host = host.rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self.timeout = timeout
        # Per-question accumulator, reset by the runner (see begin_question / end_question).
        self.calls = []

    # ---------------------------------------------------------------- server info
    def server_version(self):
        return requests.get(f"{self.host}/api/version", timeout=10).json().get("version")

    def model_info(self):
        r = requests.post(f"{self.host}/api/show", json={"model": self.tag}, timeout=30)
        r.raise_for_status()
        d = r.json()
        return {
            "tag": self.tag,
            # /api/show has no digest in Ollama 0.34; /api/tags lists it per installed model.
            "digest": d.get("digest") or self._digest_from_tags(),
            "details": d.get("details", {}),
            "modified_at": d.get("modified_at"),
        }

    def _digest_from_tags(self):
        try:
            models = requests.get(f"{self.host}/api/tags", timeout=10).json().get("models", [])
            return next((m.get("digest") for m in models if m.get("name") == self.tag), None)
        except requests.RequestException:
            return None

    # ---------------------------------------------------------------- generation
    def generate(self, prompt, max_tokens, purpose="answer"):
        """Generate a completion for a raw prompt string. Returns the response text."""
        if not isinstance(prompt, str):
            raise TypeError(f"raw prompt must be str, got {type(prompt).__name__}")
        # Upstream Llama prompts start with a literal <|begin_of_text|> (Bedrock adds no BOS itself).
        # llama.cpp/Ollama always prepends BOS, so keeping the literal would give the model TWO BOS
        # tokens (verified: "hello" -> 2 tokens, "<|begin_of_text|>hello" -> 3). Strip it so the
        # model sees exactly one, as on Bedrock.
        prompt = prompt.lstrip()
        for bos in LEADING_BOS:
            if prompt.startswith(bos):
                prompt = prompt[len(bos):]
                break
        # Upstream's Llama self-critic templates are indented triple-quoted strings, so they END with
        # "<|end_header_id|>\n        ". With that trailing whitespace Llama-3 emits <|eot_id|>
        # immediately (empty critique -> empty answer -> ~0% accuracy). Stripping trailing spaces/tabs
        # fixes it; all other upstream prompts have no trailing spaces, so they are unaffected
        # (Mistral's "<choice>\n" prefill keeps its newline).
        prompt = prompt.rstrip(" \t")
        payload = {
            "model": self.tag,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
                r.raise_for_status()
                d = r.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_err = e
                wait = min(60, 5 * attempt)
                log.warning("Ollama call failed (attempt %d/%d): %s -- retrying in %ds",
                            attempt, self.max_retries, e, wait)
                time.sleep(wait)
                continue
            # In Ollama 0.34 prompt_eval_count is the FULL prompt length (cached tokens included;
            # prompt_eval_cached_count is the cached subset) -- verified empirically.
            prompt_tokens = d.get("prompt_eval_count") or 0
            call = {
                "purpose": purpose,
                "latency_s": round(time.time() - t0, 3),
                "prompt_tokens": prompt_tokens,
                "output_tokens": d.get("eval_count"),
                "done_reason": d.get("done_reason"),
                # If the prompt fills the context window Ollama silently drops the *start* of
                # the prompt -- which is where the user's preference lives. Flag it.
                "truncated": prompt_tokens >= self.num_ctx - max_tokens,
            }
            if call["truncated"]:
                log.error("Prompt likely truncated: %d prompt tokens with num_ctx=%d", prompt_tokens, self.num_ctx)
            self.calls.append(call)
            return d.get("response", "")
        raise OllamaUnavailable(f"Ollama failed after {self.max_retries} attempts: {last_err}")

    def upstream_generate_message(self, client, model_id, model_type, system_prompt=None, messages=None,
                                  max_tokens=None, temperature=0, max_retries=10):
        """Drop-in replacement for upstream `generate_message` (same signature).

        Monkey-patched into the vendored upstream self-critic handlers so they run unchanged.
        """
        purpose = "critic" if max_tokens and max_tokens > 50 else "answer"
        return self.generate(messages, max_tokens, purpose=purpose)
