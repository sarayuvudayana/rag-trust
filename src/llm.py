"""
Module 3 - Step 2: LLM Interface & Provider Connector (llm.py)
-------------------------------------------------------------
Purpose:
    Provide a unified, provider-agnostic interface to generate responses using LLMs
    (Ollama / Llama 3, Groq / OpenAI-compatible API, HuggingFace Local, or Offline Extractive Fallback).

Why this is needed:
    Decouples the RAG logic from specific model providers, allowing you to swap between
    local Llama, cloud APIs (Groq/OpenAI), or offline testing seamlessly.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any


class LLMClient:
    def __init__(
        self,
        provider: str = "auto",
        model_name: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initializes the LLM Client.

        Args:
            provider (str): 'ollama', 'openai_compatible', 'groq', or 'auto' (detects available backend).
            model_name (str): The model identifier (e.g., 'llama3.2', 'llama3-8b-8192', 'gpt-4o-mini').
            api_key (str, optional): API key for cloud providers (or set GROQ_API_KEY / OPENAI_API_KEY).
            base_url (str, optional): Base URL for API calls.
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

        if self.provider == "auto":
            self.provider = self._detect_provider()

        print(f"[LLMClient] Active Provider: '{self.provider}' (Model: '{self.model_name}')")

    def _detect_provider(self) -> str:
        """Automatically checks if Ollama or Cloud API is available, else falls back to offline extractor."""
        # 1. Check if Ollama is running locally on port 11434
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "RAG-Monitor"})
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    return "ollama"
        except Exception:
            pass

        # 2. Check if Groq / OpenAI API key is present in environment
        if os.environ.get("GROQ_API_KEY"):
            return "groq"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai_compatible"

        # 3. Default to deterministic offline extractor
        return "offline_extractive"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Generates text using the configured LLM provider.

        Args:
            system_prompt (str): System instruction.
            user_prompt (str): User prompt containing context and question.
            temperature (float): Sampling temperature (0.0 for maximum factuality).

        Returns:
            str: The LLM's generated response text.
        """
        if self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt, temperature)
        elif self.provider in ("groq", "openai_compatible"):
            return self._call_openai_compatible(system_prompt, user_prompt, temperature)
        else:
            return self._offline_extractor(user_prompt)

    def _call_ollama(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        url = self.base_url or "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": f"<system>\n{system_prompt}\n</system>\n\n<user>\n{user_prompt}\n</user>",
            "stream": False,
            "options": {"temperature": temperature}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as e:
            print(f"[LLM Warning] Ollama call failed ({e}). Falling back to extractive mode.")
            return self._offline_extractor(user_prompt)

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        url = self.base_url
        if not url:
            url = "https://api.groq.com/openai/v1/chat/completions" if self.provider == "groq" else "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name if self.model_name != "llama3.2" else ("llama-3.1-8b-instant" if self.provider == "groq" else "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM Warning] API call failed ({e}). Falling back to extractive mode.")
            return self._offline_extractor(user_prompt)

    def _offline_extractor(self, user_prompt: str) -> str:
        """
        Deterministic, lightweight fallback that synthesizes an answer directly from the
        injected context lines matching the question key terms.
        Used for zero-setup local verification when no LLM server is active.
        """
        lines = user_prompt.splitlines()
        context_blocks = []
        is_context = False
        question = ""

        for line in lines:
            if "CONTEXT:" in line:
                is_context = True
                continue
            if "QUESTION:" in line:
                is_context = False
                question = line.replace("QUESTION:", "").strip().lower()
                continue
            if is_context and line.strip() and not line.startswith("===") and not line.startswith("[Context"):
                context_blocks.append(line.strip())

        # Full context string
        full_context = " ".join(context_blocks)
        
        # Split into individual sentences
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_context) if s.strip()]

        # Filter out pure headers/titles
        informative_sentences = [
            s for s in sentences
            if len(s.split()) >= 6 and not s.isupper() and not s.startswith("SECTION")
        ]

        q_words = set(w.lower() for w in re.findall(r'\b[a-z]{3,}\b', question))
        # Remove ultra-common stop words
        stopwords = {"what", "when", "where", "which", "who", "whom", "this", "that", "with", "from", "have", "been", "does", "will", "would", "could", "should"}
        q_keywords = q_words - stopwords

        scored_sentences = []
        for s in informative_sentences:
            s_words = set(w.lower() for w in re.findall(r'\b[a-z]{3,}\b', s))
            overlap = len(q_keywords.intersection(s_words))
            if overlap > 0:
                scored_sentences.append((overlap, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if scored_sentences:
            best_sentences = [s[1] for s in scored_sentences[:2]]
            return "Based on the official document:\n" + "\n".join(f"- {s}" for s in best_sentences)
        elif informative_sentences:
            return f"Based on the official document:\n- {informative_sentences[0]}"
        else:
            return "The provided document does not contain this information."



if __name__ == "__main__":
    client = LLMClient(provider="auto")
    sys_p = "You are a factual assistant. Answer strictly based on context."
    usr_p = "CONTEXT:\nAll students must maintain 75% attendance.\n\nQUESTION: What is the attendance requirement?"
    ans = client.generate(sys_p, usr_p)
    print("\n--- [Module 3: LLM Client Test] ---")
    print("Generated Output:")
    print(ans)

