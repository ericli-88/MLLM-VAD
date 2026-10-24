"""LLM backend adapters. Heavy model imports are intentionally lazy."""

from __future__ import annotations

from typing import Protocol


class ChatBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class Llama2ChatBackend:
    """Adapter around the original repository's Llama 2 chat generator."""

    def __init__(
        self,
        *,
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int = 4096,
        max_gen_len: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        from libs.llama.llama import Llama

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=1,
        )
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        dialogs = [[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]]
        result = self.generator.chat_completion(
            dialogs,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )[0]
        return str(result["generation"]["content"])


class TransformersChatBackend:
    """Single-process Hugging Face chat-model backend for the M3VAD runner."""

    def __init__(
        self,
        *,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-m3vad.txt to use this backend") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="auto",
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.inference_mode():
            output = self.model.generate(input_ids, **generate_kwargs)
        return self.tokenizer.decode(output[0, input_ids.shape[-1] :], skip_special_tokens=True)


class HeuristicMockBackend:
    """Small deterministic backend for integration tests; never for reported metrics."""

    DANGER_WORDS = (
        "attack",
        "assault",
        "blood",
        "explosion",
        "fight",
        "fire",
        "gun",
        "knife",
        "robbery",
        "shoot",
        "violence",
    )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import json

        lowered = user_prompt.lower()
        danger_count = sum(word in lowered for word in self.DANGER_WORDS)
        score = min(1.0, round(min(danger_count, 10) / 10.0, 1))
        return json.dumps(
            {
                "anomaly_score": score,
                "visual_evidence": "Mock evidence derived from keyword presence.",
                "temporal_evidence": "Mock temporal evidence.",
                "memory_comparison": "Mock backend does not perform semantic reasoning.",
                "memory_used": "retrieved_normal_examples\": []" not in user_prompt,
                "evidence_conflict": False,
            }
        )
