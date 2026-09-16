"""
pipeline/rag/llm_hf.py — HuggingFace LLM backend (model-agnostic)
Supports Qwen, Llama, Mistral — any chat-template model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_CACHE: dict = {}


def make_hf_fn(model_name: str, max_new_tokens: int = 512):
    """
    Load model and return callable: generate(system, user) -> str
    Models are cached — second call with same name reuses loaded model.
    """
    global _MODEL_CACHE

    if model_name not in _MODEL_CACHE:
        print(f"[HF] Loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability(0)
            except Exception:
                major = 7
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
        print(
            f"[HF] {model_name} loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}"
        )

    tokenizer, model = _MODEL_CACHE[model_name]

    def generate(system: str, user: str, temperature: float = 0.0) -> str:
        """Render `system`/`user` via the model's chat template and generate a completion (greedy if temperature==0.0, sampled otherwise); returns the decoded response text, no side effects on the cached model."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        do_sample = temperature > 0.0
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True, temperature=temperature, top_p=0.9
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        return tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()

    return generate
