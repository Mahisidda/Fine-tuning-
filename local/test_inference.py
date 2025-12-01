# test_inference.py

import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR   = "llama32_1b_study_lora_final"   # <-- default adapter dir
DEFAULT_MAX_NEW_TOKENS = 120
DEFAULT_TEMPERATURE = 0.6
REPETITION_PENALTY = 1.05

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = None


def load_model(adapter_dir: str):
    print(f"Loading LoRA fine-tuned model from {adapter_dir}...")
    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        loaded = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir,
            quantization_config=quant_cfg,
            device_map="auto",
        )
    else:
        loaded = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    loaded.eval()
    setattr(loaded, "_adapter_dir", adapter_dir)
    return loaded

SYSTEM_PROMPT = (
    "You are an offline study assistant running on a smartphone. "
    "You answer concisely in exam-ready form: short bullet points or a tight paragraph. "
    "Avoid long intros or chit-chat. Go straight to the explanation."
)

def build_generic_messages(question: str):
    user_content = (
        "Answer the following study question in 3–5 short bullet points "
        "or a short paragraph. Be concise and exam-focused.\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def build_excerpt_messages(excerpt: str, question: str):
    user_content = (
        "You are given a textbook excerpt.\n\n"
        "[EXCERPT]\n"
        f"{excerpt}\n"
        "[/EXCERPT]\n\n"
        "Using only this excerpt, answer the question in 3–5 short bullet points. "
        "Be direct and exam‑oriented.\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def chat(messages, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=DEFAULT_TEMPERATURE):
    global model
    if model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,
        )

    gen_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=str,
        default=ADAPTER_DIR,
        help="Path to the LoRA adapter folder.",
    )
    parser.add_argument(
        "--mode",
        choices=["generic", "excerpt"],
        default="generic",
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default="Explain Newton's first law in very simple terms.",
    )
    parser.add_argument(
        "--excerpt",
        type=str,
        default="",
        help="Only used in excerpt mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (lower = more deterministic).",
    )
    args = parser.parse_args()

    global model
    if model is None or getattr(model, "_adapter_dir", None) != args.adapter:
        model = load_model(args.adapter)

    if args.mode == "generic":
        msgs = build_generic_messages(args.question)
    else:
        msgs = build_excerpt_messages(args.excerpt, args.question)

    answer = chat(
        msgs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n=== MODEL ANSWER ===\n")
    print(answer)
    print("\n====================\n")

if __name__ == "__main__":
    main()
