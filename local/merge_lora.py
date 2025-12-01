import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base Llama weights."
    )
    parser.add_argument(
        "--base-model-id",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF repo or local path for the base model.",
    )
    parser.add_argument(
        "--lora-dir",
        default="llama32_1b_study_lora_final-1",
        help="Directory containing the trained LoRA adapter.",
    )
    parser.add_argument(
        "--out-dir",
        default="llama32_1b_study_merged",
        help="Where to save the merged full model.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype to load the base model with.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device to load the base model on (use 'auto' to prefer CUDA if available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    target_device = args.device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if target_device == "cuda" else "cpu"

    print(f"Loading base model ({args.base_model_id}) on {target_device} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from {args.lora_dir} ...")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)

    print("Merging LoRA into base weights...")
    merged_model = model.merge_and_unload()

    if target_device == "cuda":
        print("Moving merged model back to CPU for saving...")
        merged_model = merged_model.to("cpu")

    print(f"Saving merged model to {args.out_dir} ...")
    merged_model.save_pretrained(args.out_dir)

    print("Saving tokenizer (for convenience)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)
    tokenizer.save_pretrained(args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
