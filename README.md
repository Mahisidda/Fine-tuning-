# Llama 3.2 1B Fine-tuning for Study Assistant

Fine-tuning scripts for creating a study assistant using Llama 3.2 1B with LoRA.

## Overview

This repository contains scripts to fine-tune Llama 3.2 1B Instruct model for a study assistant application that provides concise, exam-ready answers.

## Files

- `local/train.py` - Main training script with LoRA fine-tuning
- `local/test_inference.py` - Script to test the fine-tuned model
- `local/merge_lora.py` - Script to merge LoRA adapter into base model

## Setup

1. Install dependencies:
```bash
pip install torch transformers datasets peft bitsandbytes accelerate
```

2. Configure paths in `local/train.py`:
   - Set `data_dir` to your processed datasets folder
   - Adjust `train_files` and `val_files` lists

3. Run training:
```bash
python local/train.py
```

4. Test inference:
```bash
python local/test_inference.py --question "Your question here"
```

5. Merge LoRA adapter (optional):
```bash
python local/merge_lora.py --lora-dir <your_lora_dir> --out-dir <output_dir>
```

## Model Details

- Base Model: `meta-llama/Llama-3.2-1B-Instruct`
- Fine-tuning: LoRA (r=8, alpha=16)
- Max Length: 512 tokens
- Training Style: Short, concise answers (3-5 bullet points or ≤80 words)

## License

Check base model license from Meta.

