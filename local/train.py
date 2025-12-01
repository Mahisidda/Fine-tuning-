import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ---------------- CONFIG ----------------
model_id = "meta-llama/Llama-3.2-1B-Instruct"   # HF model (needs access + HF login)

# Get workspace root (parent of 'local' folder where this script lives)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(workspace_root, "Processed Datasets")  # folder that contains your jsonl files

# Explicitly list your files
train_files = [
    os.path.join(data_dir, "alpaca_llama_train.jsonl"),
    os.path.join(data_dir, "cnn_dailymail_llama_train.jsonl"),
    os.path.join(data_dir, "eduqg_llm_llama_generic.jsonl"),  # Fixed typo: eduqg (not eduqq)
    os.path.join(data_dir, "oasst1_llama_train.jsonl"),
    os.path.join(data_dir, "squad_v2_llama_train.jsonl"),
]

val_files = [
    os.path.join(data_dir, "cnn_dailymail_llama_validation.jsonl"),
    os.path.join(data_dir, "oasst1_llama_val.jsonl"),
    os.path.join(data_dir, "squad_v2_llama_val.jsonl"),
]

out_dir = "llama32_1b_study_lora_final-1"   # where LoRA weights will be saved

max_length        = 512             # matches your app n_ctx
num_epochs        = 1               # keep 1 for now
max_train_samples = 20_000          # IMPORTANT: cap so training is manageable
max_eval_samples  = 2_000
max_answer_words  = 90              # clamp labels so the style stays short
# ----------------------------------------


SYSTEM_GENERIC = (
    "You are an offline study assistant running on a smartphone. "
    "Respond in 3-5 crisp bullet points or a single <=80-word paragraph. "
    "Skip greetings and filler."
)

SYSTEM_EXCERPT = (
    "You are an offline study assistant. Use only the provided passage to answer. "
    "Stay within 3-5 short bullet points or a <=80-word paragraph."
)

STYLE_SUFFIX = "\n\nRespond in 3-5 short bullet points or a <=80-word paragraph."


def clamp_answer(answer: str) -> str:
    """
    Keep target texts short so the model learns the desired style.
    Preserves line boundaries where possible while trimming extra words.
    """
    text = (answer or "").strip()
    if not text:
        return ""

    words_allowed = max_answer_words
    lines_out = []
    for raw_line in text.splitlines():
        tokens = raw_line.split()
        if not tokens:
            if words_allowed > 0:
                lines_out.append("")
            continue

        if words_allowed <= 0:
            break

        if len(tokens) > words_allowed:
            tokens = tokens[:words_allowed]
        lines_out.append(" ".join(tokens))
        words_allowed -= len(tokens)

        if words_allowed <= 0:
            break

    trimmed = "\n".join(lines_out).strip()
    return trimmed

# ------------- TOKENIZER -------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ------------- SCHEMA → CHAT MESSAGES -------------
def build_messages(example):
    """
    Convert one raw example (from any of your datasets) into
    a list of {role, content} messages suitable for apply_chat_template.
    """

    # Case 0: already in chat format
    if "messages" in example:
        return example["messages"]

    # Case 1: Alpaca-style (instruction / input / output)
    if "instruction" in example and "output" in example:
        user_content = example["instruction"] or ""
        if "input" in example and example["input"]:
            user_content += "\n\nExtra context:\n" + example["input"]
        user_content = user_content.strip() + STYLE_SUFFIX
        assistant_text = clamp_answer(example["output"])
        return [
            {"role": "system", "content": SYSTEM_GENERIC},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_text},
        ]

    # Case 2: CNN / DailyMail summarization (article / highlights)
    if "article" in example and "highlights" in example:
        user_content = (
            "Summarize the following passage in 3-5 short bullet points.\n\n"
            + (example["article"] or "")
        )
        user_content = user_content.strip() + STYLE_SUFFIX
        assistant_text = clamp_answer(example["highlights"])
        return [
            {"role": "system", "content": SYSTEM_EXCERPT},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_text},
        ]

    # Case 3: SQuAD-style QA (context / question / answers)
    if "context" in example and "question" in example and "answers" in example:
        answers = example["answers"]
        # answers may be a dict with "text": [...]
        if isinstance(answers, dict):
            texts = answers.get("text") or []
            answer_text = texts[0] if len(texts) > 0 else ""
        else:
            # Fallback if it's already a string
            answer_text = str(answers)

        user_content = (
            "Read the passage and answer the question briefly.\n\n"
            f"Passage:\n{example['context']}\n\n"
            f"Question: {example['question']}"
        ) + STYLE_SUFFIX
        assistant_text = clamp_answer(answer_text)
        return [
            {"role": "system", "content": SYSTEM_EXCERPT},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_text},
        ]

    # Case 4: OASST-like conversation: conversations / prompt / response, etc.
    if "conversations" in example:
        msgs = []
        for turn in example["conversations"]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            # Map roles to HF chat roles
            if role == "assistant":
                r = "assistant"
            elif role == "system":
                r = "system"
            else:
                r = "user"
            if r == "assistant":
                content = clamp_answer(content)
            msgs.append({"role": r, "content": content})
        if not any(m["role"] == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": SYSTEM_GENERIC})
        return msgs

    if "prompt" in example and "response" in example:
        # Generic prompt/response style
        user_content = (example["prompt"] or "").strip() + STYLE_SUFFIX
        assistant_text = clamp_answer(example["response"])
        return [
            {"role": "system", "content": SYSTEM_GENERIC},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_text},
        ]

    # Case 5: Preprocessed format with mode, question, excerpt, answer
    if "question" in example and "answer" in example:
        mode = example.get("mode", "")
        question = example.get("question", "")
        excerpt = example.get("excerpt", "")
        answer = example.get("answer", "")
        
        # Choose system prompt based on mode
        if mode == "excerpt":
            system_content = SYSTEM_EXCERPT
        else:
            system_content = SYSTEM_GENERIC
        
        # Build user content
        if excerpt:
            user_content = f"Excerpt:\n{excerpt}\n\nQuestion: {question}"
        else:
            user_content = question
        user_content = user_content.strip() + STYLE_SUFFIX
        assistant_text = clamp_answer(answer)
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_text},
        ]

    # If we hit something unexpected, raise clearly so you can inspect it.
    raise ValueError(f"Don't know how to convert example with keys: {list(example.keys())}")


def to_chat_text(example):
    messages = build_messages(example)
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # train on full dialog (incl. assistant)
    )
    return {"text": chat_text}


# ------------- LOAD DATA -------------
print("Validating dataset files...")
for f in train_files + val_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing dataset file: {f}")

print(f"Loading {len(train_files)} train file(s) and {len(val_files)} val file(s)...")

train_raw = load_dataset("json", data_files=train_files, split="train")
val_raw   = load_dataset("json", data_files=val_files,   split="train")

print("Train examples:", len(train_raw))
print("Val examples  :", len(val_raw))

# Optional subsampling for sanity runs
if max_train_samples is not None and len(train_raw) > max_train_samples:
    train_raw = train_raw.shuffle(seed=42).select(range(max_train_samples))
if max_eval_samples is not None and len(val_raw) > max_eval_samples:
    val_raw = val_raw.shuffle(seed=42).select(range(max_eval_samples))

print("Mapping to chat text...")
train_text = train_raw.map(to_chat_text, remove_columns=train_raw.column_names)
val_text   = val_raw.map(to_chat_text,   remove_columns=val_raw.column_names)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

print("Tokenizing...")
train_tok = train_text.map(tokenize_fn, batched=True, remove_columns=["text"])
val_tok   = val_text.map(tokenize_fn,   batched=True, remove_columns=["text"])


# ------------- MODEL + LoRA -------------
print("Loading base model...")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Very important for decoder-only models + LoRA
model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ------------- TRAINING -------------
training_args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=num_epochs,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    logging_steps=50,
    save_steps=500,          # saves periodically; LoRA is small
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,        # even if you don't use eval yet
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Saving LoRA adapter to", out_dir)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

print("Done.")
