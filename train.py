
import os
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

# -------- USER CONFIG (edit these) ----------
MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-2-7b-hf")  # replace if needed
TRAIN_FILE = "data/train.jsonl"
VALID_FILE = "data/valid.jsonl"
OUTPUT_DIR = "qlora-lora-output"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
# --------------------------------------------

def make_prompt(text: str) -> str:
    # Simple prompt: adjust to taste
    return f"Classify the following text for hate speech (1 = hate, 0 = not):\n\n{text}\n\nLabel:"

def tokenize_function(example, tokenizer):
    prompt = make_prompt(example["text"])
    # target is label token + newline
    label = str(int(example["label"]))
    full = prompt + " " + label
    enc = tokenizer(full, truncation=True, max_length=MAX_LENGTH)
    # set labels equal to input_ids (causal LM style)
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main():
    # load dataset
    data_files = {"train": TRAIN_FILE, "validation": VALID_FILE}
    ds = load_dataset("json", data_files=data_files)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 4-bit bitsandbytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # load quantized model
    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,  # for some LLaMA-family repos
    )

    # resize token embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    # prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # common for LLaMA-style
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    print("PEFT/LoRA parameters:", model.print_trainable_parameters())

    # tokenize dataset
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer), batched=False)

    # data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Training complete — model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
