---
# README.md

```markdown
# QLoRA + PEFT Fine-Tuning — LLaMA for Hate Speech Detection

This project demonstrates a minimal setup for fine-tuning a LLaMA-family language model using **QLoRA** (4-bit quantization) and **PEFT/LoRA**, applied to a simple **hate speech detection** task. The goal is to provide a lightweight, reproducible example that can be copy-pasted and run on a GPU machine.

---

## ✨ Features
- 4-bit QLoRA training using **bitsandbytes**
- **LoRA adapters** via PEFT
- Works with any LLaMA-family model you have permission to access
- JSONL dataset support (`text`, `label`)
- Minimal training code using HuggingFace `Trainer`
- Fully local, no external API required


Dataset format (one JSON object per line):
```json
{"text": "sample sentence...", "label": 0}
````

Where `label` is:

* `1` → hate speech
* `0` → non-hate

---

## 🛠 Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Make sure your machine has:

* CUDA-enabled GPU
* PyTorch with CUDA
* Access to the chosen LLaMA model on HuggingFace

---

## 🚀 Training

Edit the base model in `train.py` if needed:

```python
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
```

Then launch training:

```bash
accelerate launch train.py
```

Output LoRA adapter weights will be saved in:

```
qlora-lora-output/
```

---

## 📝 Model Behavior

The fine-tuning approach uses **generation-based classification**:
The model is trained to generate either `0` or `1` as the label after a prompt.

This keeps the code simple while still demonstrating how QLoRA + LoRA adapters work on a causal LLM.

