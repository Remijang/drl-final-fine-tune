import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import os, sys, re
import numpy as np
import random

torch.utils.checkpoint.use_reentrant = False

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "smac_finetuning_data.jsonl"
OUTPUT_DIR = "./fine_tuned_smac_llama3_adapter_dpo"

# https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if not torch.cuda.is_available():
    raise SystemError("CUDA is not available.")
print(f"Using CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

# https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={'':torch.cuda.current_device()},
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# https://huggingface.co/docs/datasets/en/loading#json
# https://huggingface.co/docs/datasets/en/process#map
raw_dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

def dpo_dataset_formatting_function(examples):
    prompt_messages = [{"role": "user", "content": examples['prompt']}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_special_tokens=True)
    return {
        "prompt": prompt_text,
        "chosen": examples['chosen'],
        "rejected": examples['rejected']
    }

dpo_dataset = raw_dataset.map(
    dpo_dataset_formatting_function,
    remove_columns=raw_dataset.column_names,
    num_proc=os.cpu_count(),
)

# https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig
# https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py#L33
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=float(1),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    beta=0.1,
    loss_type="sigmoid",
)

# https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOTrainer
# https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L161
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    processing_class=tokenizer,
    train_dataset=dpo_dataset,
    peft_config=peft_config,
)

trainer.train()

output_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.save_model(output_adapter_path)
tokenizer.save_pretrained(output_adapter_path)

del model

model_for_merge = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map={'':torch.cuda.current_device()},
    trust_remote_code=True,
)
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    output_adapter_path,
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.float16
)
merged_model = merged_model.merge_and_unload()

merged_output_dir = os.path.join(OUTPUT_DIR, "merged_model")
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)