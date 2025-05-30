import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import os
import torch.utils.checkpoint

torch.utils.checkpoint.use_reentrant = False

BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = "smac_finetuning_data.jsonl"
OUTPUT_DIR = "./fine_tuned_smac_llama3_adapter"

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
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

def preprocess_function(examples):
    formatted_texts = []
    for i in range(len(examples['prompt'])):
        messages = [
            {"role": "user", "content": examples['prompt'][i]},
            {"role": "assistant", "content": examples['completion'][i]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=True)
        formatted_texts.append(text)

    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=512,
        padding=False,
    )
    return tokenized

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count(),
)

# https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    report_to="tensorboard",
    disable_tqdm=False,
)

# https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    args=training_args,
    peft_config=peft_config,
    data_collator=data_collator,
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