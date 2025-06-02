import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig
import os, re
import numpy as np
import random
from smac.env import StarCraft2Env

torch.utils.checkpoint.use_reentrant = False

BASE_MODEL_ID = "google/gemma-3-1b-it"
DATASET_PATH = "smac_finetuning_data.jsonl"
OUTPUT_DIR = "./fine_tuned_smac_adapter"

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

## to be solved
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

## to be solved
value_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

## to be solved
reward_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# https://huggingface.co/docs/datasets/en/loading#json
# https://huggingface.co/docs/datasets/en/process#map
raw_dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

def ppo_dataset_formatting_function(examples):
    messages = [{"role": "user", "content": examples['prompt']}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=True)
    return {"query": text}

ppo_dataset = raw_dataset.map(
    ppo_dataset_formatting_function,
    remove_columns=raw_dataset.column_names,
    num_proc=os.cpu_count(),
)

def tokenize_queries(examples):
    return tokenizer(examples["query"], truncation=True, max_length=512, padding="max_length")

ppo_dataset = ppo_dataset.map(
    tokenize_queries,
    batched=True,
    remove_columns=["query"],
)

# https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.Dataset.set_format
ppo_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

def reward_fn(samples):
    rewards = []
    map_name = "3m"
    for sample_text in samples:
        current_env = StarCraft2Env(map_name=map_name, debug=False)
        env_info = current_env.get_env_info()
        n_agents = env_info["n_agents"]
        n_total_actions = env_info["n_actions"]
        action_list_match = re.search(r"\[(\d+(?:,\s*\d+)*)\]", sample_text)
        parsed_actions = []
        if action_list_match:
            try:
                parsed_actions = [int(x.strip()) for x in action_list_match.group(1).split(',')]
            except ValueError:
                pass

        if len(parsed_actions) != n_agents:
            valid_actions_mask = current_env.get_avail_actions()
            validated_actions = []
            for i in range(n_agents):
                avail_agent_actions = np.nonzero(valid_actions_mask[i])[0]
                if avail_agent_actions.size > 0:
                    validated_actions.append(random.choice(avail_agent_actions))
                else:
                    validated_actions.append(0)
        else:
            validated_actions = []
            valid_actions_mask = current_env.get_avail_actions()
            for i, action_id in enumerate(parsed_actions):
                avail_agent_actions = np.nonzero(valid_actions_mask[i])[0]
                if avail_agent_actions.size > 0 and action_id in avail_agent_actions:
                    validated_actions.append(action_id)
                else:
                    validated_actions.append(random.choice(avail_agent_actions) if avail_agent_actions.size > 0 else 0)

        current_env.reset()
        reward, terminated, info = current_env.step(validated_actions)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        current_env.close()

    return torch.stack(rewards)

# https://huggingface.co/docs/trl/main/en/ppo_trainer#trl.PPOConfig
# https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_config.py#L22
ppo_config = PPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=float(1),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    kl_coef=0.1,
    gamma=0.99,
    lam=0.95,
)

# https://huggingface.co/docs/trl/main/en/ppo_trainer#trl.PPOTrainer
# https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L98
trainer = PPOTrainer(
    ppo_config,
    model=model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=ppo_dataset,
    reward_model=reward_model,
    value_model=value_model,
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