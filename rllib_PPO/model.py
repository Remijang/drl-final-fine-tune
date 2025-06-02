import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium.spaces import Dict, Box, Text
import numpy as np
from typing import Tuple

from utils import get_action_id, system_prompt

torch, nn = try_import_torch()

class LLMMaskedActionsModel(TorchModelV2, nn.Module):
    BASE_MODEL_ID = "google/gemma-3-1b-it" 
    MAX_SEQ_LENGTH = 512

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
            and "nl_obs" in orig_space.spaces
        ), "Observation space must be a Dict with 'obs', 'action_mask', and 'nl_obs' keys."

        self.num_outputs = num_outputs
        self.model_config = model_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        print(f"Loading base LLM for policy: {self.BASE_MODEL_ID}...")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)

        # https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        )
        self.llm_model.add_adapter(peft_config)

        self.feature_extractor = TorchFC(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_feature_extractor",
        )
        
        self._last_features = None

    def forward(self, input_dict: Dict, state: list, seq_lens: any) -> Tuple[any, list]:
        obs = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]
        nl_obs_strings = [item[0] for item in input_dict["obs"]["nl_obs"]]

        self._last_features, _ = self.feature_extractor(
            {"obs": obs.float()}, state, seq_lens
        )
        
        prompts = []
        for nl_obs_string_for_agent in nl_obs_strings:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_obs_string_for_agent},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=True)
            prompts.append(prompt)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.MAX_SEQ_LENGTH).to(self.llm_model.device)

        generated_outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        generated_sequences = generated_outputs.sequences
        generated_texts = self.tokenizer.batch_decode(generated_sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        logits = torch.full((len(generated_texts), self.num_outputs), float('-inf'), device=obs.device)
        for i, text in enumerate(generated_texts):
            action_id = get_action_id(text)
            mask = action_mask[i]

            if action_id == -1 or action_id >= self.num_outputs or mask[action_id].item() == 0:
                valid_actions = (mask == 1).nonzero(as_tuple=True)[0]
                if len(valid_actions) > 0:
                    action_id = valid_actions[0].item()
                else:
                    action_id = 0

            logits[i, action_id] = 0.0

        action_mask = action_mask.to(logits.device)
        masked_logits = torch.where(
            action_mask.bool(),
            logits,
            torch.tensor(float('-inf'), device=logits.device)
        )
        
        all_masked_out_rows = torch.all(torch.isinf(masked_logits) & (masked_logits < 0), dim=1)
        
        if torch.any(all_masked_out_rows):
            masked_logits[all_masked_out_rows, 0] = 0.0

        return masked_logits, state

    def value_function(self) -> any:
        assert self._last_features is not None, "value_function() must be called after forward()"
        return torch.reshape(self.feature_extractor.value_function(), [-1])
