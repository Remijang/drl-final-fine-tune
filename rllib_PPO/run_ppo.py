import argparse
import ray
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from env import RLlibStarCraft2Env
from model import LLMMaskedActionsModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RLlib PPO with LLM for SMAC.")
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--map-name", type=str, default="3m")
    parser.add_argument("--output-dir", type=str, default="./rllib_smac_llm_results")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--local-mode", action="store_true")
    args = parser.parse_args()
    if args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()

    register_env("smac_rllib_env", lambda config: RLlibStarCraft2Env(**config))

    ModelCatalog.register_custom_model("LLMMaskedActionsModel", LLMMaskedActionsModel)

    # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ray.rllib.algorithms.ppo.ppo.PPOConfig
    # https://docs.ray.io/en/latest/_modules/ray/rllib/algorithms/ppo/ppo.html#PPOConfig
    config = (
        PPOConfig()
        .environment(
            env="smac_rllib_env",
            env_config={"map_name": args.map_name},
            disable_env_checking=True,
        )
        .framework("torch")
        .training(
            lr=1e-5,
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            num_epochs=30,
            train_batch_size=4000,
            model={
                "custom_model": "LLMMaskedActionsModel",
                "custom_model_config": {},
            },
        )
        .env_runners(num_env_runners=args.num_workers) 
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
        .debugging(
            log_level="INFO",
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .experimental(_disable_preprocessor_api=True)
    )

    if args.no_tune:
        algo = config.build()
        for i in range(args.num_iters):
            print(f"Training iteration {i + 1}/{args.num_iters}...")
            result = algo.train()
            print(f"Iteration {i + 1} reward: {result['episode_reward_mean']:.2f}")
            if (i + 1) % 5 == 0 or i == args.num_iters - 1:
                checkpoint_dir = algo.save(checkpoint_dir=os.path.join(args.output_dir, "checkpoints")).checkpoint.path
                print(f"Checkpoint saved to {checkpoint_dir}")
        algo.stop()
    else:
        results = run_experiments(
            {
                "smac_llm_ppo": {
                    "run": "PPO",
                    "env": "smac_rllib_env",
                    "config": config.to_dict(),
                    "stop": {"training_iteration": args.num_iters},
                    "checkpoint_config": {
                        "checkpoint_at_end": True,
                    },
                    "local_dir": args.output_dir,
                }
            },
        )
        best_trial = results.get_best_trial("smac_llm_ppo", "episode_reward_mean", "max")
        print(f"Best trial finished with reward: {best_trial.last_result['episode_reward_mean']:.2f}")
        print(f"Best trial checkpoint path: {best_trial.checkpoint.path}")
        checkpoint_dir = best_trial.checkpoint.path
    
    final_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        LLMMaskedActionsModel.BASE_MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True,
    )
    
    merged_model = PeftModel.from_pretrained(base_model_for_merge, checkpoint_dir)
    merged_model = merged_model.merge_and_unload()

    merged_output_dir = os.path.join(args.output_dir, "merged_model")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    final_tokenizer.save_pretrained(merged_output_dir)

    ray.shutdown()
