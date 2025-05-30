import numpy as np
import json
import os
import random
from smac.env import StarCraft2Env
from translate import get_state_NL
from test_llm import get_action_description

class DataGeneratorAgent:
    def _construct_prompt(self, global_state_nl, avail_actions_list, n_agents, n_total_actions):
        prompt = (
            f"You are an AI agent controlling {n_agents} units in a StarCraft II scenario.\n"
            "Your objective is to defeat all enemy units by issuing commands to your units.\n"
            "--- CURRENT GLOBAL STATE ---\n"
            f"{global_state_nl}\n"
            "--- ORDERS FOR YOUR UNITS ---\n"
            f"For each of your {n_agents} units, select one action from its list of available actions.\n"
            "In the last line respond with a formatted list of integer action IDs, one for each unit.\n"
            "For example, to order agent 0, 1, 2 to perform action ID 0, 1, 5 respectively, output: [0, 1, 5]\n"
            "Make sure the action ID given to each agent is available for each agent.\n"
            "Just answer with result list\n"
        )
        all_agent_available_action_indices = []
        for i in range(n_agents):
            avail_agent_actions_mask = avail_actions_list[i]
            available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
            all_agent_available_action_indices.append(available_action_indices)
            prompt += f"\n-- Unit {i} --\nAvailable Actions (ID: Description):\n"
            if len(available_action_indices) == 0:
                prompt += " - No actions available (unit may be incapacitated or have no valid moves).\n"
            else:
                for action_id in available_action_indices:
                    action_desc = get_action_description(action_id, n_total_actions)
                    prompt += f" - ID {action_id}: {action_desc}\n"
        prompt += f"\nBased on the state and available actions, provide the list of {n_agents} action IDs for your units."
        return prompt, all_agent_available_action_indices

    def act(self, global_state_nl, avail_actions_list, n_agents, env_info):
        n_total_actions = env_info["n_actions"]
        prompt_text, all_agent_available_action_indices = self._construct_prompt(
            global_state_nl, avail_actions_list, n_agents, n_total_actions
        )
        chosen_actions = []
        rejected_actions = []
        for i in range(n_agents):
            available_for_agent = all_agent_available_action_indices[i]
            if available_for_agent.size > 0:
                chosen_action = random.choice(available_for_agent)
                chosen_actions.append(chosen_action)
                if len(available_for_agent) > 1:
                    other_available_actions = [
                        a for a in available_for_agent if a != chosen_action
                    ]
                    if other_available_actions:
                        rejected_action = random.choice(other_available_actions)
                    else:
                        rejected_action = chosen_action
                else:
                    rejected_action = chosen_action
                rejected_actions.append(rejected_action)
            else:
                chosen_actions.append(0)
                rejected_actions.append(0)

        return prompt_text, chosen_actions, rejected_actions


def generate_finetuning_data(agent, map_name="3m", episodes=50, max_steps_per_episode=50, output_file="smac_finetuning_data_dpo.jsonl"):
    env = StarCraft2Env(map_name=map_name, debug=False)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    data_samples = []

    for i in range(episodes):
        env.reset()
        terminated = False
        for step in range(max_steps_per_episode):
            global_state = env.get_state()
            avail_actions_list = env.get_avail_actions()
            global_state_nl = get_state_NL(env, global_state)
            prompt_text, chosen_actions, rejected_actions = agent.act(
                global_state_nl, avail_actions_list, n_agents, env_info
            )
            completion_text_chosen = f"[{', '.join(map(str, chosen_actions))}]"
            completion_text_rejected = f"[{', '.join(map(str, rejected_actions))}]"
            data_samples.append({
                "prompt": prompt_text,
                "chosen": completion_text_chosen,
                "rejected": completion_text_rejected,
            })
            reward, terminated, info = env.step(chosen_actions)

            if terminated:
                print(f"Episode {i + 1} terminated after {step + 1} steps. Total Reward: {reward:.2f}")
                break
        if not terminated:
            print(f"Episode {i + 1} reached max steps ({max_steps_per_episode}).")

    with open(output_file, 'w') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')
    env.close()

if __name__ == "__main__":
    data_generator = DataGeneratorAgent()
    generate_finetuning_data(
        agent=data_generator,
        map_name="3m",
        episodes=50,
        max_steps_per_episode=50,
        output_file="smac_finetuning_data.jsonl"
    )
