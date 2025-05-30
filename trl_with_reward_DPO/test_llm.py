from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from smac.env import StarCraft2Env
from translate import get_state_NL
from prompt import system_prompt

import os
import openai
import random

def get_action_description(action_id, n_total_actions):
    """Provides a basic textual description for an action ID."""
    if action_id == 0: return "no_op"
    if action_id == 1: return "stop"
    if n_total_actions > 2 and action_id == 2: return "move_north"
    if n_total_actions > 3 and action_id == 3: return "move_south"
    if n_total_actions > 4 and action_id == 4: return "move_east"
    if n_total_actions > 5 and action_id == 5: return "move_west"
    if n_total_actions > 6 and (6 <= action_id < n_total_actions):
        return f"attack_{action_id - 5}"
    return f"action_id_{action_id}"
def get_action_id(action_desc):
    action_map = {
        "no_op": 0,
        "stop": 1,
        "move_north": 2,
        "move_south": 3,
        "move_east": 4,
        "move_west": 5,
    }
    try:
        if action_desc in action_map:
            return action_map[action_desc]
        elif action_desc[:7] == "attack_":
            return int(action_desc[7:]) + 5
        else:
            return -1
    except:
        return -1


class LLMAgent:
    def __init__(self, model_name="llama3.2", verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        openai.api_base = 'http://127.0.0.1:11434/v1' # test with ollama
        openai.api_key = 'ollama'


    def _construct_prompt(self, global_state_nl, avail_actions_list, n_agents, n_total_actions):
        prompt = (
            "--- CURRENT GLOBAL STATE ---\n"
            f"{global_state_nl}\n"
            "--- AVAILABLE ACTIONS ---\n"
        )
        all_agent_available_action_indices = []
        for i in range(n_agents):
            avail_agent_actions_mask = avail_actions_list[i]
            available_action_indices = np.nonzero(avail_agent_actions_mask)[0]
            all_agent_available_action_indices.append(available_action_indices)

            prompt += f"\n-- Unit {i+1} --\nAvailable Actions:\n"
            if len(available_action_indices) == 0:
                prompt += " - No actions available (unit may be incapacitated or have no valid moves).\n"
            else:
                for action_id in available_action_indices:
                    action_desc = get_action_description(action_id, n_total_actions)
                    prompt += f" - {action_desc}\n"
        return prompt, all_agent_available_action_indices
    def _get_actions_from_llm_api_macro(self, prompt_text, n_agents):
        if self.verbose:
            print("\n===== LLM PROMPT (Sending to API) =====")
            print(prompt_text)
            print("=======================================")
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                      {"role": "user", "content": prompt_text}
                  ]
            )
            llm_output_str = response.choices[0].message.content.strip()
            actions = llm_output_str.split('\n')[-1].strip().split(' | ')
            if len(actions) != n_agents:
                print("Parse action error!\nLLM output:",llm_output_str)
                return []
            ret = []
            for a in actions:
                a = a.tolower()
                act = {}
                if 'move' in a:
                    act['op'] = 'move'
                    act['x'],act['y'] = int(a[5:].split(',')[0]),int(a[5:].split(',')[1])
                elif 'stop' in a:
                    act['op'] = 'stop'
                elif 'attack' in a:
                    act['op'] = 'attack'
                    act['x'] = int(a[7:])
                else:
                    print("Parse action error!\nLLM output:",llm_output_str)
                    return []
                ret.append(act)
            if self.verbose:
                print(f"LLM response:\n{llm_output_str}")
            return ret
        except:
            print("Parse action error!\nLLM output:",llm_output_str)
            return []
    def _get_actions_from_llm_api(self, prompt_text, n_agents, all_available_action_indices, n_total_actions):
        if self.verbose:
            print("\n===== LLM PROMPT (Sending to API) =====")
            print(prompt_text)
            print("=======================================")
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                      {"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt_text}
                  ]
            )
            llm_output_str = response.choices[0].message.content.strip()
            #### action parser ####
            import re
            action_section = re.search(r"<ACTION>(.*?)</ACTION>", llm_output_str, re.DOTALL)
            if not action_section:
                return []

            # Find all Ally actions
            action_lines = re.findall(r"- Ally #(\d+): (\S+)", action_section.group(1))
            # Initialize action array with "no_op"
            action_array = ["no_op"] * n_agents

            # Populate action array
            for ally, command in action_lines:
                action_array[int(ally) - 1] = command  # Convert 1-based index to 0-based

            # Convert actions to indices
            indexed_actions = [get_action_id(action) for action in action_array]
            if self.verbose:
                print("LLM output:",llm_output_str)
                print(indexed_actions)
            return indexed_actions

        except:
            print("Parse action error!\nLLM output:",llm_output_str)
            return []
    
    


    def act(self, global_state_nl, avail_actions_list, n_agents, env_info):
        n_total_actions = env_info["n_actions"]
        
        prompt, all_agent_available_action_indices = self._construct_prompt(
            global_state_nl, avail_actions_list, n_agents, n_total_actions
        )
        chosen_actions = list(self._get_actions_from_llm_api(prompt, n_agents, all_agent_available_action_indices, n_total_actions))

        # Basic validation for action list length
        if len(chosen_actions) != n_agents: # revert to random
            chosen_actions = [random.choice(all_agent_available_action_indices[i]) for i in range(n_agents)]
            if self.verbose:
                print(f"LLMAgent Warning: Action list length mismatch")
        for i in range(n_agents): # illegal moves, revert to random
            if chosen_actions[i] not in all_agent_available_action_indices[i]:
                
                chosen_actions[i] = random.choice(all_agent_available_action_indices[i])

        if self.verbose:
            action_descs_final = [get_action_description(ac, n_total_actions) for ac in chosen_actions]
            print(f"LLMAgent final selected actions: IDs={chosen_actions}, Descriptions={action_descs_final}")
        
        return chosen_actions

def run_smac_with_agent(agent, map_name="3m", episodes=1, max_steps_per_episode=200, render=False, verbose_env=False, save_replay=False):
    """Runs the given agent on the specified SMAC map."""
    try:
        env = StarCraft2Env(map_name=map_name, 
                            replay_dir="replays" if save_replay else None, # SMAC default replay dir is SC2Replays
                            debug=verbose_env)
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        n_total_actions = env_info["n_actions"]

        print(f"Starting SMAC with {agent.__class__.__name__} on map: {map_name}")
        print(f"Number of agents: {n_agents}, Action space size: {n_total_actions}")
        print(f"Max steps per episode (set): {max_steps_per_episode}, Env episode limit: {env_info.get('episode_limit', 'N/A')}")
        print("-" * 30)

        total_rewards = []
        for e_idx in range(episodes):
            env.reset()
            terminated = False
            episode_reward = 0
            
            if verbose_env: print(f"\n--- Episode {e_idx + 1} ---")

            for step in range(max_steps_per_episode):
                if render: env.render()

                # Get information for the agent
                # obs_list = env.get_obs() # Raw observations, if needed by agent
                global_state = env.get_state()
                avail_actions_list = env.get_avail_actions()
                
                global_state_nl = get_state_NL(env, global_state)

                # Agent chooses actions
                actions = agent.act(global_state_nl, avail_actions_list, n_agents, env_info)
                
                reward, terminated, info = env.step(actions)
                episode_reward += reward

                if verbose_env:
                    print(f"  Step {step + 1}: Actions={actions}, Reward={reward:.2f}, Terminated={terminated}")

                if terminated:
                    if save_replay:
                        env.save_replay()
                        if verbose_env: print(f"Replay saved for episode {e_idx + 1}.")
                    break
            
            win_status = "UNKNOWN"
            if 'battle_won' in info: # info might be empty if episode ended due to step limit
                win_status = "WON" if info['battle_won'] else "LOST/DRAW"
            
            print(f"Episode {e_idx + 1} finished. Steps: {step + 1}. Reward: {episode_reward:.2f}. Status: {win_status}")
            total_rewards.append(episode_reward)

        print("-" * 30)
        print(f"{agent.__class__.__name__} test finished.")
        if total_rewards:
            print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f} (Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f})")
        else:
            print("No episodes were run or no rewards collected.")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure PySC2, SMAC, and dependencies (like 'translate.py') are correctly installed/accessible.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    MAP_NAME = "3m"  # Popular SMAC map
    NUM_EPISODES = 1
    MAX_STEPS = 100
    RENDER_ENV = False
    VERBOSE_AGENT = True  # Controls LLMAgent's internal prints (prompts, chosen actions)
    VERBOSE_ENV_LOOP = False # Controls step-by-step prints in the main run loop

    # Initialize the LLM Agent
    llm_agent = LLMAgent(verbose=VERBOSE_AGENT)

    run_smac_with_agent(
        agent=llm_agent,
        map_name=MAP_NAME,
        episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        render=RENDER_ENV,
        verbose_env=VERBOSE_ENV_LOOP,
        save_replay=True 
    )