import random
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict, Text
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from smac.env import StarCraft2Env
from utils import get_state_NL, get_obs_agent_NL, system_prompt

class RLlibStarCraft2Env(MultiAgentEnv):
    def __init__(self, **smac_args):
        super().__init__()
        self._env = StarCraft2Env(**smac_args)
        self._ready_agents = []
        self.env_info = self._env.get_env_info()

        self.observation_space = Dict({
            "obs": Box(-1, 1, shape=(self.env_info["obs_shape"],), dtype=np.float32),
            "action_mask": Box(0, 1, shape=(self.env_info["n_actions"],), dtype=np.int32),
            "nl_obs": Box(low=0, high=0, shape=(1,), dtype=np.float32), 
        })
        self.action_space = Discrete(self.env_info["n_actions"])

    def get_agent_ids(self):
        return list(range(self.env_info["n_agents"]))

    def reset(self, *, seed=None, options=None):
        self._env.reset()
        obs_list = self._env.get_obs()
        
        return_obs = {}
        for i, obs in enumerate(obs_list):
            nl_obs_string = get_state_NL(self._env, obs)
            return_obs[i] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(i), dtype=np.int32),
                "obs": np.array(obs, dtype=np.float32),
                "nl_obs": np.array([nl_obs_string], dtype=object),
            }
        
        self._ready_agents = list(range(self.env_info["n_agents"]))
        infos = {i: {} for i in self._ready_agents}
        return return_obs, infos

    def step(self, action_dict):
        actions = []
        for i in self._ready_agents:
            if i in action_dict:
                actions.append(action_dict[i])
            else:
                actions.append(0)

        if len(actions) != self.env_info["n_agents"]:
            if len(actions) < self.env_info["n_agents"]:
                actions.extend([0] * (self.env_info["n_agents"] - len(actions)))
            else:
                actions = actions[:self.env_info["n_agents"]]
        
        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()

        return_obs = {}
        rews = {}
        dones = {}
        truncateds = {}
        infos = {} 
        
        num_active_agents = len(obs_list)
        if num_active_agents > 0:
            per_agent_reward = rew / num_active_agents
        else:
            per_agent_reward = 0.0

        self._ready_agents = []
        for i, obs in enumerate(obs_list):
            nl_obs_string = get_state_NL(self._env, obs)
            return_obs[i] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(i), dtype=np.int32),
                "obs": np.array(obs, dtype=np.float32),
                "nl_obs": np.array([nl_obs_string], dtype=object),
            }
            rews[i] = per_agent_reward
            dones[i] = done
            truncateds[i] = False
            infos[i] = info
            self._ready_agents.append(i)

        dones["__all__"] = done
        truncateds["__all__"] = False
        
        return return_obs, rews, dones, truncateds, infos

    def close(self):
        if self._env:
            self._env.close()