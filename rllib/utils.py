import numpy as np
from pysc2.lib import units
from pysc2.lib.units import get_unit_type
from knowledge import protoss, zerg, terran

system_prompt = '''
Instruction:
  You are a military commander in StarCraft II, control units to eliminate all enemies.
  Try to kill more and loss less. Usually, concentrating all firepower on the same target(especially the closest enemy) can improve the strike effectiveness.
  Try to kill enemy as quick as possible. All units have an attackable range. Therefire, move units closer to the enemies to engage, and away from the enemies to retreat, especially for units with low HP.
  You will be given the current state of all ally units and enemy units, and the available actions for each ally unit. 
  Analyze the current situation and select the best action for each unit.
Output format:
  You should provide detailed analysis of the given state, and decide the appropriate actions for each agent.
  The output should contain 2 sections:
  The ANALYSIS section should contain the analysis of the current situation based on the current state of units.
  The ACTION section should contain the actions to be executed for each agent, one line for each agent. MAKE SURE each action are available for the acting agent. Possible actions are:
  no_op : Do nothing (only for dead agents)
  stop : Stand still
  move_south : Move south (which is -y direction)
  move_north : Move north (which is +y direction)
  move_east : Move east (which is +x direction)
  move_west : Move west (which is -x direction)
  attack_x : Attack unit #x
-------------------------------------------------- example input prompt --------------------------------------------------
--- CURRENT GLOBAL STATE ---
ALLY #1:
health : 20.0%
weapon cooldown : 0.0%
x coordinate : -0.189
y coordinate : 0.111
unit type : Marine
ALLY #2:
Dead.
ALLY #3:
health : 60.0%
weapon cooldown : 0.3%
x coordinate : -0.244
y coordinate : -0.048
unit type : Marine
ENEMY #1:
health : 60.0%
x coordinate : -0.045
y coordinate : -0.019
unit type : Marine
ENEMY #2:
health : 73.3%
x coordinate : -0.035
y coordinate : 0.030
unit type : Marine
ENEMY #3:
health : 100.0%
x coordinate : -0.039
y coordinate : 0.006
unit type : Marine
---UNIT INFO---


        Terran.Marine
                The basic terran infantry, able to upgrade hit points with a shield.
                Unit properties: ['ground', 'light', 'biological']
                Weapon info: Attack Range 5, target: ['ground', 'air'], DPS(damage per second) 6
                unit abilities:
                        Stimpack: Upgrade in TechLab. Increases the movement speed and firing rate by 50% for 11 seconds at the cost of 10 HP for a Marine.


--- AVAILABLE ACTIONS ---
-- Unit 1 --
Available Actions:
 - stop
 - move_north
 - move_south
 - move_east
 - move_west
 - attack_1
 - attack_2
 - attack_3

-- Unit 2 --
Available Actions:
 - no_op

-- Unit 3 --
Available Actions:
 - stop
 - move_north
 - move_south
 - move_east
 - move_west
 - attack_1
 - attack_3
-------------------------------------------------- example output prompt --------------------------------------------------
<ANALYSIS>
The battle scenario presents a critical challenge with one ally unit already lost and another severely damaged. The remaining two units must maximize firepower while minimizing exposure to enemy retaliation.
- Ally #1 has low health (20%) but is ready to attack. Being the most vulnerable unit, it must either retreat or take part in a decisive attack.
- Ally #3 has moderate health (60%) but a slight weapon cooldown (0.3%). Positioned slightly behind Ally #1, this unit should act aggressively while still prioritizing survival.
- Enemy #1 is closest to Ally #3 and has 60% health. Taking it down first will remove immediate threats.
- Enemy #2 has 73.3% health but is closer to Ally #1 than Enemy #3.
- Enemy #3 is at full health and presents the biggest challenge. It should be engaged after eliminating weaker targets.
The best course of action is focused fire on a single target, likely Enemy #1, while ensuring Ally #1 does not get eliminated prematurely.
This approach maximizes immediate damage while ensuring the weaker unit contributes before withdrawing if necessary in the next turn. The priority will be reassessing engagement after these strikes.
</ANALYSIS>
<ACTION>
- Ally #1: attack_2
- Ally #2: no_op
- Ally #3: attack_1
</ACTION>
-------------------------------------------------- example input prompt --------------------------------------------------
--- CURRENT GLOBAL STATE ---
ALLY #1:
health : 6.7%
weapon cooldown : 18.9%
x coordinate : -0.305
y coordinate : -0.131
unit type : Marine
ALLY #2:
Dead.
ALLY #3:
health : 86.7%
weapon cooldown : 0.0%
x coordinate : -0.372
y coordinate : -0.066
unit type : Marine
ENEMY #1:
health : 100.0%
x coordinate : -0.192
y coordinate : 0.004
unit type : Marine
ENEMY #2:
health : 73.3%
x coordinate : -0.190
y coordinate : -0.021
unit type : Marine
ENEMY #3:
health : 73.3%
x coordinate : -0.215
y coordinate : 0.010
unit type : Marine
---UNIT INFO---


        Terran.Marine
                The basic terran infantry, able to upgrade hit points with a shield.
                Unit properties: ['ground', 'light', 'biological']
                Weapon info: Attack Range 5, target: ['ground', 'air'], DPS(damage per second) 6
                unit abilities:
                        Stimpack: Upgrade in TechLab. Increases the movement speed and firing rate by 50% for 11 seconds at the cost of 10 HP for a Marine.


--- AVAILABLE ACTIONS ---

-- Unit 1 --
Available Actions:
 - stop
 - move_north
 - move_south
 - move_east
 - move_west
 - attack_1
 - attack_2
 - attack_3

-- Unit 2 --
Available Actions:
 - no_op

-- Unit 3 --
Available Actions:
 - stop
 - move_north
 - move_south
 - move_east
 - move_west
 - attack_1
 - attack_2
 - attack_3
-------------------------------------------------- example output prompt --------------------------------------------------
<ANALYSIS>
The situation remains critical:
- Ally #1 is very low on health (6.7%) and has a long weapon cooldown (72.3%), meaning it cannot attack. Immediate retreat is the best course of action.
- Ally #3 has high health (86.7%), is ready to fire, and must engage aggressively to offset Ally #1’s retreat.
- Enemy #1 is at full health (100%) and presents a major threat.
- Enemies #2 and #3 both have 73.3% health, but Enemy #3 is slightly closer to Ally #3.
This ensures Ally #1 stays alive while maintaining offensive pressure on weakened targets. Next steps will involve repositioning Ally #3 for sustained attacks.
</ANALYSIS>
<ACTION>
- Ally #1: move_west
- Ally #2: no_op
- Ally #3: attack_3
</ACTION>
'''


def get_action_description(action_id, n_total_actions):
    """Provides a basic textual description for an action ID.
    n_total_actions is needed to distinguish between move and attack actions correctly.
    """
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
    """Converts a textual action description back to a numerical action ID."""
    action_map = {
        "no_op": 0, "stop": 1, "move_north": 2, "move_south": 3,
        "move_east": 4, "move_west": 5,
    }
    try:
        if action_desc in action_map:
            return action_map[action_desc]
        elif action_desc.startswith("attack_"):
            return int(action_desc[7:]) + 5
        else:
            return -1
    except Exception:
        return -1

knowledge_dict = {}
knowledge_dict.update(protoss.DATA_PROTOSS)
knowledge_dict.update(terran.DATA_TERRAN)
knowledge_dict.update(zerg.DATA_ZERG)
type_dict = {v['name']: k for k,v in knowledge_dict.items()}

def get_single_unit_type_knowledge(unit_name) -> str:
  unit_type_knowledge = ''
  unit_type = type_dict[unit_name]
  if unit_type not in knowledge_dict.keys():
    return ''
  if 'Protoss' in str(units.get_unit_type(unit_type)):
    unit_type_knowledge += f"\n\t{str(units.Protoss(unit_type))}"
  if 'Terran' in str(units.get_unit_type(unit_type)):
    unit_type_knowledge += f"\n\t{str(units.Terran(unit_type))}"
  if 'Zerg' in str(units.get_unit_type(unit_type)):
    unit_type_knowledge += f"\n\t{str(units.Zerg(unit_type))}"

  if 'description' in knowledge_dict[unit_type].keys():
    unit_type_knowledge += f"\n\t\t{knowledge_dict[unit_type]['description']}"
  unit_knowledge = knowledge_dict[unit_type]
  unit_type_knowledge += f"\n\t\tUnit properties: {unit_knowledge['target_self'] + unit_knowledge['type_self']}"
  if 'weapon1_attack_range' in unit_knowledge.keys() and unit_knowledge['weapon1_attack_range'] not in [0, -1]:
    unit_type_knowledge += f"\n\t\tWeapon info: Attack Range {unit_knowledge['weapon1_attack_range']}"
  if 'target' in unit_knowledge.keys() and len(unit_knowledge['target']) != 0:
    unit_type_knowledge += f", target: {unit_knowledge['target']}"
  if 'type_anti' in unit_knowledge.keys() and len(unit_knowledge['type_anti']) != 0:
    unit_type_knowledge += f", anti: {unit_knowledge['type_anti']}"
  if 'weapon1_attack' in unit_knowledge.keys() and unit_knowledge['weapon1_attack'] not in [0, -1]:
    unit_type_knowledge += f", DPS(damage per second) {int(unit_knowledge['weapon1_attack'] * unit_knowledge['weapon1_attack_times'] / unit_knowledge['weapon1_cooldown'])}"
  if 'weapon1_attack_bonus' in unit_knowledge.keys() and unit_knowledge['weapon1_attack_bonus'] not in [0, -1]:
    unit_type_knowledge += f", DPS-anti {int((unit_knowledge['weapon1_attack'] + unit_knowledge['weapon1_attack_bonus']) * unit_knowledge['weapon1_attack_times'] / unit_knowledge['weapon1_cooldown'])}"
  if 'ability' in unit_knowledge.keys():
    unit_type_knowledge += f"\n\t\tunit abilities:"
    for ability in unit_knowledge['ability'].keys():
      unit_type_knowledge += f"\n\t\t\t{ability}: {unit_knowledge['ability'][ability]}"
  return unit_type_knowledge
def get_relevant_knowledge(env):
    enemy_count = env.get_obs_enemy_feats_size()[0]
    ally_count = env.get_obs_ally_feats_size()[0] + 1
    unit_dict = set()
    for unit in env.agents.values():
        unit_name = get_ally_unit_type(env,unit.unit_type)
        unit_dict.add(unit_name)
    for unit in env.enemies.values():
        unit_name = get_unit_type(unit.unit_type).name
        unit_dict.add(unit_name)
    unit_dict = list(unit_dict)
    return '---UNIT INFO---\n\n' + '\n'.join([get_single_unit_type_knowledge(u) for u in unit_dict]) + '\n\n'
def get_ally_unit_type(env,unit_id):
    if unit_id == env.marine_id:
        return "Marine"
    elif unit_id == env.stalker_id:
        return "Stalker"
    elif unit_id == env.zealot_id:
        return "Zealot"
    elif unit_id == env.colossus_id:
        return "Colossus"
    elif unit_id == env.marauder_id:
        return "Marauder"
    elif unit_id == env.medivac_id:
        return "Medivac"
    elif unit_id == env.hydralisk_id:
        return "Hydralisk"
    elif unit_id == env.baneling_id:
        return "Baneling"
    elif unit_id == env.zergling_id:
        return "Zergling"
    else:
        return "Unknown"
def get_state_NL(env,state,knowledge=True):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if env.obs_instead_of_state:
            # why?
            pass
        s = ''
        base_idx = 0
        enemy_count = env.get_obs_enemy_feats_size()[0]
        ally_count = env.get_obs_ally_feats_size()[0] + 1
        nf_al = env.get_ally_num_attributes()
        nf_en = env.get_enemy_num_attributes()
        # Ally features
        for e in range(ally_count):
            feats = state[base_idx:base_idx+nf_al] 
            base_idx += nf_al
            s += f'ALLY #{e+1}:\n'
            if np.all(feats == 0):
                s += 'Dead.\n'
                continue
            e_dict = {'health':feats[0],'weapon cooldown':feats[1],'x':feats[2],'y':feats[3],'unit':env.agents[e].unit_type}
            ind = 4
            if env.shield_bits_ally > 0:
                e_dict['shield'] = feats[ind]
                ind += 1
            if env.unit_type_bits > 0:
                ind += 1
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield' or k == 'weapon cooldown':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_ally_unit_type(env,v)}\n'
                elif k == 'x' or k == 'y':
                    s += f'{k} coordinate : {v:.3f}\n'
        for e in range(enemy_count):
            feats = state[base_idx:base_idx+nf_en] 
            base_idx += nf_en
            s += f'ENEMY #{e+1}:\n'
            if np.all(feats == 0):
                s += 'Dead.\n'
                continue
            e_dict = {'health':feats[0],'x':feats[1],'y':feats[2],'unit':env.enemies[e].unit_type}
            ind = 4
            if env.shield_bits_enemy > 0:
                e_dict['shield'] = feats[ind]
                ind += 1
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield' or k == 'weapon cooldown':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_unit_type(v).name}\n'
                elif k == 'x' or k == 'y':
                    s += f'{k} coordinate : {v:.3f}\n'
        
        if env.state_last_action:
            pass
        if env.state_timestep_number:
            pass
        if knowledge:
            return s + get_relevant_knowledge(env)
        return s
def get_obs_agent_NL(env, obs, agent_id):
        enemy_feats_dim = env.get_obs_enemy_feats_size()
        ally_feats_dim = env.get_obs_ally_feats_size()
        if np.all(obs == 0):
            return None # agent is dead
        base_idx = 4
        dir_list = ["north","south","east","west"]
        s = ''
        s += 'Available move directions:\n'+"\n".join([dir_list[i] for i in range(4) if obs[i] > 0])+'\n'
        # if you want...
        '''ind = env.n_actions_move

        if env.obs_pathing_grid:
            move_feats[
                ind : ind + env.n_obs_pathing  # noqa
            ] = env.get_surrounding_pathing(unit)
            ind += env.n_obs_pathing

        if env.obs_terrain_height:
            move_feats[ind:] = env.get_surrounding_height(unit)'''
        s += 'ENEMY STATISTICS(visible only):\n'
        # Enemy features
        for e in range(enemy_feats_dim[0]):
            
            feats = obs[base_idx:base_idx+enemy_feats_dim[1]] 
            base_idx += enemy_feats_dim[1]
            s += f'ENEMY #{e+1}:\n'
            if np.all(feats == 0):
                s += 'Not visible or dead.\n'
                continue
            e_dict = {'attack':feats[0],'distance':feats[1],'x':feats[2],'y':feats[3],'unit':env.enemies[e].unit_type}
            ind = 4
            if env.obs_all_health:
                e_dict['health'] = feats[ind]
                ind += 1
                if env.shield_bits_enemy > 0:
                    e_dict['shield'] = feats[ind]
                    ind += 1
                if env.unit_type_bits > 0:
                    ind += 1
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_unit_type(v).name}\n'
                elif k == 'x' or k == 'y':
                    s += f'relative {k} : {v:.3f}\n'
                elif k == 'distance':
                    s += f'distance : {v:.3f}\n'
                elif k == 'attack':
                    s += f'Is attackable : {v > 0}\n'
            
            
        s += '\nALLY STATISTICS(visible only):\n'
        for a in range(ally_feats_dim[0]):
            feats = obs[base_idx:base_idx+ally_feats_dim[1]] 
            base_idx += ally_feats_dim[1]
            s += f'ALLY #{a+1}:\n'
            if np.all(feats == 0):
                s += 'Not visible or dead.\n'
                continue
            e_dict = {'visible':feats[0],'distance':feats[1],'x':feats[2],'y':feats[3],'unit':env.agents[a if a < agent_id else a+1].unit_type}
            ind = 4
            if env.obs_all_health:
                e_dict['health'] = feats[ind]
                ind += 1
                if env.shield_bits_ally > 0:
                    e_dict['shield'] = feats[ind]
                    ind += 1
            if env.unit_type_bits > 0:
                ind += 1
            #if env.obs_last_action:
            #    if you want...?
            for k,v in e_dict.items():
                if k == 'health' or k == 'shield':
                    s += f'{k} : {v*100:.1f}%\n'
                elif k == 'unit':
                    s += f'unit type : {get_ally_unit_type(env,v)}\n'
                elif k == 'x' or k == 'y':
                    s += f'relative {k} : {v:.3f}\n'
                elif k == 'distance':
                    s += f'distance : {v:.3f}\n'
                elif k == 'visible': # it's always visible ......
                    continue 
        
        # Own features
        s += '\nAgent information:\n'
        if env.obs_own_health:
            s += f'Health : {100*obs[base_idx]:.1f}%\n'
            base_idx += 1
        s += f'unit type : {get_ally_unit_type(env,env.agents[agent_id].unit_type)}\n'
        base_idx += 1
        return s
def get_obs_NL(env,obs_list):
    return [get_obs_agent_NL(env,obs,i) for i,obs in enumerate(obs_list)]
