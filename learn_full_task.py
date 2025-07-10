import numpy as np
import random
import torch
import os
import torch.nn.functional as F

from math_utils import ACTIONS, get_observation_vector
from warehouse import create_warehouse_grid, add_robot, move_robot, pickup_shelf, drop_at_sorting
from models import create_models, compute_td_error, update_models, load_models, INPUT_SIZE
from reward import compute_reward

# === Configuration ===
OUTPUT_SIZE = 5
GAMMA = 0.99
NUM_RUNS = 10
NUM_SIMULATIONS = 50
NUM_PLAYERS = 50
MAX_STEPS = 50
BATCH_SIZE = 32

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer()

# === Main Training Loop ===
for run in range(NUM_RUNS):
    actor, critic, actor_opt, critic_opt = create_models(OUTPUT_SIZE)
    load_models(actor, critic, path="models", actor_name="actors")

    success_count = 0

    for sim in range(NUM_SIMULATIONS):
        grid, static_shelves, sorting_tables, charging_stations = create_warehouse_grid()
        robots = {}

        # === Initialize robot and task shelf ===
        start_pos = charging_stations[0]
        robot_id = "robot_1"

        task_shelves = {}
        available_static = [pos for pos in static_shelves if grid[pos[0], pos[1]] == 'static_shelf']
        position = random.choice(available_static)
        x, y = position
        task_id = "task_shelf_1"
        task_shelves[task_id] = (x, y)
        grid[x, y] = task_id
        if not task_shelves:
            continue

        task_pos = task_shelves[task_id]
        sorting_target = sorting_tables[0]

        robots[robot_id] = {
            'position': start_pos,
            'battery': 1.0,
            'carrying': task_id,
            'mode': 'to_shelf'
        }

        # === Generate players ===
        players = []
        for _ in range(NUM_PLAYERS):
            state = {
                'grid': np.copy(grid),
                'robots': {robot_id: robots[robot_id].copy()},
                'task_shelves': task_shelves.copy(),
                'sorting_tables': sorting_tables,
                'static_shelves': static_shelves
            }
            players.append({
                "state": state,
                "observations": [],
                "actions": [],
                "done": False,
                "start_dist": np.linalg.norm(np.array(start_pos) - np.array(task_pos))
            })

        # === Simulation ===
        for step in range(MAX_STEPS):
            for player in players:
                if player['done']:
                    continue

                curr_state = player['state']
                curr_mode = curr_state['robots'][robot_id]['mode']

                # === Drop shelf if at sorting ===
                if curr_mode == 'carrying':
                    if curr_state['robots'][robot_id]['position'] == sorting_target:
                        curr_state = drop_at_sorting(robot_id, curr_state)
                    if curr_state['robots'][robot_id]['mode'] == 'delivered':
                        player['done'] = True
                        success_count += 1
                        continue

                obs = get_observation_vector(robot_id, curr_state)
                if len(obs) < INPUT_SIZE:
                    obs = np.pad(obs, (0, INPUT_SIZE - len(obs)))
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    probs = F.softmax(actor(obs_tensor).squeeze(0), dim=0).numpy()
                action = np.random.choice(list(ACTIONS.keys()), p=probs)

                prev_state = {
                    'grid': np.copy(curr_state['grid']),
                    'robots': {robot_id: curr_state['robots'][robot_id].copy()},
                    'task_shelves': curr_state['task_shelves'].copy(),
                    'sorting_tables': curr_state['sorting_tables'],
                    'static_shelves': curr_state['static_shelves']
                }

                curr_state = move_robot(robot_id, action, curr_state)

                curr_pos = curr_state['robots'][robot_id]['position']
                if np.linalg.norm(np.array(curr_pos) - np.array(task_pos)) <= 1.0:
                    curr_state = pickup_shelf(robot_id, curr_state)

                curr_state = drop_at_sorting(robot_id, curr_state)
                player['state'] = curr_state

                goal_pos = task_pos if curr_state['robots'][robot_id]['mode'] != 'carrying' else sorting_target

                next_obs = get_observation_vector(robot_id, curr_state)
                if len(next_obs) < INPUT_SIZE:
                    next_obs = np.pad(next_obs, (0, INPUT_SIZE - len(next_obs)))
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

                before_dist = np.linalg.norm(np.array(prev_state['robots'][robot_id]['position']) - np.array(goal_pos))
                after_dist = np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))
                dist_diff = before_dist - after_dist

                shaped_reward = 0.0
                if dist_diff > 0.5:
                    shaped_reward = 0.2
                elif dist_diff > 0.1:
                    shaped_reward = 0.1
                elif dist_diff < -0.1:
                    shaped_reward = -0.1
                else:
                    shaped_reward = -0.02

                task_reward = compute_reward(robot_id, curr_state, prev_state, action, {
                    'a': 1.0, 'b': 1.0, 'c': -0.1, 'd': -0.1, 'xi': 1, 'eta': 1
                })

                total_reward = task_reward + shaped_reward

                player['observations'].append((obs_tensor, action, next_obs_tensor, total_reward))

                replay_buffer.push((obs_tensor, torch.tensor(action), torch.tensor([total_reward], dtype=torch.float32), next_obs_tensor))

        # === Train models ===
        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            for obs_tensor, action_tensor, reward, next_obs_tensor in batch:
                td_error = compute_td_error(reward, obs_tensor, next_obs_tensor, critic, GAMMA, self_idx=0)
                update_models(actor, critic, actor_opt, critic_opt, obs_tensor, action_tensor, td_error, self_idx=0)

    print(f"[Run {run + 1}] Successes: {success_count}/{NUM_SIMULATIONS * NUM_PLAYERS}")
    torch.save(actor.state_dict(), "models/actors.pt")
    torch.save(critic.state_dict(), "models/critic.pt")
