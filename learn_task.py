# learn_task.py
import numpy as np
import csv
import random
import torch
import os
import torch.nn.functional as F

from math_utils import ACTIONS, get_observation_vector
from warehouse import create_warehouse_grid, add_robot, add_task_shelf, move_robot
from models import create_models, compute_td_error, update_models, save_models, load_models, INPUT_SIZE
from reward import compute_reward

# === Hyperparameters ===
OUTPUT_SIZE = 5
GAMMA = 0.99
NUM_SIMULATION_RUNS = 10
NUM_SIMULATIONS = 100
MAX_STEPS = 30
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

for run in range(NUM_SIMULATION_RUNS):
    actor, critic, actor_opt, critic_opt = create_models(OUTPUT_SIZE)
    model_path = "models"
    actor_file = os.path.join(model_path, "actor1.pt")
    critic_file = os.path.join(model_path, "critic.pt")
    os.makedirs(model_path, exist_ok=True)
    load_models(actor, critic, model_path, actor_name="actor1")

    success_count = 0

    for sim in range(NUM_SIMULATIONS):
        grid, static_shelves, sorting_tables, charging_stations = create_warehouse_grid()
        robots = {}
        task_shelves = {}

        # One robot and one task shelf
        robots, grid = add_robot(grid, robots, charging_stations, 1)
        task_shelves, grid = add_task_shelf(grid, static_shelves, task_shelves, robots)

        robot_id = list(robots.keys())[0]
        robot_pos = robots[robot_id]['position']
        task_pos = list(task_shelves.values())[0]

        state = {
            'grid': np.copy(grid),
            'robots': {robot_id: robots[robot_id].copy()},
            'task_shelves': task_shelves.copy(),
            'sorting_tables': sorting_tables,
            'static_shelves': static_shelves
        }

        reached = False

        for step in range(MAX_STEPS):
            if state['robots'][robot_id]['mode'] == 'carrying':
                reached = True
                break

            obs = get_observation_vector(robot_id, state)
            if len(obs) < INPUT_SIZE:
                obs = np.pad(obs, (0, INPUT_SIZE - len(obs)))
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = actor(obs_tensor).squeeze(0)
                probs = F.softmax(logits, dim=0).numpy()
            action = np.random.choice(list(ACTIONS.keys()), p=probs)

            prev_state = state.copy()
            state = move_robot(robot_id, action, state)

            next_obs = get_observation_vector(robot_id, state)
            if len(next_obs) < INPUT_SIZE:
                next_obs = np.pad(next_obs, (0, INPUT_SIZE - len(next_obs)))
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

            reward = compute_reward(robot_id, state, prev_state, action, {
                'a': 0.5, 'b': 0.5, 'c': -0.1, 'd': -0.1, 'xi': 1, 'eta': 1
            })

            replay_buffer.push((obs_tensor, torch.tensor(action), torch.tensor([reward], dtype=torch.float32), next_obs_tensor))

        if reached:
            success_count += 1

        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            for obs_tensor, action_tensor, reward, next_obs_tensor in batch:
                td_error = compute_td_error(reward, obs_tensor, next_obs_tensor, critic, GAMMA, self_idx=0)
                update_models(actor, critic, actor_opt, critic_opt, obs_tensor, action_tensor, td_error, self_idx=0)

    print(f"[Run {run + 1}] Successes: {success_count}/{NUM_SIMULATIONS}")
    torch.save(actor.state_dict(), actor_file)
    torch.save(critic.state_dict(), critic_file)
