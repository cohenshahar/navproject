# Multi-Robot Warehouse Simulation (MAHAC-Based)

This project simulates a smart logistic warehouse with autonomous robots, using deep reinforcement learning with a Hybrid Attention Critic architecture based on the MAHAC (2024) model.

---

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib

---

## How to Run

### Stage 1 – Basic Learning (approaching the shelf only):
```bash
python learn_task.py
```
* Focuses on learning to reach the task shelf.
* Saves model weights to `models/actor.pt` and `models/critic.pt`.

### Stage 2 – Full Task Learning (including delivery to sorting table):
```bash
python learn_full_task.py
```
* Trains the robot to complete the full sequence:  
  approach shelf → pick up → deliver to sorting table → drop off.
* Includes distance-based reward shaping.
* Model weights are saved in the `models/` directory.

### Display Model Architecture:
```bash
python read_models.py
```

---

## Core Files

### `math_utils.py`
Defines mathematical structures:
- `ACTIONS`: Maps integers to movement directions (up, down, left, right, stay).
- `get_observation_vector()`: Computes the robot's observation vector.
- `is_valid_move()`: Checks if a move is legal.

### `warehouse.py`
Manages the warehouse environment:
- `create_warehouse_grid()`: Builds the grid with shelves, charging stations, and sorting tables.
- `add_robot()`, `add_task_shelf()`: Places robots and task shelves.
- `move_robot()`: Moves a robot according to an action.
- `pickup_shelf()`, `drop_at_sorting()`, `return_shelf()`: Handle task progress.
- `plot_warehouse()`: Visualizes the warehouse grid.

### `reward.py`
- `compute_reward()`: Reward function combining task completion, collisions, distances, and battery level.

### `models.py`
- `Actor`: A neural network that outputs action probabilities.
- `HybridAttentionCritic`: Critic network with hybrid attention.
- Includes model creation, TD-error calculation, and parameter updates.

### `learn_task.py`
- Basic learning stage.
- The robot only learns to approach the task shelf (no delivery to sorting table).
- Each run includes 1 robot, 1 shelf, and a 30-step limit.
- Saves model weights at the end of training.

### `learn_full_task.py`
- Full task learning stage.
- Includes shelf pickup, delivery to sorting table, and drop-off.
- Uses multiple simulated "players" per episode.
- Applies distance-based shaping and a replay buffer.
- Saves updated model weights after each run.

### `read_models.py`
- Prints the architecture of saved PyTorch models (`actor.pt`, `critic.pt`).

---

## Folder Structure

```
project_folder/
├── part2/
│   ├── learn_task.py
│   ├── learn_full_task.py
│   ├── read_models.py
│   ├── math_utils.py
│   ├── warehouse.py
│   ├── reward.py
│   ├── models.py
│   ├── README.md
│   └── models/
│       ├── actor.pt
│       ├── actors.pt
│       └── critic.pt
```

---

## Notes

* The `models/` folder is created automatically if it does not exist.
