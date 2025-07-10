import numpy as np

# === Action definitions ===
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0)    # stay
}

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "STAY"
}

# === Observation vector ===
def get_observation_vector(robot_id, env_state):
    robot = env_state['robots'][robot_id]
    pos_i = np.array(robot['position'], dtype=np.float32)
    battery = np.array([robot['battery']], dtype=np.float32)

    rel_robots = [
        np.array(env_state['robots'][rid]['position'], dtype=np.float32) - pos_i
        for rid in sorted(env_state['robots']) if rid != robot_id
    ]

    rel_task_shelves = [
        np.array(pos, dtype=np.float32) - pos_i
        for tid, pos in sorted(env_state['task_shelves'].items())
    ]

    rel_tables = [
        np.array(pos, dtype=np.float32) - pos_i
        for pos in sorted(env_state['sorting_tables'])
    ]

    rel_people = [
        np.array(pos, dtype=np.float32) - pos_i
        for pos in sorted(env_state.get('people', []))
    ]

    # Pads list to fixed length
    def pad(arr_list, target_count):
        return arr_list + [np.zeros(2, dtype=np.float32)] * (target_count - len(arr_list))

    # Final observation vector
    obs_vector = np.concatenate(
        [pos_i] +
        pad(rel_robots, 4) +
        pad(rel_task_shelves, 5) +
        pad(rel_tables, 3) +
        pad(rel_people, 2) +
        [battery]
    )
    return obs_vector

# === Move validity ===
def is_valid_move(robot_id, action, env_state, planned_moves):
    grid = env_state['grid']
    robots = env_state['robots']
    rows, cols = grid.shape

    x, y = robots[robot_id]['position']
    dx, dy = ACTIONS[action]
    new_x, new_y = x + dx, y + dy

    # Bounds check
    if not (0 <= new_x < rows and 0 <= new_y < cols):
        return False

    # Avoid other robots
    for other_id, other_data in robots.items():
        if other_id != robot_id and other_data['position'] == (new_x, new_y):
            return False

    # Avoid collision with planned moves
    for planned in planned_moves.values():
        if planned == (new_x, new_y):
            return False

    # Check for obstacles
    target_cell = grid[new_x, new_y]
    if target_cell in ['static_shelf', 'sorting_table']:
        return False

    return True
