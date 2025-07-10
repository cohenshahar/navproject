import numpy as np

# Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Calculate reward for one robot
def compute_reward(robot_id, env_state, prev_env_state, action, reward_params):
    a = reward_params.get('a', 0.5)
    b = reward_params.get('b', 0.5)
    c = reward_params.get('c', -0.1)
    d = reward_params.get('d', -0.1)
    xi_default = reward_params.get('xi', 1)
    eta_default = reward_params.get('eta', 1)

    reward = 0.0
    delta = 0

    curr_robot = env_state['robots'][robot_id]
    prev_robot = prev_env_state['robots'][robot_id]
    pos = curr_robot['position']
    prev_pos = prev_robot['position']

    # Task complete
    r_target = 0.0
    if prev_robot.get('mode') == 'carrying' and curr_robot.get('mode') == 'delivered':
        r_target = a + b

    # Collision check
    other_positions = [r['position'] for rid, r in env_state['robots'].items() if rid != robot_id]
    if pos in other_positions or pos in env_state['static_shelves'] or pos in env_state.get('people', []):
        delta = 1
    r_collision = c * delta

    # Distance progress
    goal_pos = None
    xi, eta = 0, 0
    if curr_robot.get('mode') == 'to_shelf':
        if curr_robot['carrying'] in env_state['task_shelves']:
            goal_pos = env_state['task_shelves'][curr_robot['carrying']]
            xi = xi_default
    elif curr_robot.get('mode') == 'carrying':
        goal_pos = min(env_state['sorting_tables'], key=lambda p: euclidean_distance(pos, p))
        eta = eta_default

    r_distance = 0.0
    if goal_pos is not None:
        prev_dist = euclidean_distance(prev_pos, goal_pos)
        curr_dist = euclidean_distance(pos, goal_pos)
        if curr_dist < prev_dist:
            r_distance = 0.1
        elif curr_dist > prev_dist:
            r_distance = -0.01

    # Low battery penalty
    r_electric = d if curr_robot['battery'] < 0.1 else 0.0

    return r_target + r_collision + r_distance + r_electric
