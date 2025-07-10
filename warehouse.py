import numpy as np
import matplotlib.pyplot as plt
import random
from math_utils import ACTIONS

# === Grid layout ===
def create_warehouse_grid():
    rows, cols = 15, 21
    grid = np.full((rows, cols), 'empty', dtype=object)
    static_shelves = []

    # Add static shelves in 4x4 groups (each 2x4)
    for row in range(4):
        for col in range(4):
            top_left_x = 1 + row * 3
            top_left_y = 1 + col * 5
            for dx in range(2):
                for dy in range(4):
                    x = top_left_x + dx
                    y = top_left_y + dy
                    if x < rows and y < cols:
                        grid[x, y] = 'static_shelf'
                        static_shelves.append((x, y))

    # Add sorting tables
    sorting_positions = [(13, 5), (13, 10), (13, 15)]
    for x, y in sorting_positions:
        grid[x, y] = 'sorting_table'

    # Add charging stations
    charging_stations = [(3, 0), (6, 0), (9, 0), (3, 20), (6, 20), (9, 20)]
    for x, y in charging_stations:
        grid[x, y] = 'charging_station'

    return grid, static_shelves, sorting_positions, charging_stations

# === Visualization ===
def plot_warehouse(grid, robots=None, task_shelves=None):
    color_map = {
        'empty': 'white',
        'static_shelf': 'gray',
        'sorting_table': 'red',
        'charging_station': 'yellow',
    }

    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw cells
    for x in range(rows):
        for y in range(cols):
            val = grid[x, y]
            color = color_map.get(val, 'white')
            if isinstance(val, str) and val.startswith('task_shelf'):
                color = 'magenta'
                shelf_num = val.split('_')[-1]
                ax.text(y + 0.5, x + 0.5, shelf_num, color='white', ha='center', va='center', fontsize=7, zorder=6)
            ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=color, edgecolor='black'))

    # Draw robots
    if robots:
        for rid, data in robots.items():
            rx, ry = data['position']
            mode = data.get('mode', 'idle')
            color = 'magenta' if mode == 'carrying' else 'gray' if mode == 'delivered' else 'green'
            ax.add_patch(plt.Circle((ry + 0.5, rx + 0.5), 0.3, color=color, zorder=5))
            ax.text(ry + 0.5, rx + 0.5, rid.split('_')[-1], color='white', ha='center', va='center', fontsize=8, zorder=6)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title("Warehouse Layout")
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Robot control ===
def add_robot(grid, robots, charging_stations, robot_number):
    robot_id = f"robot_{robot_number}"
    occupied = [r['position'] for r in robots.values()]
    free = [pos for pos in charging_stations if pos not in occupied]
    if not free:
        print("No available charging stations.")
        return robots, grid
    pos = random.choice(free)
    robots[robot_id] = {
        'position': pos,
        'battery': 1.0,
        'carrying': None,
        'mode': 'idle'
    }
    return robots, grid

def add_task_shelf(grid, static_shelves, task_shelves, robots):
    if len(task_shelves) >= len(robots):
        print("Too many task shelves.")
        return task_shelves, grid
    available = [pos for pos in static_shelves if grid[pos] == 'static_shelf']
    if not available:
        print("No free static shelf.")
        return task_shelves, grid
    pos = random.choice(available)
    task_id = f"task_shelf_{len(task_shelves) + 1}"
    task_shelves[task_id] = pos
    grid[pos] = task_id
    return task_shelves, grid

def assign_task_to_robot(grid, robots, task_shelves):
    assigned = [r['carrying'] for r in robots.values() if r['carrying']]
    unassigned = [(tid, pos) for tid, pos in task_shelves.items() if tid not in assigned]
    for rid, data in robots.items():
        if data['carrying'] is None and unassigned:
            shelf_id, _ = unassigned.pop(0)
            data['carrying'] = shelf_id
            data['mode'] = 'to_shelf'
    return robots, grid

def move_robot(robot_id, action, env_state):
    robot = env_state['robots'][robot_id]
    x, y = robot['position']
    dx, dy = ACTIONS[action]
    new_x, new_y = x + dx, y + dy
    grid = env_state['grid']
    rows, cols = grid.shape

    moved = False
    if 0 <= new_x < rows and 0 <= new_y < cols:
        cell = grid[new_x, new_y]
        occupied = [r['position'] for rid, r in env_state['robots'].items() if rid != robot_id]
        if (new_x, new_y) not in occupied and cell not in ['static_shelf', 'sorting_table']:
            robot['position'] = (new_x, new_y)
            moved = True

    robot['battery'] = max(0.0, robot['battery'] - (0.01 if moved else 0.001))
    return env_state

# === Task handling ===
def pickup_shelf(robot_id, env_state):
    robot = env_state['robots'][robot_id]
    pos = robot['position']
    task_shelves = env_state['task_shelves']
    for sid, spos in task_shelves.items():
        if spos == pos and robot['carrying'] == sid:
            robot['return_to'] = pos
            robot['mode'] = 'carrying'
            del task_shelves[sid]
            print(f"{robot_id} picked up {sid} at {pos}")
            return env_state
    return env_state

def drop_at_sorting(robot_id, env_state):
    robot = env_state['robots'][robot_id]
    pos = robot['position']
    if robot['mode'] == 'carrying' and pos in env_state['sorting_tables']:
        robot['mode'] = 'delivered'
        print(f"{robot_id} delivered shelf at sorting table {pos}")
    return env_state

def return_shelf(robot_id, env_state):
    robot = env_state['robots'][robot_id]
    pos = robot['position']
    if robot['mode'] == 'delivered' and 'return_to' in robot and pos == robot['return_to']:
        env_state['grid'][pos[0], pos[1]] = 'static_shelf'
        robot['carrying'] = None
        robot['mode'] = 'idle'
        del robot['return_to']
        print(f"{robot_id} returned shelf to {pos}")
    return env_state
