# Robot Arm Dynamics and Simulation

This folder contains symbolic modeling and simulation of a 4-DOF robotic arm, including free motion, PID control, energy tracking, animation, and 3D visualization.

---

## Files Overview

### 📌 `Robot_Dynamics_Symbolic.m`
- Symbolically derives the dynamic model for a 4-DOF robot (1 prismatic + 3 revolute).
- Generates:
  - Mass matrix `M(q)`
  - Coriolis matrix `C(q,dq)`
  - Gravity vector `G(q)`
  - Kinetic and potential energy expressions
- Automatically exports these as `.m` functions:
  - `M_matrix.m`, `C_matrix.m`, `G_vector.m`, `KineticEnergy.m`, `PotentialEnergy.m`

### 📌 `Robot_Simulation_Main_1.m`
- Simulates the motion of the robotic arm under:
  - Free motion (no control)
  - PID control to reach a target position
- Tracks joint trajectories, control torques, and energy.
- Animates and exports two videos:
  - `free_motion.mp4`
  - `controlled_motion.mp4`
- Saves results to:
  - `trajectory_data.csv`, `torques.csv`, `energy.csv`

---

## Additional Content

- `robot_arm.sldasm` or `.SLDPRT` – SolidWorks model of the robotic arm
- `free_motion.mp4` – Animation of the robot under no control input
- `controlled_motion.mp4` – Animation of the robot tracking a target with PID
- `lab_scene_video.mp4` – Video capture of the lab setup (optional)
- `fall_demo.mp4` – Free-fall demonstration video (optional)

---

## Requirements

- MATLAB with Symbolic Math Toolbox
- Optimization Toolbox (for inverse kinematics with `fmincon`)
- SolidWorks or compatible CAD viewer for 3D model

---

## Authors

- **Shahar Cohen** – 316490986  
- **Shaked Ozer** – 208423921
