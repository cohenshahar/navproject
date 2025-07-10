# inspect_models.py
import torch
import os

actor_path = os.path.join("models", "actor1.pt")
critic_path = os.path.join("models", "critic.pt")

# Load the saved model weights
actor_state = torch.load(actor_path)
critic_state = torch.load(critic_path)

print("\n=== ACTOR MODEL WEIGHTS ===")
for name, param in actor_state.items():
    print(f"{name}: {param.shape}")

print("\n=== CRITIC MODEL WEIGHTS ===")
for name, param in critic_state.items():
    print(f"{name}: {param.shape}")
