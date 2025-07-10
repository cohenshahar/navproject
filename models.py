import torch
import torch.nn as nn
import torch.optim as optim
import os

# Fixed input size for observation vector (based on env structure)
INPUT_SIZE = 31

# === Actor Network ===
# Takes observation → outputs probability for each action
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# === Critic with Hybrid Attention ===
# Combines own state with attention over others
class HybridAttentionCritic(nn.Module):
    def __init__(self, input_size, embed_size=128):
        super(HybridAttentionCritic, self).__init__()
        self.embed = nn.Linear(input_size, embed_size)  # embed obs+action
        self.V = nn.Linear(embed_size, embed_size, bias=False)  # value projection
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)

        # Extra MLP to learn similarity score (instead of simple dot)
        self.mlp_attention = nn.Sequential(
            nn.Linear(embed_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(embed_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs_actions, self_idx):
        embeds = self.embed(obs_actions)  # (N agents × embed_size)
        q = self.query_layer(embeds[self_idx])  # current agent's query

        weights = []
        values = []
        for i, emb in enumerate(embeds):
            if i == self_idx:
                continue
            k = self.key_layer(emb)
            score = torch.dot(q, k) / (k.size(0) ** 0.5)
            sim = self.mlp_attention(score.unsqueeze(0))
            weights.append(torch.softmax(sim, dim=0))
            values.append(self.V(emb))

        if values:
            weighted = torch.stack([w * v for w, v in zip(weights, values)], dim=0).sum(dim=0)
        else:
            weighted = torch.zeros_like(embeds[0])

        critic_input = torch.cat([embeds[self_idx], weighted], dim=-1)
        return self.out(critic_input)

# === Model Factory ===
def create_models(output_size, actor_lr=1e-3, critic_lr=1e-3):
    actor = Actor(INPUT_SIZE, output_size)
    critic = HybridAttentionCritic(INPUT_SIZE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    return actor, critic, actor_optimizer, critic_optimizer

# === Save / Load ===
def save_models(actor, critic, path="models"):
    os.makedirs(path, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(path, "actor.pt"))
    torch.save(critic.state_dict(), os.path.join(path, "critic.pt"))

def load_models(actor, critic, path="models", actor_name="actor"):
    actor_path = os.path.join(path, f"{actor_name}.pt")
    critic_path = os.path.join(path, "critic.pt")
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path))
    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path))

# === TD Error Calculation ===
def compute_td_error(reward, obs, next_obs, critic, gamma, self_idx):
    obs_batch = torch.stack([obs.squeeze(0)])
    next_obs_batch = torch.stack([next_obs.squeeze(0)])
    td_target = reward + gamma * critic(next_obs_batch, self_idx).detach()
    td_error = td_target - critic(obs_batch, self_idx)
    return td_error

# === Backprop and Update ===
def update_models(actor, critic, actor_optimizer, critic_optimizer, obs, action, td_error, self_idx):
    obs_batch = torch.stack([obs.squeeze(0)])

    # Update Critic
    critic_loss = td_error.pow(2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Update Actor
    action_dist = torch.distributions.Categorical(actor(obs_batch)[0])
    actor_loss = -action_dist.log_prob(action) * td_error.detach()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
