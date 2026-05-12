"""
rl_agent.py
-----------
Reinforcement Learning agent that refines exercise recommendations
based on continuous user progress and performance feedback.

Uses a Q-learning approach with a neural network (DQN-style) to
map user state -> optimal next exercise action.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# State / Action definitions
# ---------------------------------------------------------------------------

@dataclass
class UserState:
    """
    Encodes the user's current fitness state as a feature vector.
    All values normalised to [0, 1].
    """
    fitness_level: float        # 0=beginner, 0.5=intermediate, 1=advanced
    fatigue_level: float        # 0=fresh, 1=exhausted
    recent_form_score: float    # average form score last 3 sessions (0-1)
    sessions_completed: float   # normalised count (0-1 over 100 sessions)
    goal_progress: float        # 0=no progress, 1=goal achieved
    last_exercise_id: float     # normalised exercise id

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.fitness_level,
            self.fatigue_level,
            self.recent_form_score,
            self.sessions_completed,
            self.goal_progress,
            self.last_exercise_id,
        ], dtype=torch.float32)

    @property
    def dim(self) -> int:
        return 6


@dataclass
class ExerciseSession:
    exercise_id: int
    form_score: float       # 0-1
    reps_completed: int
    duration_sec: int
    perceived_effort: float  # 1-10 scale, normalised


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN network
# ---------------------------------------------------------------------------

class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden: list[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [128, 64]
        layers = []
        in_size = state_dim
        for h in hidden:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(session: ExerciseSession, prev_form: float,
                   fatigue: float) -> float:
    """
    Reward signal:
    +  form improvement
    +  completing target reps
    -  high fatigue (overtraining penalty)
    -  form degradation
    """
    reward = 0.0

    # Form quality (primary signal)
    reward += session.form_score * 2.0

    # Form improvement bonus
    reward += max(0, session.form_score - prev_form) * 1.5

    # Penalise overtraining
    if fatigue > 0.8:
        reward -= 1.0
    elif fatigue > 0.6:
        reward -= 0.4

    # Rep completion bonus
    if session.reps_completed >= 10:
        reward += 0.5

    # Penalise very poor form
    if session.form_score < 0.4:
        reward -= 1.0

    return float(np.clip(reward, -3.0, 3.0))


# ---------------------------------------------------------------------------
# RL Agent
# ---------------------------------------------------------------------------

class FitnessRLAgent:
    """
    DQN-based agent that recommends the best next exercise
    given the user's current state.

    Training loop:
    1. Agent observes state
    2. Selects action (exercise) via epsilon-greedy
    3. User performs exercise -> session result returned
    4. Reward computed, transition stored in replay buffer
    5. Network updated via Q-learning loss
    """

    def __init__(self, num_exercises: int = 15, state_dim: int = 6,
                 device: str = "cpu", lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995):

        self.num_exercises = num_exercises
        self.state_dim = state_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQNetwork(state_dim, num_exercises).to(device)
        self.target_net = DQNetwork(state_dim, num_exercises).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer()

        self._step = 0
        self._target_update_freq = 50

    def select_action(self, state: UserState,
                      exclude_ids: Optional[list[int]] = None) -> int:
        """Epsilon-greedy action selection."""
        exclude = set(exclude_ids or [])
        valid  = [i for i in range(self.num_exercises) if i not in exclude]

        if random.random() < self.epsilon:
            return random.choice(valid)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state.to_tensor().unsqueeze(0).to(self.device))
            q_values = q_values.squeeze().cpu().numpy()
        # Mask excluded
        for i in exclude:
            q_values[i] = -1e9
        return int(np.argmax(q_values))

    def update(self, state: UserState, action: int, reward: float,
               next_state: UserState, done: bool, batch_size: int = 32):
        """Store transition and update policy network."""
        self.buffer.push(
            state.to_tensor(), action, reward,
            next_state.to_tensor(), float(done)
        )

        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q values
        self.policy_net.train()
        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (Bellman)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            targets    = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.criterion(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Sync target network
        self._step += 1
        if self._step % self._target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str = "rl_agent.pt"):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"[RL] Agent saved to {path}")

    def load(self, path: str = "rl_agent.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = ckpt["epsilon"]
        print(f"[RL] Agent loaded from {path}")


# ---------------------------------------------------------------------------
# Simulation demo
# ---------------------------------------------------------------------------

def simulate_session(exercise_id: int, state: UserState) -> ExerciseSession:
    """Simulate a workout session outcome (replace with real data in production)."""
    base_form = state.recent_form_score
    form = float(np.clip(base_form + np.random.normal(0.05, 0.15), 0, 1))
    reps = int(np.clip(np.random.normal(12, 3), 5, 20))
    return ExerciseSession(
        exercise_id=exercise_id,
        form_score=form,
        reps_completed=reps,
        duration_sec=int(reps * 3.5),
        perceived_effort=float(np.clip(np.random.normal(6, 1.5), 1, 10)) / 10,
    )


def demo():
    from recommendation import EXERCISE_CATALOG
    print("=== RL Agent Simulation Demo ===\n")

    agent = FitnessRLAgent(num_exercises=len(EXERCISE_CATALOG))

    state = UserState(
        fitness_level=0.3,
        fatigue_level=0.2,
        recent_form_score=0.6,
        sessions_completed=0.1,
        goal_progress=0.05,
        last_exercise_id=0.0,
    )

    total_reward = 0.0
    num_episodes = 200

    for episode in range(1, num_episodes + 1):
        action = agent.select_action(state)
        session = simulate_session(action, state)

        prev_form = state.recent_form_score
        reward = compute_reward(session, prev_form, state.fatigue_level)
        total_reward += reward

        # Evolve state
        next_state = UserState(
            fitness_level=min(1.0, state.fitness_level + 0.002),
            fatigue_level=float(np.clip(
                state.fatigue_level + session.perceived_effort * 0.1 - 0.05, 0, 1)),
            recent_form_score=float(
                0.7 * state.recent_form_score + 0.3 * session.form_score),
            sessions_completed=min(1.0, state.sessions_completed + 0.01),
            goal_progress=min(1.0, state.goal_progress + reward * 0.01),
            last_exercise_id=action / len(EXERCISE_CATALOG),
        )

        loss = agent.update(state, action, reward, next_state, done=False)
        state = next_state

        if episode % 50 == 0:
            rec = EXERCISE_CATALOG.get(action, str(action))
            print(f"  Episode {episode:3d} | Recommended: {rec:20s} | "
                  f"Reward: {reward:+.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg reward: {total_reward/episode:.3f}")

    print(f"\nTraining complete. Final epsilon: {agent.epsilon:.4f}")
    print(f"Total cumulative reward: {total_reward:.2f}")


if __name__ == "__main__":
    demo()
