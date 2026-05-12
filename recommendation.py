"""
recommendation.py
-----------------
Neural Collaborative Filtering (NCF) for personalised exercise recommendations.
Trains on user-exercise interaction history and predicts which exercises
a user is most likely to benefit from next.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    user_id: int
    age: int
    fitness_level: str          # "beginner", "intermediate", "advanced"
    goals: list[str]            # e.g. ["weight_loss", "muscle_gain"]
    completed_exercises: list[int] = field(default_factory=list)  # exercise IDs
    ratings: dict[int, float]   = field(default_factory=dict)     # exercise_id -> 1-5


EXERCISE_CATALOG = {
    0:  "Squat",
    1:  "Push-up",
    2:  "Lunge",
    3:  "Bicep curl",
    4:  "Plank",
    5:  "Deadlift",
    6:  "Shoulder press",
    7:  "Tricep dip",
    8:  "Glute bridge",
    9:  "Mountain climber",
    10: "Burpee",
    11: "Lateral raise",
    12: "Calf raise",
    13: "Hip thrust",
    14: "Pull-up",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    """
    Positive/negative sampling dataset for NCF training.
    Each positive (user, exercise) pair is paired with num_negatives
    randomly sampled exercises the user hasn't interacted with.
    """

    def __init__(self, interactions: list[tuple[int, int, float]],
                 num_users: int, num_items: int, num_negatives: int = 4):
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives

        self.user_item_set = {(u, i) for u, i, _ in interactions}
        self.users, self.items, self.labels = [], [], []

        for user, item, rating in interactions:
            # Positive
            self.users.append(user)
            self.items.append(item)
            self.labels.append(1.0)
            # Negatives
            for _ in range(num_negatives):
                neg = np.random.randint(0, num_items)
                while (user, neg) in self.user_item_set:
                    neg = np.random.randint(0, num_items)
                self.users.append(user)
                self.items.append(neg)
                self.labels.append(0.0)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# NCF Model
# ---------------------------------------------------------------------------

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering combining:
    - Generalised Matrix Factorisation (GMF)
    - Multi-Layer Perceptron (MLP)
    """

    def __init__(self, num_users: int, num_items: int,
                 embed_dim: int = 32, mlp_layers: list[int] = None):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        # GMF embeddings
        self.gmf_user = nn.Embedding(num_users, embed_dim)
        self.gmf_item = nn.Embedding(num_items, embed_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(num_users, embed_dim)
        self.mlp_item = nn.Embedding(num_items, embed_dim)

        # MLP tower
        layers = []
        in_size = embed_dim * 2
        for out_size in mlp_layers:
            layers += [nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

        # Output
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)

        # MLP path
        mlp_in = torch.cat([self.mlp_user(user_ids),
                            self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Concatenate and predict
        out = self.output(torch.cat([gmf, mlp_out], dim=-1))
        return self.sigmoid(out).squeeze()


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class ExerciseRecommender:
    """
    Trains NCF on collected interactions and generates top-k recommendations.
    """

    def __init__(self, num_users: int, num_items: int = len(EXERCISE_CATALOG),
                 embed_dim: int = 32, device: str = "cpu"):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.model = NCFModel(num_users, num_items, embed_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCELoss()

    def train(self, interactions: list[tuple[int, int, float]],
              epochs: int = 10, batch_size: int = 64):
        dataset = InteractionDataset(interactions, self.num_users, self.num_items)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for users, items, labels in loader:
                users  = users.to(self.device)
                items  = items.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(users, items)
                loss  = self.criterion(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == 1:
                print(f"  [NCF] Epoch {epoch}/{epochs}  Loss: {total_loss/len(loader):.4f}")

    def recommend(self, user_id: int, top_k: int = 5,
                  exclude_ids: Optional[list[int]] = None) -> list[tuple[str, float]]:
        """
        Returns top-k (exercise_name, score) tuples for the given user.
        """
        exclude = set(exclude_ids or [])
        self.model.eval()
        with torch.no_grad():
            candidates = [i for i in range(self.num_items) if i not in exclude]
            user_t = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(self.device)
            item_t = torch.tensor(candidates, dtype=torch.long).to(self.device)
            scores = self.model(user_t, item_t).cpu().numpy()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(EXERCISE_CATALOG[i], float(s)) for i, s in ranked[:top_k]]

    def save(self, path: str = "ncf_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"[NCF] Model saved to {path}")

    def load(self, path: str = "ncf_model.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[NCF] Model loaded from {path}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Exercise Recommender (NCF) Demo ===\n")

    # Simulate interactions: (user_id, exercise_id, rating)
    np.random.seed(42)
    num_users = 10
    interactions = []
    for u in range(num_users):
        for _ in range(8):
            ex = np.random.randint(0, len(EXERCISE_CATALOG))
            rating = float(np.random.choice([3, 4, 4, 5]))
            interactions.append((u, ex, rating))

    recommender = ExerciseRecommender(num_users=num_users)
    print("Training NCF model...")
    recommender.train(interactions, epochs=20)

    print("\nTop-5 recommendations for user 0:")
    recs = recommender.recommend(user_id=0, top_k=5)
    for rank, (name, score) in enumerate(recs, 1):
        print(f"  {rank}. {name:20s}  score: {score:.3f}")


if __name__ == "__main__":
    demo()
