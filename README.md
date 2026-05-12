# AI Fitness Coach

A real-time AI fitness system combining **Human Pose Estimation**, **Neural Collaborative Filtering**, **Personalised Nutrition Planning**, and **Reinforcement Learning** into a single pipeline.

---

## Project Structure

```
ai_fitness_coach/
├── main.py              # Entry point — runs the full pipeline
├── pose_estimation.py   # MediaPipe pose tracking + form assessment
├── recommendation.py    # Neural Collaborative Filtering (NCF)
├── nutrition.py         # Personalised meal planning
├── rl_agent.py          # DQN-based adaptive recommendations
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Full pipeline simulation (no camera needed)
python main.py --mode full

# Run individual module demos
python main.py --mode demo

# Live webcam pose estimation
python main.py --mode live
```

---

## Module Overview

### 1. Pose Estimation (`pose_estimation.py`)
- Uses **MediaPipe Pose** to detect 33 body landmarks in real time
- Extracts `(x, y)` joint coordinates per frame
- Calculates joint angles (elbows, knees, hips, shoulders)
- `FormAssessor` provides rule-based coaching cues for squats, push-ups, bicep curls, and lunges

### 2. Exercise Recommender (`recommendation.py`)
- **Neural Collaborative Filtering** combining GMF + MLP towers
- Trained on user–exercise interaction history with negative sampling
- Outputs top-k personalised exercise recommendations with relevance scores

### 3. Nutrition Planner (`nutrition.py`)
- Calculates TDEE using the Mifflin-St Jeor equation
- Adjusts macros for goal (weight loss / muscle gain / maintenance)
- Accounts for comorbidities: diabetes, hypertension, high cholesterol, lactose intolerance
- Generates structured daily meal plans from a curated food database

### 4. RL Agent (`rl_agent.py`)
- **DQN** (Deep Q-Network) with separate policy and target networks
- State: fitness level, fatigue, recent form score, sessions completed, goal progress
- Reward: form quality, improvement, rep completion, fatigue penalty
- Continuously refines exercise choices as the user progresses

---

## Tech Stack

| Module | Technology |
|--------|-----------|
| Pose estimation | MediaPipe, OpenCV |
| Recommendation | PyTorch (NCF) |
| Nutrition | Rule-based + Mifflin-St Jeor |
| RL agent | PyTorch (DQN) |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- MediaPipe 0.10+
- OpenCV 4.8+
