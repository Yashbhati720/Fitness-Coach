"""
main.py
-------
AI Fitness Coach — Main Pipeline
Integrates all four modules:
  1. Pose Estimation + Form Assessment  (pose_estimation.py)
  2. Neural Collaborative Filtering     (recommendation.py)
  3. Personalised Nutrition Planning    (nutrition.py)
  4. Reinforcement Learning Agent       (rl_agent.py)

Run modes:
  python main.py --mode demo       # Run all module demos (no camera required)
  python main.py --mode live       # Launch live webcam pose estimation
  python main.py --mode full       # Full pipeline simulation
"""

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Full pipeline demo (no webcam)
# ---------------------------------------------------------------------------

def run_full_pipeline():
    from recommendation import ExerciseRecommender, UserProfile, EXERCISE_CATALOG
    from nutrition import NutritionPlanner, BodyProfile
    from rl_agent import FitnessRLAgent, UserState, simulate_session, compute_reward

    print_header("AI Fitness Coach — Full Pipeline")

    # ---- Step 1: User setup ------------------------------------------------
    print("\n[1/4] Setting up user profile...")
    user_profile = UserProfile(
        user_id=0,
        age=24,
        fitness_level="beginner",
        goals=["muscle_gain"],
        completed_exercises=[0, 1, 2],
        ratings={0: 4.0, 1: 3.5, 2: 4.5},
    )
    body_profile = BodyProfile(
        weight_kg=70,
        height_cm=172,
        age=24,
        gender="male",
        activity_level="moderately_active",
        goal="muscle_gain",
        body_fat_pct=20.0,
        comorbidities=[],
    )
    print(f"  User: age {user_profile.age}, goal: {user_profile.goals[0]}, "
          f"level: {user_profile.fitness_level}")

    # ---- Step 2: Nutrition plan --------------------------------------------
    print_header("Nutrition Plan")
    planner = NutritionPlanner()
    plan = planner.generate_meal_plan(body_profile)
    print(f"  Daily targets: {plan.macros}")
    for meal in plan.meals:
        print(f"\n  {meal.name}")
        print(f"    {', '.join(meal.foods)}")
        print(f"    {meal.calories} kcal | P:{meal.protein_g}g "
              f"C:{meal.carbs_g}g F:{meal.fat_g}g")
    if plan.notes:
        print("\n  Dietary notes:")
        for n in plan.notes:
            print(f"    - {n}")

    # ---- Step 3: NCF recommendations ---------------------------------------
    print_header("Exercise Recommendations (NCF)")
    num_users = 5
    np.random.seed(7)
    interactions = []
    for u in range(num_users):
        for _ in range(10):
            ex = np.random.randint(0, len(EXERCISE_CATALOG))
            interactions.append((u, ex, float(np.random.choice([3, 4, 5]))))
    # Add user 0's known interactions
    for ex_id, rating in user_profile.ratings.items():
        interactions.append((0, ex_id, rating))

    recommender = ExerciseRecommender(num_users=num_users)
    print("  Training NCF (20 epochs)...")
    recommender.train(interactions, epochs=20)
    recs = recommender.recommend(
        user_id=0, top_k=5,
        exclude_ids=user_profile.completed_exercises,
    )
    print("\n  Top recommendations for this user:")
    for rank, (name, score) in enumerate(recs, 1):
        print(f"    {rank}. {name:22s}  relevance: {score:.3f}")

    # ---- Step 4: RL refinement ---------------------------------------------
    print_header("RL Agent — Adaptive Refinement (100 episodes)")
    agent = FitnessRLAgent(num_exercises=len(EXERCISE_CATALOG))
    state = UserState(
        fitness_level=0.1,
        fatigue_level=0.1,
        recent_form_score=0.5,
        sessions_completed=0.03,
        goal_progress=0.0,
        last_exercise_id=0.0,
    )
    total_reward = 0.0
    for ep in range(1, 101):
        action = agent.select_action(state, exclude_ids=user_profile.completed_exercises)
        session = simulate_session(action, state)
        reward  = compute_reward(session, state.recent_form_score, state.fatigue_level)
        total_reward += reward

        next_state = UserState(
            fitness_level=min(1.0, state.fitness_level + 0.003),
            fatigue_level=float(np.clip(
                state.fatigue_level + session.perceived_effort * 0.1 - 0.05, 0, 1)),
            recent_form_score=0.7 * state.recent_form_score + 0.3 * session.form_score,
            sessions_completed=min(1.0, state.sessions_completed + 0.01),
            goal_progress=min(1.0, state.goal_progress + max(0, reward) * 0.01),
            last_exercise_id=action / len(EXERCISE_CATALOG),
        )
        agent.update(state, action, reward, next_state, done=False)
        state = next_state

        if ep % 25 == 0:
            print(f"  Ep {ep:3d} | Recommended: {EXERCISE_CATALOG[action]:20s} | "
                  f"Form: {session.form_score:.2f} | Reward: {reward:+.2f}")

    print(f"\n  Cumulative reward after 100 episodes: {total_reward:.2f}")
    print(f"  Final fitness level: {state.fitness_level:.3f}")
    print(f"  Goal progress:       {state.goal_progress:.3f}")

    print_header("Pipeline Complete")
    print("  All modules ran successfully.")
    print("  To add real webcam pose estimation: python main.py --mode live")


# ---------------------------------------------------------------------------
# Live webcam mode
# ---------------------------------------------------------------------------

def run_live():
    try:
        from pose_estimation import run_live_demo
        print("Starting live webcam session. Press Q to quit.")
        run_live_demo(exercise="squat")
    except ImportError as e:
        print(f"[Error] Missing dependency for live mode: {e}")
        print("Install: pip install mediapipe opencv-python")


# ---------------------------------------------------------------------------
# Individual module demos
# ---------------------------------------------------------------------------

def run_demos():
    print_header("Running All Module Demos")

    print("\n--- Nutrition Module ---")
    from nutrition import demo as nutrition_demo
    nutrition_demo()

    print("\n--- NCF Recommender ---")
    from recommendation import demo as rec_demo
    rec_demo()

    print("\n--- RL Agent ---")
    from rl_agent import demo as rl_demo
    rl_demo()

    print_header("All demos complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fitness Coach")
    parser.add_argument(
        "--mode",
        choices=[ "live", "full"],
        default="full",
        help="demo: run module demos | live: webcam | full: complete pipeline",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demos()
    elif args.mode == "live":
        run_live()
    else:
        run_full_pipeline()
