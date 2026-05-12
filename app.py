"""
app.py
------
Flask web UI for the AI Fitness Coach.
Serves a dashboard with all 5 features:
  - Live pose estimation (webcam via browser)
  - Exercise recommendations
  - Nutrition meal plan
  - RL progress tracking
  - User profile / settings

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, jsonify, request, Response, session, redirect, url_for
import cv2
import threading
import json
import numpy as np
import time
import base64
from dataclasses import asdict

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# ── Try importing project modules ──────────────────────────────────────────
try:
    from pose_estimation import PoseEstimator, FormAssessor
    from recommendation import ExerciseRecommender, EXERCISE_CATALOG
    from nutrition import NutritionPlanner, BodyProfile
    from rl_agent import FitnessRLAgent, UserState, simulate_session, compute_reward
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False

# ── Global state ───────────────────────────────────────────────────────────
camera_lock = threading.Lock()
current_feedback = {"score": 0, "cues": [], "exercise": "squat"}

USER = {
    "name": "Yash Bhati",
    "age": 24,
    "weight": 70,
    "height": 172,
    "gender": "male",
    "activity": "moderately_active",
    "goal": "muscle_gain",
    "fitness_level": "beginner",
    "comorbidities": [],
    "bmi": None,
    "diet_preference": "non_veg",
}

rl_history = []  # list of {episode, reward, form, fitness}

IDEAL_POSE_ANGLES = {
    "squat": {"left_knee": 90, "right_knee": 90, "left_hip": 80, "right_hip": 80},
    "pushup": {"left_elbow": 90, "right_elbow": 90, "left_hip": 170, "right_hip": 170},
    "bicep_curl": {"left_elbow": 55, "right_elbow": 55, "left_shoulder": 170, "right_shoulder": 170},
    "lunge": {"left_knee": 90, "right_knee": 90, "left_hip": 100, "right_hip": 100},
}


def _demo_feedback_for_exercise(exercise):
    """Fallback feedback when AI modules are unavailable."""
    feedback_map = {
        "squat": ["Keep chest up", "Push knees out", "Drive through heels"],
        "pushup": ["Keep body in straight line", "Elbows at ~45 degrees", "Engage core"],
        "bicep_curl": ["Keep elbows close to torso", "Avoid swinging", "Control lowering phase"],
        "lunge": ["Step long enough for 90 degree knees", "Keep torso upright", "Push through front heel"],
    }
    return {
        "score": 78,
        "cues": feedback_map.get(exercise, ["Good effort", "Maintain steady form"]),
        "exercise": exercise,
    }


def _bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def _build_personalized_plan(bmi, pose_score, exercise):
    bmi_group = _bmi_category(bmi)

    if bmi_group == "underweight":
        diet_focus = [
            "Eat in a mild calorie surplus (+250 to +350 kcal/day).",
            "Protein target: 1.8-2.2 g/kg bodyweight with 3-4 meals.",
            "Add calorie-dense whole foods: nuts, peanut butter, milk, rice, eggs.",
        ]
        exercise_focus = [
            "Prioritize strength training 4 days/week with progressive overload.",
            "Recommended exercises: squat, push-up, bicep curl, row variations.",
            "Keep cardio short (10-15 min) after strength sessions.",
        ]
    elif bmi_group == "normal":
        diet_focus = [
            "Maintain calories near maintenance and adjust by goal weekly.",
            "Protein target: 1.6-2.0 g/kg with balanced carbs around workouts.",
            "Use mostly whole foods and hydrate well through the day.",
        ]
        exercise_focus = [
            "Mix strength + conditioning (3-5 training days/week).",
            f"Current focus exercise: {exercise.replace('_', ' ').title()}.",
            "Add mobility and core stability work 2-3 times/week.",
        ]
    elif bmi_group == "overweight":
        diet_focus = [
            "Use a moderate calorie deficit (-300 to -450 kcal/day).",
            "Protein target: 1.8-2.2 g/kg goal bodyweight to preserve muscle.",
            "Increase fiber-rich foods: vegetables, fruits, legumes, whole grains.",
        ]
        exercise_focus = [
            "Train strength 3-4 days/week plus low-impact cardio 4-5 days/week.",
            "Recommended exercises: squat variations, incline push-up, lunges, brisk walking.",
            "Target 8k-10k steps/day and increase gradually.",
        ]
    else:
        diet_focus = [
            "Start with a sustainable calorie deficit (-400 to -600 kcal/day).",
            "Protein target: 2.0 g/kg goal bodyweight to protect lean mass.",
            "Prioritize simple meal structure and avoid sugary liquid calories.",
        ]
        exercise_focus = [
            "Begin with low-impact training: walking, cycling, assisted bodyweight moves.",
            "Strength train 2-3 days/week focusing on full-body movements.",
            "Progress volume slowly and emphasize consistency over intensity.",
        ]

    # Adjust guidance using latest photo analysis quality.
    if pose_score < 40:
        exercise_focus.append("Form score is low: reduce load and practice technique first.")
    elif pose_score < 70:
        exercise_focus.append("Form score is moderate: keep moderate load and improve control.")
    else:
        exercise_focus.append("Form score is strong: you can gradually increase intensity.")

    return {
        "bmi": round(bmi, 2),
        "bmi_category": bmi_group,
        "pose_score": int(pose_score),
        "exercise": exercise,
        "diet_recommendations": diet_focus,
        "exercise_recommendations": exercise_focus,
    }


def _answer_physique_question(question, user, feedback):
    q = (question or "").strip().lower()
    if not q:
        return "Please ask a specific question about your physique or fitness."

    bmi = user.get("bmi")
    if bmi is None:
        h_m = max(0.1, float(user.get("height", 170)) / 100.0)
        w_kg = max(1.0, float(user.get("weight", 70)))
        bmi = round(w_kg / (h_m * h_m), 2)
    bmi_category = _bmi_category(float(bmi))
    form_score = int(feedback.get("score", 0))

    if "bmi" in q or "weight" in q or "fat" in q:
        return (
            f"Your BMI is {bmi} ({bmi_category}). "
            "Use this as a trend metric, not the only health metric. "
            "Focus on consistent training, protein intake, and sleep."
        )

    if "muscle" in q or "gain" in q:
        return (
            f"Given your current form score ({form_score}%), prioritize form first, then progressive overload. "
            "Aim for 1.6-2.2 g/kg protein and train each muscle group 2x/week."
        )

    if "lose" in q or "cut" in q:
        return (
            "Use a moderate calorie deficit, keep protein high, and continue strength training "
            "to preserve muscle while losing fat."
        )

    if "posture" in q or "form" in q:
        cues = feedback.get("cues") or ["Keep a neutral spine", "Control your movement tempo"]
        return f"Your latest form score is {form_score}%. Focus on: {', '.join(cues[:3])}."

    return (
        f"Based on your profile and latest form score ({form_score}%), "
        "a balanced plan with progressive strength work, recovery, and diet consistency is best. "
        "Ask about BMI, muscle gain, fat loss, or posture for more specific guidance."
    )


def _apply_diet_preference_to_meals(meals, preference):
    """Force meal foods to match selected diet preference."""
    nonveg_to_veg = {
        "chicken": "paneer",
        "fish": "tofu",
        "salmon": "tofu",
        "egg": "tofu scramble",
        "eggs": "tofu scramble",
        "mutton": "soy chunks",
        "beef": "kidney beans",
    }
    veg_to_nonveg = {
        "paneer": "chicken breast",
        "tofu": "fish",
        "soy chunks": "chicken breast",
        "kidney beans": "lean meat",
        "chickpeas": "eggs",
    }

    mapped = []
    for meal in meals:
        foods = []
        for food in meal.get("foods", []):
            f = str(food)
            f_low = f.lower()
            if preference == "veg":
                for k, v in nonveg_to_veg.items():
                    if k in f_low:
                        f = f.replace(food, f_low.replace(k, v).title())
                        break
            else:
                for k, v in veg_to_nonveg.items():
                    if k in f_low:
                        f = f.replace(food, f_low.replace(k, v).title())
                        break
            foods.append(f)
        meal_copy = dict(meal)
        meal_copy["foods"] = foods
        mapped.append(meal_copy)
    return mapped


def _estimate_body_fat(user, feedback):
    """
    Experimental body-fat estimate using profile metrics + latest image/form signal.
    This is a fitness heuristic and not a medical measurement.
    """
    height_m = max(0.1, float(user.get("height", 170)) / 100.0)
    weight_kg = max(1.0, float(user.get("weight", 70)))
    bmi = float(user.get("bmi") or round(weight_kg / (height_m * height_m), 2))
    age = float(user.get("age", 24))
    gender = str(user.get("gender", "male")).lower()
    sex_factor = 1 if gender == "male" else 0

    # Deurenberg equation baseline estimate
    baseline = 1.2 * bmi + 0.23 * age - 10.8 * sex_factor - 5.4

    pose_score = float(feedback.get("score", 0))
    cues = feedback.get("cues") or []
    # Small adjustment based on latest form confidence from image processing
    image_adjustment = ((70 - pose_score) / 70.0) * 1.2 + (min(len(cues), 6) * 0.15)

    estimated = max(3.0, min(55.0, baseline + image_adjustment))
    category = "athletic" if estimated < 14 else "fit" if estimated < 21 else "average" if estimated < 28 else "high"
    return {
        "body_fat_percent": round(estimated, 1),
        "bmi_used": round(bmi, 2),
        "pose_score_used": int(pose_score),
        "category": category,
        "disclaimer": "Experimental estimate only. Use DEXA, BIA, or skinfold tests for better accuracy.",
    }


def _pose_difference_report(exercise, angle_map):
    target = IDEAL_POSE_ANGLES.get(exercise, {})
    if not target:
        return []
    report = []
    for joint, ideal in target.items():
        actual = float(angle_map.get(joint, 0.0))
        diff = actual - ideal
        if abs(diff) <= 10:
            status = "good"
        elif diff > 0:
            status = "too_open"
        else:
            status = "too_closed"
        report.append({
            "joint": joint.replace("_", " "),
            "actual": round(actual, 1),
            "ideal": round(float(ideal), 1),
            "difference": round(diff, 1),
            "status": status,
        })
    return report

# ── Video streaming ────────────────────────────────────────────────────────
def gen_frames(exercise="squat"):
    estimator = PoseEstimator() if MODULES_LOADED else None
    assessor  = FormAssessor()  if MODULES_LOADED else None
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MODULES_LOADED and estimator:
            frame, angles = estimator.process_frame(frame)
            if angles:
                fb = assessor.assess(exercise, angles)
                current_feedback["score"] = round(fb.score * 100)
                current_feedback["cues"]  = fb.feedback
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

    cap.release()
    if estimator:
        estimator.release()


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin":  # Simple hardcoded credentials
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/")
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", user=USER)


@app.route("/video_feed")
def video_feed():
    exercise = request.args.get("exercise", "squat")
    return Response(gen_frames(exercise),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/feedback")
def api_feedback():
    return jsonify(current_feedback)


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    data = request.get_json(silent=True) or {}
    frame_data_url = data.get("frame", "")
    exercise = data.get("exercise", "squat")

    if not frame_data_url:
        return jsonify({"error": "Missing frame data."}), 400

    # Expected format: data:image/jpeg;base64,<...>
    try:
        encoded = frame_data_url.split(",", 1)[1]
        raw_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Invalid image payload."}), 400

    if frame is None:
        return jsonify({"error": "Could not decode image."}), 400

    if MODULES_LOADED:
        try:
            estimator = PoseEstimator()
            assessor = FormAssessor()
            _, angles = estimator.process_frame(frame)
            if angles:
                fb = assessor.assess(exercise, angles)
                angle_map = asdict(angles)
                result = {
                    "score": int(round(float(fb.score) * 100)),
                    "cues": list(fb.feedback),
                    "exercise": exercise,
                    "pose_comparison": _pose_difference_report(exercise, angle_map),
                }
            else:
                result = {
                    "score": 0,
                    "cues": ["No full-body pose detected. Stand back and try again."],
                    "exercise": exercise,
                    "pose_comparison": [],
                }
            estimator.release()
            current_feedback.update(result)
            return jsonify(result)
        except Exception:
            # If processing fails for any reason, gracefully fall back.
            pass

    result = _demo_feedback_for_exercise(exercise)
    result["pose_comparison"] = []
    current_feedback.update(result)
    return jsonify(result)


@app.route("/api/personalized_plan", methods=["POST"])
def api_personalized_plan():
    data = request.get_json(silent=True) or {}
    bmi_value = data.get("bmi")

    if bmi_value in (None, ""):
        height_m = max(0.1, float(USER.get("height", 170)) / 100.0)
        weight_kg = max(1.0, float(USER.get("weight", 70)))
        bmi = weight_kg / (height_m * height_m)
    else:
        try:
            bmi = float(bmi_value)
        except (TypeError, ValueError):
            return jsonify({"error": "BMI must be a valid number."}), 400

    if bmi <= 0 or bmi > 80:
        return jsonify({"error": "BMI value is out of expected range."}), 400

    USER["bmi"] = round(bmi, 2)
    pose_score = float(current_feedback.get("score", 0))
    exercise = current_feedback.get("exercise", "squat")
    plan = _build_personalized_plan(bmi, pose_score, exercise)

    if USER.get("diet_preference") == "veg":
        plan["diet_recommendations"].append(
            "Veg mode: prioritize paneer, tofu, soy chunks, lentils, beans, curd, milk for protein."
        )
    else:
        plan["diet_recommendations"].append(
            "Non-veg mode: include eggs, chicken, fish with vegetables and whole grains."
        )

    return jsonify(plan)


@app.route("/api/ask_physique", methods=["POST"])
def api_ask_physique():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    answer = _answer_physique_question(question, USER, current_feedback)
    return jsonify({"answer": answer})


@app.route("/api/body_fat_estimate", methods=["POST"])
def api_body_fat_estimate():
    data = request.get_json(silent=True) or {}
    if "age" in data and data.get("age") not in (None, ""):
        USER["age"] = int(data["age"])
    if "gender" in data and data.get("gender") in ("male", "female"):
        USER["gender"] = data["gender"]
    if "weight" in data and data.get("weight") not in (None, ""):
        USER["weight"] = float(data["weight"])
    if "height" in data and data.get("height") not in (None, ""):
        USER["height"] = float(data["height"])
    if "bmi" in data and data.get("bmi") not in (None, ""):
        USER["bmi"] = float(data["bmi"])

    result = _estimate_body_fat(USER, current_feedback)
    return jsonify(result)


@app.route("/api/recommendations")
def api_recommendations():
    if not MODULES_LOADED:
        return jsonify([
            {"name": "Squat",          "score": 0.92},
            {"name": "Push-up",        "score": 0.87},
            {"name": "Plank",          "score": 0.83},
            {"name": "Bicep curl",     "score": 0.79},
            {"name": "Mountain climber","score": 0.74},
        ])
    np.random.seed(42)
    rec = ExerciseRecommender(num_users=5)
    interactions = [(u, np.random.randint(0,15), float(np.random.choice([3,4,5])))
                    for u in range(5) for _ in range(10)]
    rec.train(interactions, epochs=15)
    recs = rec.recommend(user_id=0, top_k=5)
    return jsonify([{"name": n, "score": round(s, 3)} for n, s in recs])

@app.route("/api/nutrition")
def api_nutrition():
    selected_pref = request.args.get("diet_preference")
    if selected_pref in ("veg", "non_veg"):
        USER["diet_preference"] = selected_pref

    if not MODULES_LOADED:
        pref = USER.get("diet_preference", "non_veg")
        if pref == "veg":
            meals = [
                {"name": "Breakfast", "foods": ["Oats 80G", "Milk", "Banana", "Almonds 30G"], "calories": 680, "protein_g": 24, "carbs_g": 92, "fat_g": 24},
                {"name": "Lunch", "foods": ["Paneer 120G", "Brown Rice 180G", "Broccoli"], "calories": 790, "protein_g": 39, "carbs_g": 95, "fat_g": 24},
                {"name": "Snack", "foods": ["Greek Yogurt 200G", "Berries 100G"], "calories": 180, "protein_g": 21, "carbs_g": 22, "fat_g": 1},
                {"name": "Dinner", "foods": ["Tofu 160G", "Sweet Potato", "Spinach Salad"], "calories": 700, "protein_g": 36, "carbs_g": 68, "fat_g": 26},
            ]
        else:
            meals = [
                {"name": "Breakfast", "foods": ["Oats 80G", "Eggs 2", "Banana", "Almonds 30G"], "calories": 700, "protein_g": 30, "carbs_g": 90, "fat_g": 27},
                {"name": "Lunch", "foods": ["Chicken Breast 150G", "Brown Rice 180G", "Broccoli"], "calories": 820, "protein_g": 55, "carbs_g": 95, "fat_g": 18},
                {"name": "Snack", "foods": ["Greek Yogurt 200G", "Berries 100G"], "calories": 180, "protein_g": 21, "carbs_g": 22, "fat_g": 1},
                {"name": "Dinner", "foods": ["Salmon 150G", "Sweet Potato", "Spinach Salad"], "calories": 750, "protein_g": 48, "carbs_g": 65, "fat_g": 28},
            ]
        return jsonify({
            "macros": {"calories": 2800, "protein_g": 154, "carbs_g": 320, "fat_g": 87},
            "meals": meals,
            "notes": [f"Diet preference: {'Vegetarian' if pref == 'veg' else 'Non-vegetarian'}"],
            "diet_preference": pref,
        })
    profile = BodyProfile(
        weight_kg=USER["weight"], height_cm=USER["height"],
        age=USER["age"], gender=USER["gender"],
        activity_level=USER["activity"], goal=USER["goal"],
        comorbidities=USER["comorbidities"],
    )
    planner = NutritionPlanner()
    plan = planner.generate_meal_plan(profile)
    payload = {
        "macros": {
            "calories": plan.macros.calories,
            "protein_g": plan.macros.protein_g,
            "carbs_g": plan.macros.carbs_g,
            "fat_g": plan.macros.fat_g,
        },
        "meals": [{"name": m.name, "foods": m.foods,
                   "calories": m.calories, "protein_g": m.protein_g,
                   "carbs_g": m.carbs_g, "fat_g": m.fat_g}
                  for m in plan.meals],
        "notes": plan.notes,
    }
    pref = USER.get("diet_preference", "non_veg")
    payload["meals"] = _apply_diet_preference_to_meals(payload["meals"], pref)
    payload["notes"] = payload["notes"] + [
        f"Diet preference: {'Vegetarian' if pref == 'veg' else 'Non-vegetarian'}"
    ]
    payload["diet_preference"] = pref
    return jsonify(payload)


@app.route("/api/rl_progress")
def api_rl_progress():
    if not rl_history:
        # Generate demo data
        data, fitness, form = [], 0.1, 0.5
        for ep in range(1, 51):
            r = float(np.random.normal(0.6, 0.4))
            fitness = min(1.0, fitness + 0.008)
            form    = min(1.0, 0.7 * form + 0.3 * np.clip(form + np.random.normal(0.05,0.1), 0, 1))
            data.append({"episode": ep, "reward": round(r, 3),
                         "form": round(form, 3), "fitness": round(fitness, 3)})
        return jsonify(data)
    return jsonify(rl_history)


@app.route("/api/update_profile", methods=["POST"])
def update_profile():
    data = request.get_json(silent=True) or {}
    USER.update({k: v for k, v in data.items() if k in USER})
    return jsonify({"status": "ok", "user": USER})


@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
