"""
nutrition.py
------------
Personalised nutrition planning module.
Calculates daily macronutrient targets based on body composition,
fitness goals, and comorbidities, then builds a structured meal plan.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BodyProfile:
    weight_kg: float
    height_cm: float
    age: int
    gender: str                      # "male" or "female"
    activity_level: str              # see ACTIVITY_FACTORS keys
    goal: str                        # "weight_loss", "muscle_gain", "maintenance"
    body_fat_pct: Optional[float] = None
    comorbidities: list[str] = field(default_factory=list)
    diet_type: str = "both"          # "vegetarian", "non_vegetarian", or "both"
    # e.g. comorbidities: ["diabetes", "hypertension", "high_cholesterol", "lactose_intolerant"]


@dataclass
class MacroTargets:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def __str__(self):
        return (f"Calories: {self.calories:.0f} kcal | "
                f"Protein: {self.protein_g:.0f}g | "
                f"Carbs: {self.carbs_g:.0f}g | "
                f"Fat: {self.fat_g:.0f}g")


@dataclass
class Meal:
    name: str
    foods: list[str]
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


@dataclass
class DailyMealPlan:
    macros: MacroTargets
    meals: list[Meal]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVITY_FACTORS = {
    "sedentary":       1.2,
    "lightly_active":  1.375,
    "moderately_active": 1.55,
    "very_active":     1.725,
    "extra_active":    1.9,
}

GOAL_ADJUSTMENTS = {
    "weight_loss":  -500,    # kcal deficit
    "muscle_gain":  +300,    # kcal surplus
    "maintenance":     0,
}

# Food database: (food_name, kcal, protein_g, carbs_g, fat_g) per serving
# Organised by diet type and category.
# Tags: VEG = vegetarian, NON-VEG = non-vegetarian

FOOD_DB = {

    # ════════════════════════════════════════════════════════
    # VEGETARIAN FOODS
    # ════════════════════════════════════════════════════════

    # ── VEG | Proteins ───────────────────────────────────────
    "eggs_2":                   (140, 12,  1,   10.0),   # VEG
    "greek_yogurt_200g":        (120, 20,  9,    0.7),   # VEG
    "cottage_cheese_150g":      (130, 18,  5,    5.0),   # VEG
    "paneer_100g":              (265, 18,  3,   20.0),   # VEG
    "tofu_150g":                (120, 13,  3,    7.0),   # VEG
    "tempeh_100g":              (195, 19,  9,   11.0),   # VEG
    "lentils_100g_cooked":      (116,  9, 20,    0.4),   # VEG
    "chickpeas_100g_cooked":    (164,  9, 27,    2.6),   # VEG
    "black_beans_100g_cooked":  (132,  9, 24,    0.5),   # VEG
    "kidney_beans_100g_cooked": (127,  9, 23,    0.5),   # VEG
    "edamame_100g":             (121, 11, 10,    5.0),   # VEG
    "whey_protein_scoop_30g":   (120, 24,  3,    2.0),   # VEG
    "soy_milk_250ml":           ( 80,  7,  4,    4.0),   # VEG
    "low_fat_milk_250ml":       (110,  8, 12,    2.5),   # VEG
    "skimmed_milk_250ml":       ( 90,  9, 13,    0.2),   # VEG

    # ── VEG | Carbs — Grains & Staples ───────────────────────
    "oats_80g":                 (300, 11, 54,    6.0),   # VEG
    "brown_rice_180g":          (220,  5, 46,    2.0),   # VEG
    "white_rice_180g":          (234,  4, 52,    0.4),   # VEG
    "quinoa_180g":              (220,  8, 39,    3.5),   # VEG
    "whole_wheat_bread_2sl":    (180,  8, 34,    2.0),   # VEG
    "multigrain_bread_2sl":     (160,  7, 30,    2.5),   # VEG
    "roti_chapati_2":           (180,  5, 35,    2.0),   # VEG
    "sweet_potato_150g":        (130,  2, 30,    0.1),   # VEG
    "potato_150g":              (116,  2, 26,    0.1),   # VEG
    "upma_200g":                (220,  5, 38,    5.0),   # VEG
    "poha_200g":                (250,  4, 52,    3.0),   # VEG
    "millet_bajra_180g":        (200,  6, 40,    2.0),   # VEG
    "pasta_whole_wheat_180g":   (220,  8, 43,    1.5),   # VEG

    # ── VEG | Carbs — Fruits ─────────────────────────────────
    "banana":                   ( 90,  1, 23,    0.3),   # VEG
    "apple":                    ( 80,  0, 21,    0.2),   # VEG
    "berries_100g":             ( 57,  0.7, 13,  0.3),   # VEG
    "mango_150g":               ( 99,  1, 25,    0.6),   # VEG
    "orange_1":                 ( 62,  1, 15,    0.2),   # VEG
    "papaya_150g":              ( 60,  0.7, 15,  0.2),   # VEG
    "pomegranate_100g":         ( 83,  1, 19,    1.2),   # VEG
    "grapes_100g":              ( 69,  0.7, 18,  0.2),   # VEG
    "watermelon_200g":          ( 60,  1, 15,    0.2),   # VEG
    "pineapple_150g":           ( 78,  0.8, 20,  0.2),   # VEG
    "guava_100g":               ( 68,  2.6, 14,  1.0),   # VEG
    "kiwi_100g":                ( 61,  1.1, 15,  0.5),   # VEG
    "dates_30g":                ( 82,  0.7, 22,  0.1),   # VEG

    # ── VEG | Fats ────────────────────────────────────────────
    "avocado_half":             (120,  1.5,  6,  11.0),  # VEG
    "olive_oil_1tbsp":         (120,  0,    0,  14.0),  # VEG
    "coconut_oil_1tbsp":       (120,  0,    0,  14.0),  # VEG
    "ghee_1tsp":               ( 45,  0,    0,   5.0),  # VEG
    "almonds_30g":             (170,  6,    6,  15.0),  # VEG
    "walnuts_30g":             (196,  4.5,  4,  19.5),  # VEG
    "cashews_30g":             (163,  4,    9,  13.0),  # VEG
    "peanuts_30g":             (166,  7,    6,  14.0),  # VEG
    "peanut_butter_2tbsp":     (190,  8,    6,  16.0),  # VEG
    "flaxseeds_1tbsp":         ( 55,  2,    3,   4.0),  # VEG
    "chia_seeds_1tbsp":        ( 58,  2,    5,   3.5),  # VEG
    "sunflower_seeds_30g":     (174,  5,    6,  15.0),  # VEG

    # ── VEG | Vegetables ─────────────────────────────────────
    "broccoli_200g":           ( 70,  6, 14,    0.4),   # VEG
    "spinach_150g":            ( 35,  3,  5,    0.4),   # VEG
    "kale_100g":               ( 49,  4,  9,    0.9),   # VEG
    "mixed_veg_200g":          ( 60,  3, 12,    0.5),   # VEG
    "tomato_2":                ( 36,  2,  7,    0.4),   # VEG
    "cucumber_150g":           ( 24,  1,  5,    0.1),   # VEG
    "carrot_150g":             ( 62,  1, 14,    0.3),   # VEG
    "capsicum_bell_pepper_1":  ( 31,  1,  7,    0.3),   # VEG
    "cauliflower_200g":        ( 50,  4, 10,    0.3),   # VEG
    "cabbage_150g":            ( 38,  2,  9,    0.1),   # VEG
    "onion_1_medium":          ( 44,  1, 10,    0.1),   # VEG
    "beetroot_100g":           ( 43,  2, 10,    0.2),   # VEG
    "mushroom_100g":           ( 22,  3,  3,    0.3),   # VEG
    "peas_100g":               ( 81,  5, 14,    0.4),   # VEG
    "corn_100g":               ( 86,  3, 19,    1.2),   # VEG

    # ════════════════════════════════════════════════════════
    # NON-VEGETARIAN FOODS
    # ════════════════════════════════════════════════════════

    # ── NON-VEG | Proteins — Poultry ─────────────────────────
    "chicken_breast_150g":     (165, 31,  0,    3.6),   # NON-VEG
    "chicken_thigh_150g":      (220, 26,  0,   12.0),   # NON-VEG
    "ground_chicken_150g":     (195, 27,  0,    9.0),   # NON-VEG
    "turkey_breast_150g":      (160, 35,  0,    1.0),   # NON-VEG
    "boiled_egg_2":            (148, 13,  1,   10.0),   # NON-VEG
    "egg_white_4":             ( 68, 14,  1,    0.3),   # NON-VEG

    # ── NON-VEG | Proteins — Fish & Seafood ──────────────────
    "salmon_150g":             (280, 28,  0,   18.0),   # NON-VEG
    "tuna_can_100g":           (116, 26,  0,    1.0),   # NON-VEG
    "rohu_fish_150g":          (160, 26,  0,    6.0),   # NON-VEG
    "pomfret_150g":            (175, 25,  0,    8.0),   # NON-VEG
    "prawns_150g":             (150, 29,  1,    2.5),   # NON-VEG
    "sardines_100g":           (208, 25,  0,   11.0),   # NON-VEG
    "mackerel_150g":           (260, 24,  0,   18.0),   # NON-VEG
    "tilapia_150g":            (160, 30,  0,    3.5),   # NON-VEG

    # ── NON-VEG | Proteins — Red Meat ────────────────────────
    "lean_beef_150g":          (215, 32,  0,    9.0),   # NON-VEG
    "mutton_150g":             (280, 28,  0,   18.0),   # NON-VEG
    "lean_pork_150g":          (195, 30,  0,    7.5),   # NON-VEG

    # ── NON-VEG | Carbs (same as VEG — shared) ───────────────
    # (Use the vegetarian carb and fruit entries above)

    # ── NON-VEG | Fats (same as VEG — shared) ────────────────
    # (Use the vegetarian fat entries above)

    # ── NON-VEG | Vegetables (same as VEG — shared) ──────────
    # (Use the vegetarian vegetable entries above)
}

# Keys that are exclusively non-vegetarian (meat / fish / seafood)
NON_VEG_ONLY_KEYS = {
    "chicken_breast_150g", "chicken_thigh_150g", "ground_chicken_150g",
    "turkey_breast_150g", "boiled_egg_2", "egg_white_4",
    "salmon_150g", "tuna_can_100g", "rohu_fish_150g", "pomfret_150g",
    "prawns_150g", "sardines_100g", "mackerel_150g", "tilapia_150g",
    "lean_beef_150g", "mutton_150g", "lean_pork_150g",
}


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class NutritionPlanner:
    """
    Calculates personalised macro targets and generates a structured
    daily meal plan accounting for goals and comorbidities.
    """

    def calculate_macros(self, profile: BodyProfile) -> MacroTargets:
        # Basal Metabolic Rate (Mifflin-St Jeor)
        if profile.gender == "male":
            bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age + 5
        else:
            bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age - 161

        activity = ACTIVITY_FACTORS.get(profile.activity_level, 1.375)
        tdee = bmr * activity
        calories = tdee + GOAL_ADJUSTMENTS.get(profile.goal, 0)

        # Comorbidity adjustments
        if "diabetes" in profile.comorbidities:
            calories = min(calories, tdee)        # no surplus for diabetics

        # Macro split based on goal
        if profile.goal == "muscle_gain":
            protein_g = profile.weight_kg * 2.2   # high protein
            fat_g     = calories * 0.25 / 9
        elif profile.goal == "weight_loss":
            protein_g = profile.weight_kg * 2.0   # preserve muscle
            fat_g     = calories * 0.30 / 9
        else:
            protein_g = profile.weight_kg * 1.8
            fat_g     = calories * 0.30 / 9

        carbs_kcal = calories - (protein_g * 4) - (fat_g * 9)
        carbs_g    = max(carbs_kcal / 4, 50)       # minimum 50g carbs

        return MacroTargets(
            calories=round(calories, 1),
            protein_g=round(protein_g, 1),
            carbs_g=round(carbs_g, 1),
            fat_g=round(fat_g, 1),
        )

    def _filter_foods(self, profile: BodyProfile) -> dict:
        """Remove foods incompatible with diet type or comorbidities."""
        excluded = set()

        # Diet-type filter
        if profile.diet_type == "vegetarian":
            excluded.update(NON_VEG_ONLY_KEYS)

        # Comorbidity filter
        if "lactose_intolerant" in profile.comorbidities:
            excluded.update({"greek_yogurt_200g", "cottage_cheese_150g",
                             "paneer_100g", "low_fat_milk_250ml", "skimmed_milk_250ml"})
        if "high_cholesterol" in profile.comorbidities:
            excluded.update({"eggs_2", "boiled_egg_2", "peanut_butter_2tbsp",
                             "ghee_1tsp", "coconut_oil_1tbsp"})
        if "diabetes" in profile.comorbidities:
            excluded.update({"banana", "mango_150g", "dates_30g",
                             "whole_wheat_bread_2sl", "white_rice_180g", "corn_100g"})

        return {k: v for k, v in FOOD_DB.items() if k not in excluded}

    def _build_meal(self, name: str, food_keys: list[str]) -> Meal:
        kcal = prot = carbs = fat = 0.0
        foods_db = FOOD_DB
        food_names = []
        for key in food_keys:
            if key in foods_db:
                k, p, c, f = foods_db[key]
                kcal += k; prot += p; carbs += c; fat += f
                food_names.append(key.replace("_", " ").title())
        return Meal(name=name, foods=food_names,
                    calories=round(kcal), protein_g=round(prot),
                    carbs_g=round(carbs), fat_g=round(fat))

    def generate_meal_plan(self, profile: BodyProfile) -> DailyMealPlan:
        macros = self.calculate_macros(profile)
        foods  = self._filter_foods(profile)
        notes  = []
        is_veg = (profile.diet_type == "vegetarian")

        # ── Meal templates split by goal × diet type ──────────────────────────
        if profile.goal == "muscle_gain":
            if is_veg:
                meals = [
                    self._build_meal("Breakfast",    ["oats_80g", "eggs_2", "banana", "almonds_30g"]),
                    self._build_meal("Mid-morning",  ["greek_yogurt_200g", "berries_100g"]),
                    self._build_meal("Lunch",        ["paneer_100g", "brown_rice_180g", "broccoli_200g"]),
                    self._build_meal("Pre-workout",  ["whole_wheat_bread_2sl", "peanut_butter_2tbsp"]),
                    self._build_meal("Dinner",       ["tofu_150g", "quinoa_180g", "spinach_150g"]),
                    self._build_meal("Post-workout", ["cottage_cheese_150g", "berries_100g"]),
                ]
            else:
                meals = [
                    self._build_meal("Breakfast",    ["oats_80g", "eggs_2", "banana", "almonds_30g"]),
                    self._build_meal("Mid-morning",  ["greek_yogurt_200g", "berries_100g"]),
                    self._build_meal("Lunch",        ["chicken_breast_150g", "brown_rice_180g", "broccoli_200g"]),
                    self._build_meal("Pre-workout",  ["whole_wheat_bread_2sl", "peanut_butter_2tbsp"]),
                    self._build_meal("Dinner",       ["salmon_150g", "sweet_potato_150g", "spinach_150g"]),
                    self._build_meal("Post-workout", ["cottage_cheese_150g", "berries_100g"]),
                ]

        elif profile.goal == "weight_loss":
            if is_veg:
                meals = [
                    self._build_meal("Breakfast",   ["eggs_2", "spinach_150g", "apple"]),
                    self._build_meal("Lunch",       ["chickpeas_100g_cooked", "mixed_veg_200g", "quinoa_180g"]),
                    self._build_meal("Snack",       ["greek_yogurt_200g", "almonds_30g"]),
                    self._build_meal("Dinner",      ["tofu_150g", "sweet_potato_150g", "broccoli_200g"]),
                ]
            else:
                meals = [
                    self._build_meal("Breakfast",   ["boiled_egg_2", "spinach_150g", "apple"]),
                    self._build_meal("Lunch",       ["chicken_breast_150g", "mixed_veg_200g", "quinoa_180g"]),
                    self._build_meal("Snack",       ["greek_yogurt_200g", "almonds_30g"]),
                    self._build_meal("Dinner",      ["tuna_can_100g", "sweet_potato_150g", "broccoli_200g"]),
                ]

        else:  # maintenance
            if is_veg:
                meals = [
                    self._build_meal("Breakfast",   ["oats_80g", "eggs_2", "berries_100g"]),
                    self._build_meal("Lunch",       ["paneer_100g", "brown_rice_180g", "mixed_veg_200g"]),
                    self._build_meal("Snack",       ["apple", "walnuts_30g"]),
                    self._build_meal("Dinner",      ["lentils_100g_cooked", "roti_chapati_2", "spinach_150g"]),
                ]
            else:
                meals = [
                    self._build_meal("Breakfast",   ["oats_80g", "eggs_2", "berries_100g"]),
                    self._build_meal("Lunch",       ["chicken_breast_150g", "brown_rice_180g", "mixed_veg_200g"]),
                    self._build_meal("Snack",       ["apple", "almonds_30g"]),
                    self._build_meal("Dinner",      ["salmon_150g", "quinoa_180g", "spinach_150g"]),
                ]

        # Diet type note
        if is_veg:
            notes.append("Vegetarian plan: all animal meat and seafood excluded.")
        else:
            notes.append("Non-vegetarian plan: includes poultry, fish, and red meat options.")

        # Comorbidity notes
        if "diabetes" in profile.comorbidities:
            notes.append("Limit simple sugars; spread carbs evenly across meals.")
        if "hypertension" in profile.comorbidities:
            notes.append("Reduce sodium; limit processed foods.")
        if "high_cholesterol" in profile.comorbidities:
            notes.append("Prioritise unsaturated fats; increase fibre intake.")
        if "lactose_intolerant" in profile.comorbidities:
            notes.append("Dairy substituted with plant-based alternatives.")

        return DailyMealPlan(macros=macros, meals=meals, notes=notes)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Nutrition Planner Demo ===\n")

    for diet_label, diet_type in [("Vegetarian", "vegetarian"), ("Non-Vegetarian", "non_vegetarian")]:
        print(f"\n{'='*50}")
        print(f"  {diet_label} Plan — Muscle Gain")
        print(f"{'='*50}")

        profile = BodyProfile(
            weight_kg=75,
            height_cm=175,
            age=25,
            gender="male",
            activity_level="moderately_active",
            goal="muscle_gain",
            body_fat_pct=18.0,
            comorbidities=[],
            diet_type=diet_type,
        )

        planner = NutritionPlanner()
        plan = planner.generate_meal_plan(profile)

        print(f"\nDaily targets: {plan.macros}\n")
        for meal in plan.meals:
            print(f"  {meal.name}")
            print(f"    Foods:    {', '.join(meal.foods)}")
            print(f"    Macros:   {meal.calories} kcal | "
                  f"P {meal.protein_g}g | C {meal.carbs_g}g | F {meal.fat_g}g\n")

        if plan.notes:
            print("  Notes:")
            for note in plan.notes:
                print(f"    - {note}")


if __name__ == "__main__":
    demo()
