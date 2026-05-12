"""
pose_estimation.py
------------------
Real-time body pose estimation using MediaPipe.
Detects 33 body landmarks, extracts x/y coordinates,
and evaluates exercise form for common movements.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class JointAngles:
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_knee: float = 0.0
    right_knee: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0


@dataclass
class FormFeedback:
    is_correct: bool
    score: float          # 0.0 - 1.0
    feedback: list[str]   # list of cues


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle (degrees) at point b formed by a-b-c."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class PoseEstimator:
    """
    Wraps MediaPipe Pose to track landmarks frame-by-frame.
    Call process_frame() per video frame to get landmarks + angles.
    """

    # MediaPipe landmark indices
    _LM = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
        "left_knee": 25,     "right_knee": 26,
        "left_ankle": 27,    "right_ankle": 28,
    }

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _lm_to_array(self, landmarks, key: str, w: int, h: int) -> np.ndarray:
        lm = landmarks[self._LM[key]]
        return np.array([lm.x * w, lm.y * h])

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[JointAngles]]:
        """
        Args:
            frame: BGR image from OpenCV.
        Returns:
            annotated_frame, JointAngles (or None if no pose detected)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return frame, None

        self.mp_draw.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )

        lms = results.pose_landmarks.landmark
        g = lambda key: self._lm_to_array(lms, key, w, h)

        angles = JointAngles(
            left_elbow=_angle(g("left_shoulder"),  g("left_elbow"),  g("left_wrist")),
            right_elbow=_angle(g("right_shoulder"), g("right_elbow"), g("right_wrist")),
            left_knee=_angle(g("left_hip"),   g("left_knee"),  g("left_ankle")),
            right_knee=_angle(g("right_hip"),  g("right_knee"), g("right_ankle")),
            left_hip=_angle(g("left_shoulder"),  g("left_hip"),  g("left_knee")),
            right_hip=_angle(g("right_shoulder"), g("right_hip"), g("right_knee")),
            left_shoulder=_angle(g("left_elbow"),  g("left_shoulder"),  g("left_hip")),
            right_shoulder=_angle(g("right_elbow"), g("right_shoulder"), g("right_hip")),
        )
        return frame, angles

    def release(self):
        self.pose.close()


class FormAssessor:
    """
    Rule-based form checker for common exercises.
    Returns a FormFeedback with score and coaching cues.
    """

    EXERCISES = ["squat", "pushup", "bicep_curl", "lunge"]

    def assess(self, exercise: str, angles: JointAngles) -> FormFeedback:
        method = getattr(self, f"_check_{exercise}", None)
        if method is None:
            return FormFeedback(is_correct=False, score=0.0,
                                feedback=["Unknown exercise."])
        return method(angles)

    def _check_squat(self, a: JointAngles) -> FormFeedback:
        cues, score = [], 1.0
        # Knees should be ~90° at bottom
        avg_knee = (a.left_knee + a.right_knee) / 2
        if avg_knee > 110:
            cues.append("Go deeper — aim for 90° knee bend.")
            score -= 0.3
        elif avg_knee < 70:
            cues.append("You're going too deep — back off slightly.")
            score -= 0.2
        # Hips should stay open
        avg_hip = (a.left_hip + a.right_hip) / 2
        if avg_hip < 70:
            cues.append("Keep chest up and hips back.")
            score -= 0.2
        if not cues:
            cues.append("Great squat form!")
        return FormFeedback(is_correct=score >= 0.7, score=max(score, 0.0), feedback=cues)

    def _check_pushup(self, a: JointAngles) -> FormFeedback:
        cues, score = [], 1.0
        avg_elbow = (a.left_elbow + a.right_elbow) / 2
        if avg_elbow > 160:
            cues.append("Lower your chest closer to the ground.")
            score -= 0.4
        avg_hip = (a.left_hip + a.right_hip) / 2
        if avg_hip < 160:
            cues.append("Keep your core tight — hips shouldn't sag.")
            score -= 0.3
        if not cues:
            cues.append("Solid push-up form!")
        return FormFeedback(is_correct=score >= 0.7, score=max(score, 0.0), feedback=cues)

    def _check_bicep_curl(self, a: JointAngles) -> FormFeedback:
        cues, score = [], 1.0
        avg_elbow = (a.left_elbow + a.right_elbow) / 2
        if avg_elbow > 160:
            cues.append("Curl the weight up more — full range of motion.")
            score -= 0.3
        avg_shoulder = (a.left_shoulder + a.right_shoulder) / 2
        if avg_shoulder < 160:
            cues.append("Keep elbows pinned to your sides.")
            score -= 0.3
        if not cues:
            cues.append("Perfect curl technique!")
        return FormFeedback(is_correct=score >= 0.7, score=max(score, 0.0), feedback=cues)

    def _check_lunge(self, a: JointAngles) -> FormFeedback:
        cues, score = [], 1.0
        front_knee = min(a.left_knee, a.right_knee)
        if front_knee > 100:
            cues.append("Step further forward — front knee at 90°.")
            score -= 0.3
        back_knee = max(a.left_knee, a.right_knee)
        if back_knee < 80:
            cues.append("Lower your back knee closer to the floor.")
            score -= 0.2
        if not cues:
            cues.append("Excellent lunge depth!")
        return FormFeedback(is_correct=score >= 0.7, score=max(score, 0.0), feedback=cues)


def run_live_demo(exercise: str = "squat"):
    """Run pose estimation + form feedback from webcam."""
    estimator = PoseEstimator()
    assessor = FormAssessor()
    cap = cv2.VideoCapture(0)

    print(f"[PoseEstimator] Starting live demo for: {exercise}. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, angles = estimator.process_frame(frame)

        if angles:
            feedback = assessor.assess(exercise, angles)
            y = 30
            for line in feedback.feedback:
                color = (0, 200, 0) if feedback.is_correct else (0, 80, 255)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, color, 2)
                y += 28
            cv2.putText(frame, f"Form score: {feedback.score:.0%}",
                        (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 200, 0), 2)

        cv2.imshow("AI Fitness Coach — Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    estimator.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo("squat")
