import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    movement_score = 0

    left_moves = 0
    right_moves = 0

    jump_count = 0
    smash_count = 0

    prev_x = None
    prev_y = None

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            hip = landmarks[24]
            shoulder = landmarks[12]
            wrist = landmarks[16]

            x = hip.x
            y = hip.y

            if prev_x is not None:

                dx = abs(x - prev_x)
                dy = abs(y - prev_y)

                movement_score += dx + dy

                if x > prev_x:
                    right_moves += 1
                else:
                    left_moves += 1

                # Jump Detection
                if dy > 0.03:
                    jump_count += 1

                # Smash Detection
                if wrist.y < shoulder.y:
                    smash_count += 1

            prev_x = x
            prev_y = y

    cap.release()

    movement_percent = min(int(movement_score * 100), 100)

    # FITNESS
    if movement_percent > 70:
        fitness = "Excellent"
    elif movement_percent > 40:
        fitness = "Average"
    else:
        fitness = "Needs Improvement"

    # FOOTWORK
    if movement_percent > 70:
        footwork = "Fast"
    elif movement_percent > 40:
        footwork = "Average"
    else:
        footwork = "Slow"

    # STAMINA
    if movement_percent > 75:
        stamina = "High"
    elif movement_percent > 45:
        stamina = "Moderate"
    else:
        stamina = "Low"

    # PERFORMANCE GRADE
    if movement_percent > 80:
        grade = "A+"
    elif movement_percent > 60:
        grade = "A"
    elif movement_percent > 40:
        grade = "B"
    else:
        grade = "C"

    weak_side = "Left Side" if left_moves < right_moves else "Right Side"

    court_coverage = min((left_moves + right_moves) // 2, 100)

    attack_score = min(smash_count * 5, 100)

    defense_score = 100 - attack_score

    dominant_zone = "Front Court" if smash_count > 10 else "Mid Court"

    suggestions = []

    if movement_percent < 50:
        suggestions.append(
            "Player movement speed is low. Improve footwork drills."
        )

    if jump_count < 5:
        suggestions.append(
            "Jump smash frequency is low. Improve explosive training."
        )

    if smash_count < 10:
        suggestions.append(
            "Attacking playstyle needs improvement."
        )

    if weak_side == "Left Side":
        suggestions.append(
            "Player reaction on left side is slower than right side."
        )

    if len(suggestions) == 0:
        suggestions.append("Overall performance is impressive.")

    return {
        "frames": frame_count,
        "movement_score": movement_percent,
        "left_moves": left_moves,
        "right_moves": right_moves,
        "jump_count": jump_count,
        "smash_count": smash_count,
        "fitness": fitness,
        "weak_side": weak_side,
        "suggestions": suggestions,
        "footwork": footwork,
        "stamina": stamina,
        "grade": grade,
        "court_coverage": court_coverage,
        "attack_score": attack_score,
        "defense_score": defense_score,
        "dominant_zone": dominant_zone
    }