import cv2
import numpy as np

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    active_frames = 0

    left_motion = 0
    right_motion = 0

    total_motion = 0

    prev_gray = None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        # Frame Difference
        frame_diff = cv2.absdiff(prev_gray, gray)

        thresh = cv2.threshold(
            frame_diff,
            25,
            255,
            cv2.THRESH_BINARY
        )[1]

        motion_pixels = np.sum(thresh > 0)

        total_motion += motion_pixels

        # Active Frame Detection
        if motion_pixels > 5000:
            active_frames += 1

        # Left / Right Motion
        left_side = thresh[:, :frame_width // 2]
        right_side = thresh[:, frame_width // 2:]

        left_motion += np.sum(left_side > 0)
        right_motion += np.sum(right_side > 0)

        prev_gray = gray

    cap.release()

    # SAFETY
    if total_frames == 0:
        total_frames = 1

    # REAL DYNAMIC CALCULATIONS

    activity_percent = int(
        (active_frames / total_frames) * 100
    )

    movement_score = min(
        int(total_motion / 100000),
        100
    )

    court_coverage = min(
        int((left_motion + right_motion) / 500000),
        100
    )

    attack_score = min(
        int(right_motion / 300000),
        100
    )

    defense_score = 100 - attack_score

    weak_side = (
        "Left Side"
        if left_motion < right_motion
        else "Right Side"
    )

    # FITNESS
    if movement_score > 75:
        fitness = "Excellent"
    elif movement_score > 45:
        fitness = "Average"
    else:
        fitness = "Needs Improvement"

    # STAMINA
    if activity_percent > 70:
        stamina = "High"
    elif activity_percent > 40:
        stamina = "Moderate"
    else:
        stamina = "Low"

    # FOOTWORK
    if movement_score > 70:
        footwork = "Fast"
    elif movement_score > 40:
        footwork = "Average"
    else:
        footwork = "Slow"

    # GRADE
    if movement_score > 80:
        grade = "A+"
    elif movement_score > 60:
        grade = "A"
    elif movement_score > 40:
        grade = "B"
    else:
        grade = "C"

    # DOMINANT ZONE
    dominant_zone = (
        "Right Court"
        if right_motion > left_motion
        else "Left Court"
    )

    # AI FEEDBACK
    suggestions = []

    if movement_score < 40:
        suggestions.append(
            "Player movement intensity is low."
        )

    if activity_percent < 50:
        suggestions.append(
            "Player activity level decreases during rallies."
        )

    if weak_side == "Left Side":
        suggestions.append(
            "Left side court recovery is weaker."
        )

    if weak_side == "Right Side":
        suggestions.append(
            "Right side movement speed needs improvement."
        )

    if movement_score > 75:
        suggestions.append(
            "Excellent court movement and recovery speed."
        )

    return {

        "movement_score": movement_score,

        "activity_percent": activity_percent,

        "court_coverage": court_coverage,

        "attack_score": attack_score,

        "defense_score": defense_score,

        "fitness": fitness,

        "stamina": stamina,

        "footwork": footwork,

        "grade": grade,

        "weak_side": weak_side,

        "dominant_zone": dominant_zone,

        "left_motion": int(left_motion / 1000),

        "right_motion": int(right_motion / 1000),

        "active_frames": active_frames,

        "total_frames": total_frames,

        "suggestions": suggestions
    }