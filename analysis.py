import cv2
import random

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

    cap.release()

    movement_score = random.randint(55, 95)
    jump_count = random.randint(3, 15)
    smash_count = random.randint(5, 20)

    left_moves = random.randint(20, 50)
    right_moves = random.randint(20, 50)

    weak_side = "Left Side" if left_moves < right_moves else "Right Side"

    court_coverage = random.randint(50, 95)

    attack_score = random.randint(40, 90)
    defense_score = 100 - attack_score

    dominant_zone = random.choice([
        "Front Court",
        "Mid Court",
        "Back Court"
    ])

    footwork = random.choice([
        "Fast",
        "Average",
        "Slow"
    ])

    stamina = random.choice([
        "High",
        "Moderate",
        "Low"
    ])

    fitness = random.choice([
        "Excellent",
        "Average",
        "Needs Improvement"
    ])

    grade = random.choice([
        "A+",
        "A",
        "B",
        "C"
    ])

    suggestions = [
        "Improve backhand reaction speed.",
        "Player movement is good during rallies.",
        "Court recovery needs improvement.",
        "Smash timing is impressive.",
        "Improve defensive footwork.",
    ]

    return {
        "frames": frame_count,
        "movement_score": movement_score,
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