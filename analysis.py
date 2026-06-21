import cv2
import numpy as np

def validate_badminton_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps
    if duration < 2:
        cap.release()
        return False, "Video too short (minimum 2 seconds required)"
    court_score = 0
    checked = 0
    sample_points = [int(total_frames * x) for x in [0.1, 0.3, 0.5, 0.7, 0.9]]
    for pos in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        checked += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,40,255]))
        white_pixels = np.sum(white_mask > 0)
        green_mask = cv2.inRange(hsv, np.array([35,30,30]), np.array([85,255,255]))
        green_pixels = np.sum(green_mask > 0)
        blue_mask = cv2.inRange(hsv, np.array([90,50,50]), np.array([130,255,255]))
        blue_pixels = np.sum(blue_mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        has_lines = white_pixels > (total_pixels * 0.01)
        has_court_color = (green_pixels + blue_pixels) > (total_pixels * 0.05)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=60, maxLineGap=15)
        has_parallel_lines = lines is not None and len(lines) > 8
        if (has_lines and has_court_color) or (has_lines and has_parallel_lines):
            court_score += 1
    cap.release()
    if checked == 0:
        return False, "Could not read video frames"
    confidence = court_score / checked
    if confidence >= 0.3:
        return True, "Valid badminton video"
    else:
        return False, "This doesn't appear to be a badminton court video. Please upload a video showing a badminton court with players."


def detect_player_count(video_path):
    cap = cv2.VideoCapture(video_path)
    player_counts = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_points = [int(total_frames * x) for x in [0.2, 0.4, 0.6, 0.8]]
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(min(30, total_frames)):
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, (320, 240))
            bg_subtractor.apply(small)
    for pos in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        small = cv2.resize(frame, (320, 240))
        fg_mask = bg_subtractor.apply(small)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        player_blobs = [c for c in contours if 500 < cv2.contourArea(c) < 15000]
        player_counts.append(len(player_blobs))
    cap.release()
    if not player_counts:
        return "Single", 1
    avg_players = np.mean(player_counts)
    if avg_players >= 2.5:
        return "Doubles", int(round(avg_players))
    else:
        return "Singles", max(1, int(round(avg_players)))


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    active_frames = 0
    left_motion = 0
    right_motion = 0
    total_motion = 0
    prev_gray = None
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        frame_diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pixels = np.sum(thresh > 0)
        total_motion += motion_pixels
        if motion_pixels > 5000:
            active_frames += 1
        left_side = thresh[:, :frame_width // 2]
        right_side = thresh[:, frame_width // 2:]
        left_motion += np.sum(left_side > 0)
        right_motion += np.sum(right_side > 0)
        prev_gray = gray

    cap.release()

    if total_frames == 0:
        total_frames = 1

    activity_percent = int((active_frames / total_frames) * 100)
    movement_score = min(int(total_motion / 100000), 100)
    court_coverage = min(int((left_motion + right_motion) / 500000), 100)
    attack_score = min(int(right_motion / 300000), 100)
    defense_score = 100 - attack_score
    weak_side = "Left Side" if left_motion < right_motion else "Right Side"
    dominant_zone = "Right Court" if right_motion > left_motion else "Left Court"

    if movement_score > 75: fitness = "Excellent"
    elif movement_score > 45: fitness = "Average"
    else: fitness = "Needs Improvement"

    if activity_percent > 70: stamina = "High"
    elif activity_percent > 40: stamina = "Moderate"
    else: stamina = "Low"

    if movement_score > 70: footwork = "Fast"
    elif movement_score > 40: footwork = "Average"
    else: footwork = "Slow"

    if movement_score > 80: grade = "A+"
    elif movement_score > 60: grade = "A"
    elif movement_score > 40: grade = "B"
    else: grade = "C"

    suggestions = []
    if movement_score < 40:
        suggestions.append("Player movement intensity is low. Increase footwork drills.")
    if activity_percent < 50:
        suggestions.append("Player activity level drops during rallies. Improve stamina.")
    if weak_side == "Left Side":
        suggestions.append("Left side court recovery is weaker. Practice backhand movements.")
    if weak_side == "Right Side":
        suggestions.append("Right side movement speed needs improvement. Focus on forehand reach.")
    if movement_score > 75:
        suggestions.append("Excellent court movement and recovery speed detected.")
    if court_coverage < 40:
        suggestions.append("Court coverage is limited. Practice reaching all 4 corners.")

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