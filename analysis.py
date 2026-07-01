import cv2
import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def _check_court_color(frame):
    """
    Signal 1: Detect synthetic badminton court surface vs natural grass.
    Badminton courts use solid-color synthetic mats (green, blue, red/orange).
    Grass (football/cricket) has high texture variance and a specific green hue.
    Returns a score from 0.0 to 1.0.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    total_pixels = h * w

    # --- Reject grass-like texture ---
    # Grass has a narrow green hue (35-75) with high saturation AND high texture variance
    grass_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([75, 255, 255]))
    grass_ratio = np.sum(grass_mask > 0) / total_pixels

    # Check texture variance in the green regions (grass is textured, synthetic courts are smooth)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate local variance using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = laplacian.var()

    # High grass ratio + high texture = real grass (football/cricket) → penalize
    is_grass = grass_ratio > 0.25 and texture_variance > 500

    # --- Detect synthetic court colors ---
    # Badminton synthetic green (broader, more saturated, smoother)
    synth_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    # Badminton blue courts
    synth_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    # Badminton red/orange courts
    synth_red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
    synth_red2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))

    court_color_ratio = (np.sum(synth_green > 0) + np.sum(synth_blue > 0) +
                         np.sum(synth_red > 0) + np.sum(synth_red2 > 0)) / total_pixels

    if is_grass:
        return 0.0  # Definitely grass, not badminton
    elif court_color_ratio > 0.15 and texture_variance < 1500:
        return 1.0  # Smooth synthetic court
    elif court_color_ratio > 0.10:
        return 0.6
    elif court_color_ratio > 0.05:
        return 0.3
    else:
        return 0.0


def _detect_net(frame):
    """
    Signal 2: Detect a horizontal net line crossing the middle of the frame.
    Badminton has a prominent net dividing the court horizontally.
    Football/cricket don't have a central net.
    Returns a score from 0.0 to 1.0.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Focus on the center third of the frame (where the net typically appears)
    center_top = int(h * 0.25)
    center_bottom = int(h * 0.75)
    center_region = gray[center_top:center_bottom, :]

    edges = cv2.Canny(center_region, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=int(w * 0.15), maxLineGap=20)

    if lines is None:
        return 0.0

    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Nearly horizontal lines (within 15 degrees)
        if angle < 15 or angle > 165:
            horizontal_lines.append(length)

    if not horizontal_lines:
        return 0.0

    max_line_length = max(horizontal_lines)
    line_coverage = max_line_length / w

    # A prominent horizontal line spanning a good portion of the frame
    if line_coverage > 0.4 and len(horizontal_lines) >= 2:
        return 1.0
    elif line_coverage > 0.25:
        return 0.7
    elif line_coverage > 0.15:
        return 0.4
    else:
        return 0.1


def _analyze_line_pattern(frame):
    """
    Signal 3: Analyze court line density and pattern.
    Badminton courts have dense, structured boundary + service lines.
    Football pitches have fewer lines relative to visible area.
    Returns a score from 0.0 to 1.0.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect white/bright lines
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    white_ratio = np.sum(white_mask > 0) / (h * w)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=int(min(h, w) * 0.08), maxLineGap=15)

    if lines is None:
        return 0.0

    num_lines = len(lines)

    # Separate horizontal and vertical lines
    h_lines = 0
    v_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 25 or angle > 155:
            h_lines += 1
        elif 65 < angle < 115:
            v_lines += 1

    # Badminton courts have both horizontal and vertical lines (grid-like)
    has_grid = h_lines >= 3 and v_lines >= 2

    # White lines present + good line count + grid pattern = badminton court
    if white_ratio > 0.01 and num_lines > 12 and has_grid:
        return 1.0
    elif white_ratio > 0.008 and num_lines > 8 and has_grid:
        return 0.7
    elif white_ratio > 0.005 and num_lines > 5:
        return 0.4
    else:
        return 0.1


def _check_court_ratio(frame):
    """
    Signal 4: Check if the dominant rectangular region has a badminton-like aspect ratio.
    Badminton court is ~2.2:1 (length:width). We detect the largest rectangular contour.
    Returns a score from 0.0 to 1.0.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Find the largest contour that could be a court
    h, w = frame.shape[:2]
    frame_area = h * w
    best_score = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.05:  # Too small to be a court
            continue

        # Approximate to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Get bounding rectangle
        rect = cv2.minAreaRect(contour)
        box_w, box_h = rect[1]
        if box_w == 0 or box_h == 0:
            continue

        aspect = max(box_w, box_h) / min(box_w, box_h)

        # Badminton court ratio is ~2.2:1, allow range 1.5:1 to 3.0:1
        if 1.5 <= aspect <= 3.0:
            # Closer to 2.2 is better
            ratio_diff = abs(aspect - 2.2)
            if ratio_diff < 0.3:
                best_score = max(best_score, 1.0)
            elif ratio_diff < 0.6:
                best_score = max(best_score, 0.7)
            elif ratio_diff < 1.0:
                best_score = max(best_score, 0.4)

    return best_score


def _check_indoor_setting(frame):
    """
    Signal 5: Check if the setting appears indoor/enclosed.
    Badminton is typically played indoors with dark ceiling/walls around the court.
    Football is outdoor with sky/stands. 
    Returns a score from 0.0 to 1.0.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Check the top 15% of the frame for sky-like colors (outdoor indicator)
    top_region = hsv[:int(h * 0.15), :, :]
    # Sky: low saturation, high value (bright) or blue hue
    sky_mask = cv2.inRange(top_region, np.array([90, 20, 150]), np.array([130, 255, 255]))
    bright_mask = cv2.inRange(top_region, np.array([0, 0, 200]), np.array([180, 40, 255]))
    sky_ratio = (np.sum(sky_mask > 0) + np.sum(bright_mask > 0)) / (top_region.shape[0] * top_region.shape[1])

    # Check the periphery (edges) for dark regions (indoor indicator)
    # Top strip
    top_strip = frame[:int(h * 0.1), :]
    # Bottom strip
    bottom_strip = frame[int(h * 0.9):, :]
    # Left strip
    left_strip = frame[:, :int(w * 0.1)]
    # Right strip
    right_strip = frame[:, int(w * 0.9):]

    periphery = np.concatenate([
        top_strip.reshape(-1, 3),
        bottom_strip.reshape(-1, 3),
        left_strip.reshape(-1, 3),
        right_strip.reshape(-1, 3)
    ])
    avg_brightness = np.mean(periphery)

    # Large sky area = outdoor
    if sky_ratio > 0.3:
        return 0.0
    elif sky_ratio > 0.15:
        return 0.3

    # Dark periphery = likely indoor
    if avg_brightness < 80:
        return 1.0
    elif avg_brightness < 120:
        return 0.6
    elif avg_brightness < 160:
        return 0.3
    else:
        return 0.1


def validate_badminton_video(video_path):
    """
    Multi-signal badminton video validation.
    Uses 5 weighted signals to determine if a video shows a badminton court:
    1. Court color (synthetic vs grass) - 25%
    2. Net detection - 25%
    3. Line pattern density - 20%
    4. Court rectangle ratio - 15%
    5. Indoor/enclosed setting - 15%
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps

    if duration < 2:
        cap.release()
        return False, "Video too short (minimum 2 seconds required)"

    # Sample frames at different points in the video
    sample_points = [int(total_frames * x) for x in [0.1, 0.3, 0.5, 0.7, 0.9]]
    frame_scores = []

    for pos in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))

        # Calculate all 5 signals
        court_color_score = _check_court_color(frame)
        net_score = _detect_net(frame)
        line_score = _analyze_line_pattern(frame)
        ratio_score = _check_court_ratio(frame)
        indoor_score = _check_indoor_setting(frame)

        # Weighted combination
        weighted_score = (
            court_color_score * 0.25 +
            net_score * 0.25 +
            line_score * 0.20 +
            ratio_score * 0.15 +
            indoor_score * 0.15
        )
        frame_scores.append({
            'total': weighted_score,
            'court_color': court_color_score,
            'net': net_score,
            'lines': line_score,
            'ratio': ratio_score,
            'indoor': indoor_score
        })

    cap.release()

    if not frame_scores:
        return False, "Could not read video frames"

    # Average score across sampled frames
    avg_score = np.mean([s['total'] for s in frame_scores])

    # Check for strong grass detection (automatic reject)
    avg_court_color = np.mean([s['court_color'] for s in frame_scores])
    if avg_court_color == 0.0:
        # Every frame detected grass — almost certainly football/cricket
        avg_net = np.mean([s['net'] for s in frame_scores])
        if avg_net < 0.4:
            return False, ("This appears to be an outdoor field sport video (football, cricket, etc.), "
                           "not badminton. Please upload a video showing a badminton court.")

    # Threshold for acceptance
    if avg_score >= 0.35:
        return True, "Valid badminton video"
    elif avg_score >= 0.25:
        # Borderline — check if at least 2 strong signals are present
        avg_signals = {k: np.mean([s[k] for s in frame_scores])
                       for k in ['court_color', 'net', 'lines', 'ratio', 'indoor']}
        strong_signals = sum(1 for v in avg_signals.values() if v >= 0.5)
        if strong_signals >= 2:
            return True, "Valid badminton video"
        else:
            return False, ("This doesn't appear to be a badminton video. The video lacks "
                           "key badminton indicators (court markings, net, indoor setting). "
                           "Please upload a clear badminton court video.")
    else:
        return False, ("This doesn't appear to be a badminton video. Please upload a video "
                       "showing a badminton court with visible court lines, net, and players.")


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


# ═══════════════════════════════════════════════════════════════
# REAL PLAYER TRACKING & ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

# Badminton court dimensions in meters
COURT_LENGTH = 13.4
COURT_WIDTH = 6.1


def _track_player(video_path, sample_rate=2):
    """
    Track the primary player's position frame-by-frame using
    background subtraction + contour detection.
    Returns list of (norm_x, norm_y, frame_idx) positions, fps, total_frames.
    Positions are normalized to 0.0-1.0 range.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=50, detectShadows=False
    )

    proc_w, proc_h = 320, 240
    positions = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        small = cv2.resize(frame, (proc_w, proc_h))
        fg_mask = bg_sub.apply(small)

        # Let BG model warm up for first 20 frames, then track every Nth frame
        if frame_idx < 20 or frame_idx % sample_rate != 0:
            continue

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        player_contours = [
            c for c in contours if 400 < cv2.contourArea(c) < 25000
        ]

        if player_contours:
            largest = max(player_contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                norm_x = np.clip(cx / proc_w, 0, 1)
                norm_y = np.clip(cy / proc_h, 0, 1)
                positions.append((norm_x, norm_y, frame_idx))

    cap.release()
    return positions, fps, total_frames


def _draw_court(width=600, height=280):
    """Draw a badminton court diagram (top-down view)."""
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (34, 85, 34)  # BGR dark green

    mx = int(width * 0.05)
    my = int(height * 0.05)
    cw = width - 2 * mx
    ch = height - 2 * my
    white = (255, 255, 255)

    # Outer boundary
    cv2.rectangle(court, (mx, my), (mx + cw, my + ch), white, 2)
    # Net (center horizontal)
    net_y = my + ch // 2
    cv2.line(court, (mx, net_y), (mx + cw, net_y), (200, 200, 255), 3)
    # Singles sidelines
    so = int(cw * 0.075)
    cv2.line(court, (mx + so, my), (mx + so, my + ch), white, 1)
    cv2.line(court, (mx + cw - so, my), (mx + cw - so, my + ch), white, 1)
    # Short service lines (1.98m from net)
    svc = int(ch * 0.148)
    cv2.line(court, (mx + so, net_y - svc), (mx + cw - so, net_y - svc), white, 1)
    cv2.line(court, (mx + so, net_y + svc), (mx + cw - so, net_y + svc), white, 1)
    # Long service lines for doubles (0.76m from back)
    ls = int(ch * 0.057)
    cv2.line(court, (mx, my + ls), (mx + cw, my + ls), white, 1)
    cv2.line(court, (mx, my + ch - ls), (mx + cw, my + ch - ls), white, 1)
    # Center line (each half)
    cx = mx + cw // 2
    cv2.line(court, (cx, my), (cx, net_y - svc), white, 1)
    cv2.line(court, (cx, net_y + svc), (cx, my + ch), white, 1)

    return court


def _generate_heatmap(positions, output_dir):
    """Generate a court heatmap from tracked player positions."""
    court_w, court_h = 600, 280
    heatmap = np.zeros((court_h, court_w), dtype=np.float32)

    for x, y, _ in positions:
        px = int(np.clip(x * (court_w - 1), 0, court_w - 1))
        py = int(np.clip(y * (court_h - 1), 0, court_h - 1))
        heatmap[py, px] += 1

    heatmap = cv2.GaussianBlur(heatmap, (61, 61), 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    court = _draw_court(court_w, court_h)
    result = cv2.addWeighted(heatmap_color, 0.6, court, 0.4, 0)

    # Re-draw white court lines on top
    gray_court = cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
    result[gray_court > 180] = [255, 255, 255]

    path = os.path.join(output_dir, "heatmap.png")
    cv2.imwrite(path, result)
    return path


def _generate_trajectory(positions, output_dir):
    """Draw player movement trajectory on court diagram."""
    court_w, court_h = 600, 280
    court = _draw_court(court_w, court_h)
    path = os.path.join(output_dir, "trajectory.png")

    if len(positions) < 2:
        cv2.imwrite(path, court)
        return path

    pts = [
        (int(x * (court_w - 1)), int(y * (court_h - 1)))
        for x, y, _ in positions
    ]

    for i in range(1, len(pts)):
        progress = i / len(pts)
        color = (int(255 * (1 - progress)), 50, int(255 * progress))
        cv2.line(court, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    # Start (green) and end (red) markers
    cv2.circle(court, pts[0], 8, (0, 255, 0), -1)
    cv2.circle(court, pts[-1], 8, (0, 0, 255), -1)
    cv2.putText(court, "S", (pts[0][0] - 4, pts[0][1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(court, "E", (pts[-1][0] - 4, pts[-1][1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.imwrite(path, court)
    return path


def _detect_shots(positions, fps, sample_rate=2):
    """
    Detect shots from movement acceleration spikes.
    A shot is identified when player acceleration exceeds mean + 1.5*std,
    classified by court zone: Front (net/drop), Mid (drive/clear), Rear (smash/lob).
    """
    if len(positions) < 5:
        return []

    time_step = sample_rate / fps

    velocities = []
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i - 1][0]) * COURT_WIDTH
        dy = (positions[i][1] - positions[i - 1][1]) * COURT_LENGTH
        vel = np.sqrt(dx ** 2 + dy ** 2) / time_step
        velocities.append(vel)

    if len(velocities) < 3:
        return []

    accelerations = np.abs(np.diff(velocities))
    mean_acc = np.mean(accelerations)
    std_acc = np.std(accelerations)
    threshold = max(mean_acc + 1.5 * std_acc, 2.0)

    shots = []
    min_gap = max(int(fps * 0.5 / sample_rate), 3)
    last_idx = -min_gap

    for i, acc in enumerate(accelerations):
        if acc > threshold and (i - last_idx) >= min_gap:
            pos_idx = i + 1
            x, y, frame = positions[pos_idx]

            if y < 0.33:
                zone, shot_type = "Front Court", "Net/Drop"
            elif y < 0.66:
                zone, shot_type = "Mid Court", "Drive/Clear"
            else:
                zone, shot_type = "Rear Court", "Smash/Lob"

            shots.append({
                "frame": frame, "time": round(frame / fps, 1),
                "x": x, "y": y, "zone": zone,
                "type": shot_type, "intensity": round(float(acc), 2)
            })
            last_idx = i

    return shots


def _calculate_metrics(positions, fps, total_frames, sample_rate=2):
    """Calculate real performance metrics from tracked player positions."""
    time_step = sample_rate / fps
    duration = total_frames / max(fps, 1)
    empty = {
        "distance_covered": 0.0, "avg_speed": 0.0, "max_speed": 0.0,
        "direction_changes": 0, "court_coverage": 0, "activity_percent": 0,
        "left_pct": 50, "right_pct": 50, "front_pct": 50, "rear_pct": 50,
        "velocities": [], "frame_times": []
    }
    if len(positions) < 3:
        return empty

    # Distance & Speed
    distances, velocities, frame_times = [], [], []
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i - 1][0]) * COURT_WIDTH
        dy = (positions[i][1] - positions[i - 1][1]) * COURT_LENGTH
        d = np.sqrt(dx ** 2 + dy ** 2)
        distances.append(d)
        velocities.append(d / time_step)
        frame_times.append(positions[i][2] / fps)

    total_dist = sum(distances)
    avg_speed = total_dist / max(duration, 0.1)
    max_speed = max(velocities) if velocities else 0

    # Direction changes (angle > 45 deg between consecutive vectors)
    dir_changes = 0
    for i in range(1, len(positions) - 1):
        v1x = positions[i][0] - positions[i - 1][0]
        v1y = positions[i][1] - positions[i - 1][1]
        v2x = positions[i + 1][0] - positions[i][0]
        v2y = positions[i + 1][1] - positions[i][1]
        m1 = np.sqrt(v1x ** 2 + v1y ** 2)
        m2 = np.sqrt(v2x ** 2 + v2y ** 2)
        if m1 > 0.005 and m2 > 0.005:
            cos_a = np.clip((v1x * v2x + v1y * v2y) / (m1 * m2), -1, 1)
            if np.degrees(np.arccos(cos_a)) > 45:
                dir_changes += 1

    # Court coverage (10x10 grid)
    visited = set()
    for x, y, _ in positions:
        visited.add((min(int(x * 10), 9), min(int(y * 10), 9)))
    coverage = int(len(visited) / 100 * 100)

    # Activity rate
    active = sum(1 for v in velocities if v > 0.5)
    activity = int(active / max(len(velocities), 1) * 100)

    # Balance
    n = max(len(positions), 1)
    left = sum(1 for x, y, _ in positions if x < 0.5)
    front = sum(1 for x, y, _ in positions if y < 0.5)

    return {
        "distance_covered": round(total_dist, 1),
        "avg_speed": round(avg_speed, 2),
        "max_speed": round(max_speed, 2),
        "direction_changes": dir_changes,
        "court_coverage": coverage,
        "activity_percent": activity,
        "left_pct": int(left / n * 100),
        "right_pct": 100 - int(left / n * 100),
        "front_pct": int(front / n * 100),
        "rear_pct": 100 - int(front / n * 100),
        "velocities": velocities,
        "frame_times": frame_times
    }


def _generate_timeline_chart(velocities, frame_times, output_dir):
    """Generate movement speed timeline chart using matplotlib."""
    if not HAS_MATPLOTLIB or not velocities:
        return None

    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
    fig.patch.set_facecolor('#0a1120')
    ax.set_facecolor('#0a1120')

    ax.fill_between(frame_times, velocities, alpha=0.3, color='#3b82f6')
    ax.plot(frame_times, velocities, color='#3b82f6', linewidth=1.5)

    if len(velocities) > 10:
        win = min(20, len(velocities) // 3)
        ma = np.convolve(velocities, np.ones(win) / win, mode='valid')
        ax.plot(frame_times[:len(ma)], ma, color='#f59e0b', linewidth=2,
                label='Avg', linestyle='--')
        ax.legend(facecolor='#0a1120', edgecolor='#1f2937',
                  labelcolor='#9ca3af', fontsize=8)

    ax.set_xlabel('Time (s)', color='#6b7280', fontsize=9)
    ax.set_ylabel('Speed (m/s)', color='#6b7280', fontsize=9)
    ax.set_title('Movement Intensity Over Time', color='#e2e8f0',
                 fontsize=11, fontweight='bold')
    ax.tick_params(colors='#4b5563', labelsize=8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color('#1f2937')
    ax.grid(axis='y', alpha=0.1, color='#374151')

    plt.tight_layout()
    path = os.path.join(output_dir, "timeline.png")
    fig.savefig(path, facecolor='#0a1120', bbox_inches='tight')
    plt.close(fig)
    return path


def _generate_shot_chart(shots, output_dir):
    """Generate shot distribution bar chart."""
    if not HAS_MATPLOTLIB or not shots:
        return None

    zones = {"Front Court": 0, "Mid Court": 0, "Rear Court": 0}
    for s in shots:
        zones[s["zone"]] += 1

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
    fig.patch.set_facecolor('#0a1120')
    ax.set_facecolor('#0a1120')

    colors = ['#10b981', '#3b82f6', '#f59e0b']
    bars = ax.bar(zones.keys(), zones.values(), color=colors, width=0.5,
                  edgecolor='#1f2937', linewidth=0.5)
    for bar, val in zip(bars, zones.values()):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha='center', color='#e2e8f0', fontsize=10,
                    fontweight='bold')

    ax.set_ylabel('Shots', color='#6b7280', fontsize=9)
    ax.set_title('Shot Distribution by Zone', color='#e2e8f0',
                 fontsize=11, fontweight='bold')
    ax.tick_params(colors='#4b5563', labelsize=8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color('#1f2937')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(output_dir, "shot_chart.png")
    fig.savefig(path, facecolor='#0a1120', bbox_inches='tight')
    plt.close(fig)
    return path


def analyze_video(video_path):
    """
    Complete badminton video analysis with real player tracking.
    Uses background subtraction for player detection, tracks centroid positions,
    generates heatmap & trajectory images, detects shots via acceleration spikes,
    and calculates genuine performance metrics.
    """
    output_dir = os.path.dirname(os.path.abspath(video_path))
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = 2

    # ── Step 1: Track player positions ──
    positions, fps, total_frames = _track_player(video_path, sample_rate)
    duration = total_frames / max(fps, 1)

    # ── Step 2: Calculate real metrics ──
    m = _calculate_metrics(positions, fps, total_frames, sample_rate)

    # ── Step 3: Detect shots ──
    shots = _detect_shots(positions, fps, sample_rate)
    total_shots = len(shots)
    shot_zones = {"Front Court": 0, "Mid Court": 0, "Rear Court": 0}
    for s in shots:
        shot_zones[s["zone"]] += 1

    # ── Step 4: Generate visualizations ──
    heatmap_path = _generate_heatmap(positions, output_dir) if positions else None
    trajectory_path = _generate_trajectory(positions, output_dir) if positions else None
    timeline_path = _generate_timeline_chart(
        m["velocities"], m["frame_times"], output_dir
    )
    shot_chart_path = _generate_shot_chart(shots, output_dir)

    # ── Step 5: Derive composite scores from REAL data ──
    dist = m["distance_covered"]
    avg_spd = m["avg_speed"]
    dir_ch = m["direction_changes"]
    coverage = m["court_coverage"]
    activity = m["activity_percent"]

    # Movement score: composite of distance, speed, agility
    dist_score = min(int(dist / 0.8), 100)       # ~80m = 100
    speed_score = min(int(avg_spd * 25), 100)     # ~4 m/s avg = 100
    agility_score = min(int(dir_ch * 1.5), 100)   # ~67 dir changes = 100
    movement_score = min(int(
        dist_score * 0.35 + speed_score * 0.35 + agility_score * 0.30
    ), 100)

    # Attack / Defense from real position data
    front_shots = shot_zones.get("Front Court", 0) + shot_zones.get("Mid Court", 0)
    total_s = max(total_shots, 1)
    attack_score = min(int(
        m["front_pct"] * 0.4 +
        (front_shots / total_s) * 60 +
        min(total_shots * 3, 40)
    ), 100)
    defense_score = min(int(
        m["rear_pct"] * 0.4 +
        coverage * 0.3 +
        min(dir_ch * 0.8, 30)
    ), 100)

    # Weak side
    if m["left_pct"] < m["right_pct"] - 15:
        weak_side = "Left Side"
    elif m["right_pct"] < m["left_pct"] - 15:
        weak_side = "Right Side"
    else:
        weak_side = "Balanced"

    # Dominant zone
    if m["front_pct"] > 60:
        dominant_zone = "Front Court"
    elif m["rear_pct"] > 60:
        dominant_zone = "Rear Court"
    elif m["left_pct"] > 60:
        dominant_zone = "Left Court"
    elif m["right_pct"] > 60:
        dominant_zone = "Right Court"
    else:
        dominant_zone = "Balanced"

    # Labels
    if movement_score > 75: fitness = "Excellent"
    elif movement_score > 45: fitness = "Average"
    else: fitness = "Needs Improvement"

    if activity > 70: stamina = "High"
    elif activity > 40: stamina = "Moderate"
    else: stamina = "Low"

    if avg_spd > 3.0: footwork = "Fast"
    elif avg_spd > 1.5: footwork = "Average"
    else: footwork = "Slow"

    if movement_score > 80: grade = "A+"
    elif movement_score > 60: grade = "A"
    elif movement_score > 40: grade = "B"
    elif movement_score > 25: grade = "C"
    else: grade = "D"

    # ── Step 6: Generate real suggestions ──
    suggestions = []
    if movement_score < 40:
        suggestions.append(
            f"\u26a1 Movement intensity is low ({movement_score}%). "
            "Increase footwork drills and court sprints."
        )
    if activity < 50:
        suggestions.append(
            f"\U0001f3c3 Active only {activity}% of the time. "
            "Work on maintaining engagement during rallies."
        )
    if coverage < 40:
        suggestions.append(
            f"\U0001f4d0 Court coverage is only {coverage}%. "
            "Practice reaching all 4 corners with shadow footwork."
        )
    if weak_side == "Left Side":
        suggestions.append(
            "\u2b05\ufe0f Left side movement is weaker. "
            "Practice backhand lunges and cross-court recovery."
        )
    elif weak_side == "Right Side":
        suggestions.append(
            "\u27a1\ufe0f Right side needs work. "
            "Focus on forehand reach and side-step drills."
        )
    if total_shots > 0 and shot_zones.get("Rear Court", 0) > total_shots * 0.6:
        suggestions.append(
            "\U0001f3af Most shots from rear court. "
            "Move to net more aggressively for winning shots."
        )
    if dir_ch < 10 and duration > 5:
        suggestions.append(
            "\U0001f504 Very few direction changes detected. "
            "Improve lateral agility and reaction speed."
        )
    if movement_score > 75:
        suggestions.append(
            "\U0001f31f Excellent court movement and recovery speed! "
            "Maintain this level."
        )
    if total_shots == 0:
        suggestions.append(
            "\u26a0\ufe0f No clear shot actions detected. "
            "This may indicate low rally activity or camera angle issues."
        )
    if avg_spd > 3.0:
        suggestions.append(
            "\U0001f680 Great speed on court! "
            "Focus on maintaining this pace throughout longer rallies."
        )

    return {
        # Composite scores (0-100)
        "movement_score": movement_score,
        "activity_percent": activity,
        "court_coverage": coverage,
        "attack_score": attack_score,
        "defense_score": defense_score,
        "fitness": fitness,
        "stamina": stamina,
        "footwork": footwork,
        "grade": grade,
        "weak_side": weak_side,
        "dominant_zone": dominant_zone,

        # Real tracked metrics
        "distance_covered": m["distance_covered"],
        "avg_speed": m["avg_speed"],
        "max_speed": m["max_speed"],
        "direction_changes": dir_ch,
        "total_shots": total_shots,
        "shot_zones": shot_zones,
        "shots": shots,
        "duration": round(duration, 1),

        # Position balance
        "left_pct": m["left_pct"],
        "right_pct": m["right_pct"],
        "front_pct": m["front_pct"],
        "rear_pct": m["rear_pct"],

        # Image paths
        "heatmap_path": heatmap_path,
        "trajectory_path": trajectory_path,
        "timeline_path": timeline_path,
        "shot_chart_path": shot_chart_path,

        # Frame data
        "active_frames": int(len(positions) * (activity / 100)) if positions else 0,
        "total_frames": total_frames,
        "tracked_positions": len(positions),

        # Suggestions
        "suggestions": suggestions
    }