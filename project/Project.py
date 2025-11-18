import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import random
Appropriate_distance=True #F451
total_distance = 0.0
distance_samples = 0  # per-frame samples for pointing error
clicks = 0
latency_start = None
latency_total = 0.0
latency_count = 0
cursor_movements = []

# System latency (avg per-frame processing time)
proc_time_total = 0.0
proc_frames = 0

# =========================
# Setup
# =========================
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# ---------- Test Mode (Circles) ----------
# test_mode = False
# TEST_TRIALS = 10
# TARGET_RADIUS = 40              # in FRAME (camera) pixels
# test_trials_done = 0

# Keep last gaze point in FRAME coords for hit-testing
gaze_fx, gaze_fy = None, None

# ---------- Calibration ----------
calibrated = False
cal_pts = []
cal_idx = 0
gx_min = gx_max = None
gy_min = gy_max = None

def face_region_from_landmarks(landmarks, frame_w, frame_h):
    if not landmarks:
        return None  # no face detected

    # Scale normalized landmark coords to frame (pixel) space
    xs = [int(lm.x * frame_w) for lm in landmarks]
    ys = [int(lm.y * frame_h) for lm in landmarks]

    # Clamp to valid frame bounds
    x_min = max(0, min(xs))
    x_max = min(frame_w - 1, max(xs))
    y_min = max(0, min(ys))
    y_max = min(frame_h - 1, max(ys))

    # Dimensions of the face region
    Rx = max(0, x_max - x_min)
    Ry = max(0, y_max - y_min)

    return {
        "Bx": frame_w,
        "By": frame_h,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "Rx": Rx,
        "Ry": Ry,
    }


def map_to_screen_from_cal(gx, gy, screen_w, screen_h):
    """Map gaze (FRAME coords) to SCREEN coords using calibrated bounds."""
    x = (gx - gx_min) / max(1e-6, (gx_max - gx_min))
    y = (gy - gy_min) / max(1e-6, (gy_max - gy_min))
    sx = int(np.clip(x, 0, 1) * (screen_w - 1))
    sy = int(np.clip(y, 0, 1) * (screen_h - 1))
    return sx, sy


instruction_text = "'c' = Calibrate | 'q' = Quit"
print(instruction_text)

while True:
    frame_start = time.time()  # for system latency

    ok, frame = cam.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points: #F451
        landmarks = landmark_points[0].landmark

        box = face_region_from_landmarks(landmarks, frame_w, frame_h)
        if box:
            Rx, Ry = box["Rx"], box["Ry"]
            Bx, By = box["Bx"], box["By"]
            x_min, y_min, x_max, y_max = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            ratio_calc = 1 - (((By * Bx) - (Ry * Rx)) / (By * Bx))
            HigherThreshold = 0.3
            LowerThreshold = 0.1

            color = (0, 255, 0) if (ratio_calc < HigherThreshold and ratio_calc > LowerThreshold) else (0, 0,255)  # green if smaller, red if larger
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            if (ratio_calc < HigherThreshold and ratio_calc > LowerThreshold):
                Appropriate_distance = True

            elif ratio_calc < 0.2:
                # Make the frame completely black
                Appropriate_distance = False
                frame[:] = (0, 0, 0)

                # Display a message at the center of the frame
                msg = "Face too far! Please move closer."
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = (Bx - text_size[0]) // 2
                text_y = (By + text_size[1]) // 2
                cv2.putText(frame, msg, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            elif ratio_calc > 0.1:
                frame[:] = (0, 0, 0)
                Appropriate_distance = False
                # Display a message at the center of the frame
                msg = "Face too close! Please move back."
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = (Bx - text_size[0]) // 2
                text_y = (By + text_size[1]) // 2
                cv2.putText(frame, msg, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw iris landmarks (474..477)
        if Appropriate_distance==True:
            for idx, lm in enumerate(landmarks[474:478]):
                x = int(lm.x * frame_w)
                y = int(lm.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                if idx == 1:
                    # Save latest gaze in FRAME coords (for test-mode hit test)
                    gaze_fx, gaze_fy = x, y

                    # Default mapping (normalized -> screen)
                    screen_x = screen_w * lm.x
                    screen_y = screen_h * lm.y

                    # Use calibration if available (FRAME -> screen)
                    if calibrated and gx_min is not None:
                        screen_x, screen_y = map_to_screen_from_cal(gaze_fx, gaze_fy, screen_w, screen_h)

                    # Clamp
                    screen_x = int(np.clip(screen_x, 0, screen_w - 1))
                    screen_y = int(np.clip(screen_y, 0, screen_h - 1))

                    # ---- Metrics (per frame) ----
                    cur_x, cur_y = pyautogui.position()
                    d = np.hypot(screen_x - cur_x, screen_y - cur_y)
                    total_distance += d
                    distance_samples += 1

                    cursor_movements.append((screen_x, screen_y))
                    if len(cursor_movements) > 10:
                        cursor_movements.pop(0)

                    pyautogui.moveTo(screen_x, screen_y)

            # Blink landmarks
            left = [landmarks[145], landmarks[159]]
            right = [landmarks[374], landmarks[386]]

            # Scrolling landmarks
            left_up = [landmarks[63], landmarks[119]]  # left eye upper lid and eyebrow
            right_down = [landmarks[293], landmarks[348]]  # right eye lower lid and cheek

            # Draw left eyelid points
            for p in left:
                x = int(p.x * frame_w)
                y = int(p.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Draw left_up points
            for landmark in left_up:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue circles for left_up

            # Draw right_down points
            for landmark in right_down:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for right_down

            # Both eyes blinked - skip to avoid double clicks
            if (left[0].y - left[1].y < 0.0099) and (right[0].y - right[1].y) < 0.009:
                # Both eyes blinked - skip to avoid double clicks
                continue

            # Left blink -> left click OR calibration capture
            if (left[0].y - left[1].y) < 0.0099:
                # During calibration: capture corner and skip click
                if 0 <= cal_idx < 4 and gaze_fx is not None:
                    cal_pts.append((gaze_fx, gaze_fy))
                    cal_idx += 1
                    steps = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
                    if cal_idx < 4:
                        print(f"[Calibration] Captured {steps[cal_idx - 1]}. Next: {steps[cal_idx]}")
                        time.sleep(1.5) #F451
                    else:
                        xs = [p[0] for p in cal_pts]
                        ys = [p[1] for p in cal_pts]
                        gx_min, gx_max = min(xs), max(xs)
                        gy_min, gy_max = min(ys), max(ys)
                        calibrated = True
                        print(f"[Calibration Done] X:[{gx_min},{gx_max}] Y:[{gy_min},{gy_max}]")
                    continue  # don't click while calibrating

                # Normal left click
                pyautogui.click()
                pyautogui.sleep(2)  # Comment to enable faster clicking
                clicks += 1

                # Inter-click interval
                now = time.time()
                if latency_start:
                    latency_total += now - latency_start
                    latency_count += 1
                latency_start = now

                # if test_mode and (test_target is not None) and (test_start_time is not None):
                #     sel_time = time.time() - test_start_time

                #     hit = False
                #     if (gaze_fx is not None) and (gaze_fy is not None):
                #         dx = gaze_fx - test_target[0]
                #         dy = gaze_fy - test_target[1]
                #         dist_click = (dx**2 + dy**2) ** 0.5
                #         hit = dist_click <= TARGET_RADIUS

                #     if hit:
                #         test_hits += 1
                #         # test_times.append(sel_time)

                #     test_trials_done += 1
                #     test_target = None
                #     test_start_time = None

            # Right blink -> right click
            if (right[0].y - right[1].y) < 0.009:
                pyautogui.click(button='right')
                pyautogui.sleep(1)

            # Scrolling
            if calibrated:  # Only allow scrolling if not in calibration mode
                if (left_up[0].y - left_up[1].y) < -0.1:
                    pyautogui.scroll(-1)  # Scroll down 10 steps

                if (right_down[0].y - right_down[1].y) < -0.12:
                    pyautogui.scroll(1)  # Scroll up 10 steps

        # ---------- Calibration HUD ----------
        if 0 <= cal_idx < 4:
            prompts = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
            cv2.putText(frame, f"CALIBRATION: Look at {prompts[cal_idx]} and blink (L)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---------- Test Mode (Targets) ----------
        # if test_mode:
        #     if (test_target is None) and (test_trials_done < TEST_TRIALS):
        #         margin = TARGET_RADIUS + 20
        #         tx = random.randint(margin, max(margin, frame_w - margin))
        #         ty = random.randint(margin, max(margin, frame_h - margin))
        #         test_target = (tx, ty)
        #         test_start_time = time.time()

        #     if test_target is not None:
        #         cv2.circle(frame, test_target, TARGET_RADIUS, (0, 0, 255), 2)
        #         cv2.circle(frame, test_target, 3, (0, 0, 255), -1)

        #     cv2.putText(frame, f"TEST MODE  {test_trials_done}/{TEST_TRIALS}",
        #                 (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        #     if test_trials_done >= TEST_TRIALS:
        #         cv2.putText(frame, "Test complete - press 't' to reset/exit",
        #                     (10, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---------- Instruction overlay ----------
        cv2.putText(frame, instruction_text, (10, frame_h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Eye Controlled Mouse', frame)

        # System latency accumulation
        frame_proc_ms = (time.time() - frame_start) * 1000.0
        proc_time_total += frame_proc_ms
        proc_frames += 1

        key = cv2.waitKey(1) & 0xFF

    # Toggle Test Mode
    # if key == ord('t'):
    #     test_mode = not test_mode
    #     if test_mode:
    #         test_target = None
    #         test_start_time = None
    #         print("\n[Test Mode Started] Blink (left) to select red circles.\n")
    #     else:
    #         print("[Test Mode Stopped]")
    #         test_target = None
    #         test_start_time = None

    # Start Calibration
    if key == ord('c'):
        calibrated = False
        cal_pts.clear()
        cal_idx = 0
        print("\n[Calibration Started] Look at TOP-LEFT and blink (L).\n")

    # Quit
    if key == ord('q'):
        # ---- Final metrics ----
        avg_distance = total_distance / max(1, distance_samples)
        avg_interval = latency_total / max(1, latency_count)

        if cursor_movements:
            pts = np.array(cursor_movements)
            mean_xy = pts.mean(axis=0)
            cursor_stability = np.sqrt(np.mean(np.sum((pts - mean_xy) ** 2, axis=1)))
        else:
            cursor_stability = 0.0

        avg_system_latency_ms = proc_time_total / max(1, proc_frames)
        break

        print('\n--- Performance Metrics ---')
        print(f"Avg. Pointing Error (px):     {avg_distance:.2f}")  # Cursor presision
        print(f"Cursor Stability (RMS px):    {cursor_stability:.2f}")  # Cursor smoothness
        print(f"System Latency (ms):          {avg_system_latency_ms:.1f}")  # Processing delay
        print(f"Inter-click Interval (s):     {avg_interval:.2f}")  # Time between clicks

cam.release()
cv2.destroyAllWindows()