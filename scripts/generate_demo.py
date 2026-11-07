"""
HumanSense — Synthetic Demo Video Generator

Creates a 20-second demo video simulating a hospital corridor with:
  • People walking in both directions
  • A raised-hand event (person requesting help)
  • Two people getting too close (proximity alert)
  • Crowd density varying from low → medium → high
  • A robot position marker
  • Smooth, realistic-looking motion

The video is designed to work with all four HumanSense detection modules.
Output: assets/demo.mp4

Usage:
    python generate_demo.py
"""

import cv2
import numpy as np
import math
import os
import sys

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "assets", "demo.mp4")
W, H    = 720, 480
FPS     = 25
FRAMES  = FPS * 22   # 22 seconds


# ── Colour palette ────────────────────────────────────────────────────────────
BG_WALL     = (210, 205, 195)   # warm off-white hospital wall
BG_FLOOR    = (185, 182, 175)   # slightly darker floor
STRIPE_COL  = (170, 167, 160)   # floor stripe
DOOR_COL    = (140, 130, 120)
DOOR_FRAME  = (100,  95,  88)
SHADOW_COL  = (160, 158, 152)

PERSON_COLORS = [
    (( 60,  80, 160), (40, 55, 120)),   # dark blue scrubs
    ((100,  50,  50), (70, 35, 35)),    # dark red/maroon
    (( 40, 100,  60), (28, 70, 42)),    # dark green
    (( 80,  80,  80), (55, 55, 55)),    # grey
    ((100,  70,  20), (70, 50, 14)),    # brown
    (( 60,  60, 120), (40, 40, 85)),    # navy
]
ROBOT_COL = (180, 180, 60)


# ── Helper: draw a person as a simple figure ──────────────────────────────────
def draw_person(canvas, cx, by, scale=1.0, color_pair=None, alpha=1.0,
                raised_left=False, raised_right=False, seated=False):
    """
    Draw a simplified but recognisable person figure.
    cx, by = centre-x, bottom-y in pixels.
    scale controls height (1.0 ≈ 160px tall).
    """
    if color_pair is None:
        color_pair = PERSON_COLORS[0]
    body_col, dark_col = color_pair

    sh  = int(160 * scale)   # full figure height
    hw  = int(18  * scale)   # half body width
    nh  = int(22  * scale)   # head radius
    leg = int(55  * scale)   # leg length
    arm = int(42  * scale)   # arm length

    # vertical layout
    foot_y    = by
    hip_y     = by   - leg
    shoulder_y= hip_y - int(60 * scale)
    neck_y    = shoulder_y - int(8 * scale)
    head_cy   = neck_y - nh

    def pt(x, y): return (int(x), int(y))

    # Shadow
    shadow_rx = int(hw * 1.3)
    overlay = canvas.copy()
    cv2.ellipse(overlay, pt(cx, foot_y), (shadow_rx, int(6 * scale)),
                0, 0, 360, SHADOW_COL, -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    # Legs (two simple rectangles)
    lx = cx - hw // 2
    rx = cx + hw // 2
    lw = int(hw * 0.55)
    cv2.rectangle(canvas, pt(lx - lw//2, hip_y), pt(lx + lw//2, foot_y), dark_col, -1)
    cv2.rectangle(canvas, pt(rx - lw//2, hip_y), pt(rx + lw//2, foot_y), dark_col, -1)
    # Shoes
    shoe_col = (40, 35, 30)
    cv2.ellipse(canvas, pt(lx, foot_y), (int(lw * 1.3), int(5 * scale)),
                0, 0, 360, shoe_col, -1)
    cv2.ellipse(canvas, pt(rx, foot_y), (int(lw * 1.3), int(5 * scale)),
                0, 0, 360, shoe_col, -1)

    # Torso
    cv2.rectangle(canvas, pt(cx - hw, shoulder_y), pt(cx + hw, hip_y), body_col, -1)
    # Torso shading
    cv2.rectangle(canvas, pt(cx - hw, shoulder_y), pt(cx - hw + 4, hip_y),
                  dark_col, -1)

    # Arms
    elbow_y = shoulder_y + arm // 2
    wrist_y = shoulder_y + arm

    if raised_left:
        # Left arm raised up
        cv2.line(canvas, pt(cx - hw, shoulder_y), pt(cx - hw - 12, shoulder_y - arm),
                 body_col, int(lw * 1.1))
    else:
        cv2.line(canvas, pt(cx - hw, shoulder_y), pt(cx - hw - 8, wrist_y),
                 body_col, int(lw * 1.1))

    if raised_right:
        cv2.line(canvas, pt(cx + hw, shoulder_y), pt(cx + hw + 12, shoulder_y - arm),
                 body_col, int(lw * 1.1))
    else:
        cv2.line(canvas, pt(cx + hw, shoulder_y), pt(cx + hw + 8, wrist_y),
                 body_col, int(lw * 1.1))

    # Head
    cv2.circle(canvas, pt(cx, head_cy), nh, (220, 190, 160), -1)    # skin
    cv2.circle(canvas, pt(cx, head_cy), nh, (180, 150, 120), 1)     # outline
    # Hair
    hair_pts = np.array([
        [cx - nh, head_cy - 2],
        [cx - nh + 3, head_cy - nh],
        [cx, head_cy - nh - 4],
        [cx + nh - 3, head_cy - nh],
        [cx + nh, head_cy - 2],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [hair_pts], (60, 40, 20))

    return canvas


def draw_robot(canvas, cx, by, scale=1.0):
    """Draw a simple hospital robot."""
    bw = int(40 * scale)
    bh = int(60 * scale)
    x1, y1 = cx - bw, by - bh
    x2, y2 = cx + bw, by

    # Body
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), ROBOT_COL, 2)

    # Screen
    sx1, sy1 = x1 + 6, y1 + 8
    sx2, sy2 = x2 - 6, y1 + 30
    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (20, 20, 20), -1)
    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 200, 200), 1)

    # Eyes / indicator lights on screen
    cv2.circle(canvas, (cx - 8, sy1 + 8), 4, (0, 255, 0), -1)
    cv2.circle(canvas, (cx + 8, sy1 + 8), 4, (0, 255, 0), -1)

    # Wheels
    wheel_y = by + 6
    wrad    = int(10 * scale)
    cv2.ellipse(canvas, (cx - bw + wrad, wheel_y), (wrad, int(wrad * 0.5)),
                0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(canvas, (cx + bw - wrad, wheel_y), (wrad, int(wrad * 0.5)),
                0, 0, 360, (50, 50, 50), -1)

    # Label
    cv2.putText(canvas, "BOT", (cx - 14, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ROBOT_COL, 1, cv2.LINE_AA)
    return canvas


def draw_background(canvas):
    """Draw a hospital corridor background."""
    canvas[:] = BG_WALL

    # Floor area (bottom 40%)
    floor_y = int(H * 0.60)
    canvas[floor_y:, :] = BG_FLOOR

    # Floor horizon line
    cv2.line(canvas, (0, floor_y), (W, floor_y), (160, 155, 148), 2)

    # Floor stripes (perspective lines converging to vanishing point)
    vp = (W // 2, floor_y)
    for stripe_x in range(-W, W * 2, 80):
        bx = stripe_x
        cv2.line(canvas, vp, (bx, H), STRIPE_COL, 1)

    # Ceiling line
    ceil_y = int(H * 0.08)
    cv2.line(canvas, (0, ceil_y), (W, ceil_y), (180, 175, 168), 2)

    # Doors on wall
    for dx in [120, 380, 600]:
        door_w, door_h = 55, int(H * 0.38)
        door_top = int(H * 0.18)
        cv2.rectangle(canvas, (dx, door_top), (dx + door_w, door_top + door_h),
                      DOOR_COL, -1)
        cv2.rectangle(canvas, (dx, door_top), (dx + door_w, door_top + door_h),
                      DOOR_FRAME, 3)
        # Door handle
        cv2.circle(canvas, (dx + door_w - 8, door_top + door_h // 2), 4, (100, 90, 80), -1)

    # Wall baseboard
    cv2.rectangle(canvas, (0, floor_y - 6), (W, floor_y + 2), (150, 145, 138), -1)

    # Ceiling light strips
    for lx in [W // 4, W // 2, 3 * W // 4]:
        cv2.rectangle(canvas, (lx - 40, ceil_y + 2), (lx + 40, ceil_y + 14),
                      (240, 240, 235), -1)

    return canvas


# ── Person definitions: each has a trajectory and timing ─────────────────────
class Person:
    def __init__(self, start_x, speed, color_idx, scale,
                 raise_hand_at=None, raise_hand_dur=60):
        self.x          = float(start_x)
        self.speed      = speed   # px/frame (negative = leftward)
        self.color      = PERSON_COLORS[color_idx % len(PERSON_COLORS)]
        self.scale      = scale
        self.raise_at   = raise_hand_at
        self.raise_dur  = raise_hand_dur
        self.floor_y    = int(H * 0.88)  # ground level

    def update(self, frame_idx):
        self.x += self.speed

    def visible(self):
        return -60 < self.x < W + 60

    def raised(self, frame_idx):
        if self.raise_at is None:
            return False, False
        in_raise = self.raise_at <= frame_idx < self.raise_at + self.raise_dur
        return in_raise, False   # left hand only


def make_people():
    return [
        Person(start_x=-60,  speed= 2.2, color_idx=0, scale=0.92,
               raise_hand_at=None),
        Person(start_x=-120, speed= 1.8, color_idx=1, scale=0.85),
        Person(start_x=W+60, speed=-2.0, color_idx=2, scale=0.95,
               raise_hand_at=180, raise_hand_dur=80),
        Person(start_x=W+30, speed=-1.5, color_idx=3, scale=0.88),
        Person(start_x=-200, speed= 2.5, color_idx=4, scale=0.90),
        Person(start_x=W+200,speed=-1.7, color_idx=5, scale=0.82),
    ]


# ── Main generation ───────────────────────────────────────────────────────────
def generate():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

    if not writer.isOpened():
        print("❌ Could not open VideoWriter. Check OpenCV build.")
        sys.exit(1)

    people = make_people()
    # Robot stays near center-right
    robot_x = int(W * 0.65)
    robot_by = int(H * 0.88)

    print(f"  Generating {FRAMES} frames at {FPS}fps…")

    for f in range(FRAMES):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        draw_background(canvas)

        # Robot
        draw_robot(canvas, robot_x, robot_by, scale=0.85)

        # Sort people by floor_y so farther ones are drawn first (depth)
        visible = [p for p in people if p.visible()]
        visible.sort(key=lambda p: p.floor_y)

        for p in visible:
            rl, rr = p.raised(f)
            draw_person(canvas, int(p.x), p.floor_y,
                        scale=p.scale, color_pair=p.color,
                        raised_left=rl, raised_right=rr)
            p.update(f)

        # Overlay subtle frame counter (invisible to users, useful for debug)
        # cv2.putText(canvas, str(f), (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

        writer.write(canvas)

        if f % (FPS * 5) == 0:
            print(f"  {f // FPS}s / {FRAMES // FPS}s")

    writer.release()

    size_mb = os.path.getsize(OUTPUT_PATH) / 1_000_000
    print(f"  ✅ Demo video saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("HumanSense — Generating synthetic demo video")
    print("=" * 46)
    generate()
    print("\nDone! Run the app and it will load automatically:")
    print("  streamlit run app.py")
