"""Synthetic MediaPipe-like frames for the pose/ tests (no mediapipe needed).

Owner: C (Backend/LLM). A signer faces the camera in a 640x480 image: shoulders at y = 0.5,
signer's left shoulder (11) on the image right at x = 0.6, right shoulder (12) at x = 0.4.
Shoulder distance after the aspect fix is 0.2 * 640 / 480 = 0.2667, so with raise_y = 1.0 a wrist
counts as "up" when its image y < 0.5 + 0.2667. Use UP = 0.6 and DOWN = 0.95.
"""
import numpy as np

W, H = 640, 480
UP, DOWN = 0.6, 0.95
FPS_MS = 1000 / 30

_rng = np.random.default_rng(0)
_POSE = _rng.uniform(0.3, 0.7, size=(33, 2))
_POSE[0] = (0.5, 0.3)  # nose
_POSE[11], _POSE[12] = (0.6, 0.5), (0.4, 0.5)
_HAND = _rng.uniform(-0.04, 0.04, size=(21, 2))
_HAND[0] = (0.0, 0.0)  # wrist
_HAND[9] = (0.0, -0.05)  # middle finger base, straight up from the wrist


def fake_frame(t, left_y=DOWN, right_y=DOWN, hands=("left", "right"), shoulder_vis=0.99,
               shift=(0.0, 0.0), zoom=1.0, pose=True):
    """One contract Frame dict. left_y/right_y: image y of the signer's left/right wrist.
    hands: which hands MediaPipe "detects", in output order. shift/zoom move every point in the image."""
    wrists = {"left": (0.65, left_y), "right": (0.35, right_y)}
    p = _POSE.copy()
    p[15], p[16] = wrists["left"], wrists["right"]

    def place(xy):
        return (np.asarray(xy) - 0.5) * zoom + 0.5 + shift

    vis = np.full(33, 0.99)
    vis[[11, 12]] = shoulder_vis
    pose_list = [[*place(xy), 0.0, v] for xy, v in zip(p, vis)]
    hand_lists = [[[*place(np.add(wrists[side], d)), 0.0] for d in _HAND] for side in hands]
    return {"type": "frame", "t": t, "pose": pose_list if pose else None, "hands": hand_lists}


def sequence(*parts):
    """parts: (n_frames, frame kwargs) pairs -> frames at 30 fps."""
    frames = []
    for n, kwargs in parts:
        frames += [fake_frame((len(frames) + k) * FPS_MS, **kwargs) for k in range(n)]
    return frames


def raise_both(y=UP):
    return {"left_y": y, "right_y": y}
