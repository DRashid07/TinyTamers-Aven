"""Cut one sign out of a frame stream by raised hands, live (Segmenter) and offline.

Owner: C (Backend/LLM). Parameters in pose/segment_config.json. See CONTRACT.md "Segmentation".
"""


class Segmenter:
    def __init__(self, w, h, cfg):
        raise NotImplementedError

    def push(self, frame_dict):
        """Return None | {"event": "start"} | {"event": "end", "frames": [...], "status": ...}."""
        raise NotImplementedError


def segment_offline(pose, hands, t, w, h, cfg):
    """Return (start_idx, end_idx) from the first start to the last end, or None."""
    raise NotImplementedError
