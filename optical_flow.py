"""
optical_flow.py — Relative Position Tracking via Optical Flow
=============================================================
Tracks instrument displacement from a reference frame (frame 0)
using Farneback dense optical flow. Each camera maintains its own
reference baseline — every subsequent frame computes delta from zero.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class ReferenceTracker:
    """
    Tracks relative displacement from a reference frame for a single camera.
    
    Frame 0 = reference zero point.
    Every subsequent frame → compute delta from that reference.
    Delta = relative position of instrument from starting point.
    """

    def __init__(self, camera_id: str = "default"):
        self.camera_id = camera_id
        self.reference_frame: Optional[np.ndarray] = None   # grayscale
        self.reference_color: Optional[np.ndarray] = None    # original BGR
        self.prev_gray: Optional[np.ndarray] = None
        self.cumulative_delta = np.array([0.0, 0.0])
        self.frame_count = 0

    def set_reference(self, frame: np.ndarray):
        """Store frame 0 as the zero-point baseline."""
        self.reference_color = frame.copy()
        self.reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = self.reference_frame.copy()
        self.cumulative_delta = np.array([0.0, 0.0])
        self.frame_count = 0

    def get_relative_position(self, current_frame: np.ndarray) -> Tuple[float, float]:
        """
        Compute (Δx, Δy) displacement from reference frame using Farneback optical flow.
        Returns mean displacement across the entire frame.
        """
        if self.reference_frame is None:
            raise ValueError(f"[{self.camera_id}] Reference frame not set. Call set_reference() first.")

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Flow from reference → current (absolute displacement from frame 0)
        flow = cv2.calcOpticalFlowFarneback(
            self.reference_frame, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        delta_x = float(np.mean(flow[..., 0]))
        delta_y = float(np.mean(flow[..., 1]))

        self.prev_gray = curr_gray
        self.frame_count += 1

        return (delta_x, delta_y)

    def get_frame_to_frame_delta(self, current_frame: np.ndarray) -> Tuple[float, float]:
        """
        Compute frame-to-frame displacement (not from reference, but from previous frame).
        Useful for velocity/jitter calculations.
        """
        if self.prev_gray is None:
            raise ValueError(f"[{self.camera_id}] No previous frame. Call set_reference() first.")

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        delta_x = float(np.mean(flow[..., 0]))
        delta_y = float(np.mean(flow[..., 1]))

        self.cumulative_delta += np.array([delta_x, delta_y])
        self.prev_gray = curr_gray
        self.frame_count += 1

        return (delta_x, delta_y)

    def get_dense_flow(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Returns full dense flow field from reference frame.
        Shape: (H, W, 2) — flow vectors at each pixel.
        """
        if self.reference_frame is None:
            raise ValueError(f"[{self.camera_id}] Reference frame not set.")

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.reference_frame, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def get_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        """Convert flow field to HSV color visualization."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @property
    def is_initialized(self) -> bool:
        return self.reference_frame is not None


class MultiCameraFlowTracker:
    """
    Manages ReferenceTracker instances for all cameras in the rig.
    Provides synchronized relative-position output across all views.
    """

    def __init__(self, camera_ids: list = None):
        self.trackers: Dict[str, ReferenceTracker] = {}
        if camera_ids:
            for cam_id in camera_ids:
                self.trackers[cam_id] = ReferenceTracker(cam_id)

    def initialize(self, camera_id: str, frame: np.ndarray):
        """Set reference (frame 0) for one camera."""
        if camera_id not in self.trackers:
            self.trackers[camera_id] = ReferenceTracker(camera_id)
        self.trackers[camera_id].set_reference(frame)

    def initialize_all(self, camera_frames: Dict[str, np.ndarray]):
        """Set reference for all cameras at once (synchronized frame 0)."""
        for cam_id, frame in camera_frames.items():
            self.initialize(cam_id, frame)

    def update(self, camera_id: str, frame: np.ndarray) -> Tuple[float, float]:
        """Get relative position delta for one camera."""
        if camera_id not in self.trackers:
            raise ValueError(f"Camera '{camera_id}' not registered.")
        return self.trackers[camera_id].get_relative_position(frame)

    def update_all(self, camera_frames: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Get relative position deltas for all cameras."""
        deltas = {}
        for cam_id, frame in camera_frames.items():
            if cam_id in self.trackers and self.trackers[cam_id].is_initialized:
                deltas[cam_id] = self.update(cam_id, frame)
        return deltas

    def get_all_deltas(self, camera_frames: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Alias for update_all — returns dict of {camera_id: (Δx, Δy)}."""
        return self.update_all(camera_frames)

    def get_frame_to_frame_deltas(self, camera_frames: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Get frame-to-frame deltas (not from reference) for all cameras."""
        deltas = {}
        for cam_id, frame in camera_frames.items():
            if cam_id in self.trackers and self.trackers[cam_id].is_initialized:
                deltas[cam_id] = self.trackers[cam_id].get_frame_to_frame_delta(frame)
        return deltas

    @property
    def camera_ids(self) -> list:
        return list(self.trackers.keys())

    @property
    def all_initialized(self) -> bool:
        return all(t.is_initialized for t in self.trackers.values())
