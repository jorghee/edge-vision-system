"""
Camera abstraction module.

Provides a unified interface to capture frames from:
- USB webcams via OpenCV (current backend)
- Raspberry Pi Camera via Picamera2 + libcamera

Usage:
    from camera import create_camera
    cam = create_camera(camera_index=0)
    cam.set_resolution(640, 480)
    cam.open()
    ret, frame = cam.read()   # numpy array BGR
    cam.release()
"""

import os
import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "auto")


class Camera(Protocol):
    """Protocol that every video source must implement."""

    def open(self) -> None: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
    def set_resolution(self, width: int, height: int) -> None: ...


# Backend: OpenCV (USB webcam)
class OpenCVCamera:
    """Captures frames from a USB webcam using OpenCV VideoCapture."""

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap = None
        self._width = 640
        self._height = 480

    def set_resolution(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open OpenCV camera (index={self._index})"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, 15)
        log.info(
            "[CAM] OpenCV: camera %d opened (%dx%d)",
            self._index, self._width, self._height,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("[CAM] OpenCV: camera released")


# Backend: Picamera2 (Raspberry Pi CSI Camera)
class PiCamera:
    """Captures frames from Raspberry Pi Camera via Picamera2 / libcamera."""

    def __init__(self) -> None:
        self._picam = None
        self._width = 640
        self._height = 480

    def set_resolution(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def open(self) -> None:
        from picamera2 import Picamera2  # type: ignore[import-untyped]

        self._picam = Picamera2()
        config = self._picam.create_still_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"},
        )
        self._picam.configure(config)
        self._picam.start()
        log.info(
            "[CAM] Picamera2: CSI camera opened (%dx%d)", self._width, self._height,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._picam is None:
            return False, None
        try:
            # Picamera2 returns RGB; convert to BGR for consistency with OpenCV
            import cv2

            frame_rgb = self._picam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception:
            log.warning("[CAM] Picamera2: error capturing frame",
                        exc_info=True)
            return False, None

    def release(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None
            log.info("[CAM] Picamera2: camera released")


# Factory
def _try_picamera() -> bool:
    """Attempts to import Picamera2 and detect a CSI camera."""
    try:
        from picamera2 import Picamera2  # type: ignore[import-untyped]

        cam = Picamera2()
        cam.close()
        return True
    except Exception:
        return False


def create_camera(camera_index: int = 0) -> Camera:
    """Creates the appropriate camera based on ``CAMERA_BACKEND``.

    Supported values:
        - ``"picamera2"``  — forces Raspberry Pi Camera backend.
        - ``"opencv"``     — forces USB webcam backend.
        - ``"auto"``       — attempts Picamera2 first, then OpenCV.
    """
    backend = CAMERA_BACKEND.lower()

    if backend == "picamera2":
        log.info("[CAM] Backend forced: Picamera2")
        return PiCamera()

    if backend == "opencv":
        log.info("[CAM] Backend forced: OpenCV (index=%d)", camera_index)
        return OpenCVCamera(camera_index)

    if backend == "auto":
        if _try_picamera():
            log.info("[CAM] Auto-detect: Picamera2 available")
            return PiCamera()
        log.info("[CAM] Auto-detect: using OpenCV (index=%d)", camera_index)
        return OpenCVCamera(camera_index)

    raise ValueError(f"Unknown camera backend: {backend!r}")
