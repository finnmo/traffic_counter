# traffic_counter/utils.py

import cv2
import numpy as np
from typing import Tuple  # Add this line

def draw_semi_transparent_rectangle(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.2
) -> np.ndarray:
    """
    Draws a semi-transparent rectangle on the frame.

    Args:
        frame (np.ndarray): The image frame.
        top_left (Tuple[int, int]): Top-left corner of the rectangle.
        bottom_right (Tuple[int, int]): Bottom-right corner of the rectangle.
        color (Tuple[int, int, int], optional): Color of the rectangle in BGR. Defaults to (0, 255, 255).
        alpha (float, optional): Transparency factor. Defaults to 0.2.

    Returns:
        np.ndarray: The frame with the semi-transparent rectangle.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
