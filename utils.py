# utils.py
import cv2
import numpy as np
from collections import deque

class AtariPreprocessor:
    def __init__(self):
        self.last_raw_frame = None

    def process(self, raw_frame):
        if self.last_raw_frame is not None:
            raw_frame = np.maximum(raw_frame, self.last_raw_frame)
        self.last_raw_frame = raw_frame.copy()

        gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        cropped = gray[34:194, :]
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self):
        self.frames.clear()

    def add_frame(self, processed_frame):
        if len(self.frames) == 0:
            for _ in range(self.num_frames):
                self.frames.append(processed_frame)
        else:
            self.frames.append(processed_frame)

    def get_state(self):
        return np.stack(self.frames, axis=0)