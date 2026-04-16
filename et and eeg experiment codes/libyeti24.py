import os
import random
import itertools
import logging as log
from time import sleep, time

import numpy as np
from numpy import array as ary
import pandas as pd
from sklearn import linear_model as lm

import cv2 as cv
import pygame as pg
from pygame.draw import circle

# Small standalone test function for showing the calibration grid
def main():
    print("main")
    pg.init()
    pg.display.set_mode((800, 800))
    screen = pg.display.get_surface()
    cal = Calib(screen)
    print(str(cal.targets))
    print("active: " + str(cal.active))
    cal.draw()
    pg.display.update()
    sleep(2)

# Draw text on a pygame surface using relative screen coordinates
def draw_text(
    text: str,
    Surf: pg.Surface,
    rel_pos: tuple,
    Font: pg.font.Font,
    color=(0, 0, 0),
    center=False,
):
    surf_size = Surf.get_size()
    x, y = np.array(rel_pos) * np.array(surf_size)
    rendered_text = Font.render(text, True, color)
    box = rendered_text.get_rect()
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)
    Surf.blit(rendered_text, box)


class Stimulus:
    stim_dir = "Stimuli/"

    def __init__(self, entry):
        if isinstance(entry, pd.DataFrame):
            entry = entry.to_dict()
        self.file = entry["File"]
        self.path = os.path.join(self.stim_dir, self.file)
        self.original_size = ary((entry["width"], entry["height"]))
        self.size = self.original_size.copy()

    # Load a stimulus image and scale it to fit the current screen
    def load(self, surface: pg.Surface, scale=True):
        image = pg.image.load(self.path)
        self.surface = surface
        self.surf_size = ary(self.surface.get_size())
        if scale:
            self.scale = min(self.surf_size / self.original_size)
            scale_to = ary(self.original_size * self.scale).astype(int)
            self.image = pg.transform.smoothscale(image, scale_to)
            self.size = ary(self.image.get_size())
        else:
            self.scale = 1
            self.image = image
        self.pos = ary((self.surf_size - self.size) / 2).astype(int)

    def draw(self):
        self.surface.blit(self.image, self.pos)

    # Draw a blurred preview of the stimulus for the quick-calibration phase
    def draw_preview(self):
        blur = ary(self.surf_size / 4).astype("int")
        img = pg.surfarray.array3d(self.image)
        img = cv.blur(img, tuple(blur)).astype("uint8")
        img = pg.surfarray.make_surface(img)
        self.surface.blit(img, self.pos)

    def average_brightness(self):
        return pg.surfarray.array3d(self.image).mean()


class StimulusSet:
    # Load all stimuli listed in the stimulus csv file
    def __init__(self, path):
        self.table = pd.read_csv(path)
        self.Stimuli = []
        for _, row in self.table.iterrows():
            this_stim = Stimulus(row)
            self.Stimuli.append(this_stim)
        self.active = 0

    def n(self):
        return len(self.Stimuli)

    def remaining(self):
        return len(self.Stimuli) - self.active

    def next(self):
        if self.active < len(self.Stimuli):
            this_stim = self.Stimuli[self.active]
            self.active += 1
            return True, this_stim
        return False, None

    def reset(self):
        self.active = 0

    def pop(self):
        return self.Stimuli.pop()

    def shuffle(self, reset=True):
        if reset:
            self.reset()
        random.shuffle(self.Stimuli)

# Convert an OpenCV frame to a pygame surface for display
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = np.rot90(img)
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


class YETI24:
    frame = None
    new_frame = False
    connected = False
    cascade = False
    eye_detection = False
    eye_detected = False
    eye_frame_coords = (0, 0, 0, 0)
    eye_frame = []
    quad_bright = (0, 0, 0, 0)
    offsets = (0, 0)

    # Output columns for the eye-tracking csv file
    data_cols = (
        "Exp",
        "Part",
        "trial_index",
        "Stim",
        "time",
        "xL",
        "yL",
        "xL_pro",
        "yL_pro",
        "xR",
        "yR",
        "xR_pro",
        "yR_pro",
        "xF",
        "yF",
        "xF_pro",
        "yF_pro",
        "inside_stimulus",
    )

    def __init__(self, usb: int, surface: pg.Surface) -> None:
        self.connected = False
        self.surface = surface
        self.surf_size = self.surface.get_size()

        # Allow either one USB index or a tuple of two camera indices
        if isinstance(usb, (tuple, list)) and len(usb) == 2:
            self.usb_L, self.usb_R = int(usb[0]), int(usb[1])
        else:
            self.usb_L = int(usb)
            self.usb_R = int(usb) + 1
        
        # Set up video recording variables
        self.video_writer_L = None
        self.video_writer_R = None
        self.is_recording = False
        self.video_path_L = None
        self.video_path_R = None
        self.recording_dir = None
        self.recording_partnum = None
        self.recording_trialnum = None

        # Initialize current camera frames and eye regions
        self.new_frame = False
        self.frame_L = None
        self.frame_R = None
        self.eye_frame_L = None
        self.eye_frame_R = None

        self.eye_detected_L = False
        self.eye_detected_R = False
        self.eye_frame_coords_L = (0, 0, 0, 0)
        self.eye_frame_coords_R = (0, 0, 0, 0)

        self.frame = None
        self.eye_frame = None
        self.quad_bright = (0, 0, 0, 0, 0, 0, 0, 0)
        self.current_stim_original_size = None
        
        # Open the left and right eye cameras
        if os.name == "nt":
            self.device_L = cv.VideoCapture(self.usb_L, cv.CAP_DSHOW)
            self.device_R = cv.VideoCapture(self.usb_R, cv.CAP_DSHOW)
        else:
            self.device_L = cv.VideoCapture(self.usb_L)
            self.device_R = cv.VideoCapture(self.usb_R)

        try:
            target_width = 640
            target_height = 480

            #for name, cap in (("L", self.device_L), ("R", self.device_R)):
            #    cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
            #    cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)

            ok_L, frame_L = self.device_L.read()
            ok_R, frame_R = self.device_R.read()
            
            if not ok_L or frame_L is None or frame_L.size == 0:
                raise RuntimeError(f"Left camera failed to deliver a valid frame: {self.usb_L}")
            if not ok_R or frame_R is None or frame_R.size == 0:
                raise RuntimeError(f"Right camera failed to deliver a valid frame: {self.usb_R}")

            self.frame_L = frame_L
            self.frame_R = frame_R
            self.frame_size_L = (self.frame_L.shape[1], self.frame_L.shape[0])
            self.frame_size_R = (self.frame_R.shape[1], self.frame_R.shape[0])

            # webcam-reported fps is often unreliable; use fixed write fps
            self.fps = 30.0
            self.connected = True

        except Exception as e:
            log.error(f"Could not connect/init cameras L={self.usb_L}, R={self.usb_R}: {e}")
            self.connected = False
            return

        # Storage for calibration samples and recorded gaze data
        self.calib_data = np.zeros(shape=(0, 10))
        self.data = pd.DataFrame(columns=YETI24.data_cols)

    # Start recording the raw left and right camera videos for one trial
    def start_recording(self, partnum, trialnum, location="Videos") -> tuple:
        if not self.connected:
            raise RuntimeError("Cannot start recording: cameras are not connected.")

        if self.is_recording:
            log.warning("Recording already in progress. Stopping previous recording first.")
            self.stop_recording()

        if self.frame_L is None or self.frame_R is None:
            ok = self.update_frame()
            if not ok:
                raise RuntimeError("Cannot start recording: no valid frames available.")

        partnum = str(partnum)
        trialnum = str(trialnum)

        self.recording_dir = os.path.join(location, partnum, trialnum)
        os.makedirs(self.recording_dir, exist_ok=True)

        self.video_path_L = os.path.join(
            self.recording_dir, f"{partnum}_{trialnum}_Left.avi"
        )
        self.video_path_R = os.path.join(
            self.recording_dir, f"{partnum}_{trialnum}_Right.avi"
        )

        fourcc = cv.VideoWriter_fourcc(*"MJPG")

        hL, wL = self.frame_L.shape[:2]
        hR, wR = self.frame_R.shape[:2]
        self.frame_size_L = (wL, hL)
        self.frame_size_R = (wR, hR)

        self.video_writer_L = cv.VideoWriter(
            self.video_path_L, fourcc, self.fps, self.frame_size_L
        )
        self.video_writer_R = cv.VideoWriter(
            self.video_path_R, fourcc, self.fps, self.frame_size_R
        )

        if not self.video_writer_L.isOpened():
            self.video_writer_L = None
            raise RuntimeError(f"Failed to open left writer: {self.video_path_L}")

        if not self.video_writer_R.isOpened():
            if self.video_writer_L is not None:
                self.video_writer_L.release()
                self.video_writer_L = None
            self.video_writer_R = None
            raise RuntimeError(f"Failed to open right writer: {self.video_path_R}")

        self.recording_partnum = partnum
        self.recording_trialnum = trialnum
        self.is_recording = True

        log.info(f"Started recording: {self.video_path_L} | {self.video_path_R}")
        return self.video_path_L, self.video_path_R

    def stop_recording(self) -> tuple:
        if self.video_writer_L is not None:
            self.video_writer_L.release()
            self.video_writer_L = None

        if self.video_writer_R is not None:
            self.video_writer_R.release()
            self.video_writer_R = None

        was_recording = self.is_recording
        self.is_recording = False

        if was_recording:
            log.info(f"Stopped recording: {self.video_path_L} | {self.video_path_R}")

        return self.video_path_L, self.video_path_R

    def release(self):
        self.stop_recording()
        for obj_name in ("device_L", "device_R"):
            obj = getattr(self, obj_name, None)
            if obj is not None:
                obj.release()

    # Load the Haar cascade model used to detect the eye region in each camera frame
    def init_eye_detection(self, cascade_file: str):
        self.eye_detection = False
        self.cascade = cv.CascadeClassifier(cascade_file)
        self.eye_detection = True

    # Read a new frame from both cameras and optionally save it to the trial videos
    def update_frame(self) -> bool:
        self.new_frame = False

        ok_L, frame_L = self.device_L.read()
        ok_R, frame_R = self.device_R.read()

        valid_L = ok_L and frame_L is not None and frame_L.size > 0
        valid_R = ok_R and frame_R is not None and frame_R.size > 0

        if valid_L:
            self.frame_L = frame_L
            if self.is_recording and self.video_writer_L is not None:
                self.video_writer_L.write(self.frame_L)

        if valid_R:
            self.frame_R = frame_R
            if self.is_recording and self.video_writer_R is not None:
                self.video_writer_R.write(self.frame_R)

        self.new_frame = valid_L and valid_R

        if self.new_frame:
            try:
                L = getattr(self, "debug_L", self.frame_L)
                R = getattr(self, "debug_R", self.frame_R)
                h = min(L.shape[0], R.shape[0])
                self.frame = np.hstack([L[:h], R[:h]])
            except Exception:
                self.frame = self.frame_L

        return self.new_frame

    # Detect one eye region in each camera image
    def detect_eye(self) -> bool:
        self.eye_detected_L = False
        self.eye_detected_R = False

        if self.new_frame and self.eye_detection and self.cascade is not None:
            gray_L = cv.cvtColor(self.frame_L, cv.COLOR_BGR2GRAY)
            gray_R = cv.cvtColor(self.frame_R, cv.COLOR_BGR2GRAY)

            Eyes_L = self.cascade.detectMultiScale(
                gray_L, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            Eyes_R = self.cascade.detectMultiScale(
                gray_R, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

            self.debug_L = self.frame_L.copy()
            self.debug_R = self.frame_R.copy()

            for x, y, w, h in Eyes_L:
                cv.rectangle(self.debug_L, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for x, y, w, h in Eyes_R:
                cv.rectangle(self.debug_R, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(Eyes_L) == 1:
                self.eye_detected_L = True
                self.eye_frame_coords_L = Eyes_L[0]

            if len(Eyes_R) == 1:
                self.eye_detected_R = True
                self.eye_frame_coords_R = Eyes_R[0]

        self.eye_detected = self.eye_detected_L and self.eye_detected_R
        return self.eye_detected

    # Crop the detected eye regions and resize them to a fixed size for analysis
    def update_eye_frame(self):
        if not self.new_frame:
            return None

        if self.frame_L is None or self.frame_R is None:
            return None

        xL, yL, wL, hL = self.eye_frame_coords_L
        xR, yR, wR, hR = self.eye_frame_coords_R

        if wL <= 0 or hL <= 0 or wR <= 0 or hR <= 0:
            return None

        FIXED_EYE_SIZE = (64, 48)

        self.eye_frame_L = self.frame_L[yL:yL + hL, xL:xL + wL]
        self.eye_frame_R = self.frame_R[yR:yR + hR, xR:xR + wR]

        if self.eye_frame_L.size == 0 or self.eye_frame_R.size == 0:
            return None

        self.eye_frame_L = cv.cvtColor(self.eye_frame_L, cv.COLOR_BGR2GRAY)
        self.eye_frame_R = cv.cvtColor(self.eye_frame_R, cv.COLOR_BGR2GRAY)

        self.eye_frame_L = cv.resize(
            self.eye_frame_L, FIXED_EYE_SIZE, interpolation=cv.INTER_AREA
        )
        self.eye_frame_R = cv.resize(
            self.eye_frame_R, FIXED_EYE_SIZE, interpolation=cv.INTER_AREA
        )

        return self.eye_frame_L, self.eye_frame_R

    # Measure brightness in four quadrants of each eye image
    def update_quad_bright(self) -> tuple:
        if not self.new_frame:
            return self.quad_bright

        if self.eye_frame_L is None or self.eye_frame_R is None:
            return self.quad_bright

        def quad(img):
            h, w = img.shape
            h2, w2 = h // 2, w // 2
            b_NW = np.mean(img[0:h2, 0:w2])
            b_NE = np.mean(img[0:h2, w2:w])
            b_SW = np.mean(img[h2:h, 0:w2])
            b_SE = np.mean(img[h2:h, w2:w])
            return (b_NW, b_NE, b_SW, b_SE)

        qb_L = quad(self.eye_frame_L)
        qb_R = quad(self.eye_frame_R)
        self.quad_bright = qb_L + qb_R

        return self.quad_bright

    # Save one calibration sample: eye brightness values + known target position
    def record_calib_data(self, target_pos: tuple) -> ary:
        new_data = np.append(self.quad_bright, ary(target_pos))
        self.calib_data = np.append(self.calib_data, [new_data], axis=0)
        return new_data

    # Train one regression model per eye to predict gaze position from brightness features
    def train(self):
        X_L = self.calib_data[:, 0:4]
        X_R = self.calib_data[:, 4:8]
        Y = self.calib_data[:, 8:10]

        self.model_L = lm.LinearRegression().fit(X_L, Y)
        self.model_R = lm.LinearRegression().fit(X_R, Y)

        return self.model_L, self.model_R

    def update_offsets(self, target_pos: tuple) -> tuple:
        new_offsets = ary(target_pos) - ary(self.eye_raw)
        self.offsets = tuple(new_offsets)
        return self.offsets

    def reset_offsets(self) -> None:
        self.offsets = (0, 0)

    # Convert current eye brightness values into gaze coordinates on the screen
    def update_eye_pos(self) -> tuple:
        quad = ary(self.quad_bright)

        quad_L = quad[0:4].reshape(1, 4)
        quad_R = quad[4:8].reshape(1, 4)

        raw_L = self.model_L.predict(quad_L)[0, :]
        raw_R = self.model_R.predict(quad_R)[0, :]

        self.eye_raw_L = tuple(raw_L)
        self.eye_raw_R = tuple(raw_R)
        self.eye_raw = tuple((ary(self.eye_raw_L) + ary(self.eye_raw_R)) / 2.0)

        self.eye_pos_L = tuple(ary(self.eye_raw_L) + ary(self.offsets))
        self.eye_pos_R = tuple(ary(self.eye_raw_R) + ary(self.offsets))

        self.eye_pro_L = tuple(ary(self.eye_pos_L) / ary(self.surf_size))
        self.eye_pro_R = tuple(ary(self.eye_pos_R) / ary(self.surf_size))

        raw_fused = (ary(self.eye_pos_L) + ary(self.eye_pos_R)) / 2.0

        x = int(np.clip(raw_fused[0], 0, self.surf_size[0] - 1))
        y = int(np.clip(raw_fused[1], 0, self.surf_size[1] - 1))
        self.eye_pos = (x, y)

        self.eye_pro = (x / self.surf_size[0], y / self.surf_size[1])

        return self.eye_pos_L, self.eye_pos_R

    # Convert gaze coordinates from screen space to original stimulus space
    def update_eye_stim(self, Stim: Stimulus) -> tuple:
        offsets = ary(Stim.pos)
        scale = ary(Stim.scale)
        self.current_stim_original_size = ary(Stim.original_size)

        self.eye_stim_L = tuple((ary(self.eye_pos_L) - offsets) / scale)
        self.eye_stim_R = tuple((ary(self.eye_pos_R) - offsets) / scale)

        self.eye_pro_L = tuple(ary(self.eye_stim_L) / ary(Stim.size))
        self.eye_pro_R = tuple(ary(self.eye_stim_R) / ary(Stim.size))

        self.eye_stim = tuple((ary(self.eye_stim_L) + ary(self.eye_stim_R)) / 2.0)
        self.eye_pro = tuple(ary(self.eye_stim) / ary(Stim.size))

        return self.eye_stim_L, self.eye_stim_R

    # Save one row of eye-tracking output for the current trial
    def record(self, Exp_ID: str, Part_ID: str, trial_index: int, Stim_ID: str) -> pd.DataFrame:
        inside_stimulus = False
        if self.current_stim_original_size is not None and hasattr(self, "eye_stim"):
            inside_stimulus = (
                0 <= self.eye_stim[0] <= self.current_stim_original_size[0]
                and 0 <= self.eye_stim[1] <= self.current_stim_original_size[1]
            )

        new_data = pd.DataFrame(
            {
                "Exp": Exp_ID,
                "Part": Part_ID,
                "trial_index": trial_index,
                "Stim": Stim_ID,
                "time": time(),
                "xL": self.eye_stim_L[0],
                "yL": self.eye_stim_L[1],
                "xL_pro": self.eye_pro_L[0],
                "yL_pro": self.eye_pro_L[1],
                "xR": self.eye_stim_R[0],
                "yR": self.eye_stim_R[1],
                "xR_pro": self.eye_pro_R[0],
                "yR_pro": self.eye_pro_R[1],
                "xF": self.eye_stim[0],
                "yF": self.eye_stim[1],
                "xF_pro": self.eye_pro[0],
                "yF_pro": self.eye_pro[1],
                "inside_stimulus": inside_stimulus,
            },
            index=[0],
        )

        self.data = pd.concat([self.data, new_data], ignore_index=True)
        return new_data

    def reset_calib(self) -> None:
        self.calib_data = np.zeros(shape=(0, 10))
        for m in ("model_L", "model_R"):
            if hasattr(self, m):
                delattr(self, m)

    def reset_data(self) -> None:
        self.data = pd.DataFrame(columns=YETI24.data_cols)

    # Reset both calibration and recorded gaze data
    def reset(self) -> None:
        self.reset_calib()
        self.reset_data()

    # Draw the estimated gaze point on the screen for visual feedback/debugging
    def draw_follow(self, surface: pg.Surface, add_raw=False, add_stim=False) -> None:
        if not hasattr(self, "eye_pos"):
            return

        surf_w, surf_h = surface.get_size()

        circ_size = int(min(surf_w, surf_h) / 50)
        circ_stroke = int(min(surf_w, surf_h) / 200)

        x = int(np.clip(self.eye_pos[0], 0, surf_w - 1))
        y = int(np.clip(self.eye_pos[1], 0, surf_h - 1))
        circle(surface, (255, 0, 0), (x, y), circ_size, circ_stroke)

        if add_raw and hasattr(self, "eye_raw"):
            rx = int(np.clip(self.eye_raw[0], 0, surf_w - 1))
            ry = int(np.clip(self.eye_raw[1], 0, surf_h - 1))
            circle(surface, (0, 255, 0), (rx, ry), circ_size, circ_stroke)

        if add_stim and hasattr(self, "eye_stim"):
            sx = int(np.clip(self.eye_stim[0], 0, surf_w - 1))
            sy = int(np.clip(self.eye_stim[1], 0, surf_h - 1))
            circle(surface, (0, 0, 255), (sx, sy), circ_size, circ_stroke)


class Calib:
    color = (160, 160, 160)
    active_color = (255, 120, 0)
    radius = 20
    stroke = 10

    def __init__(self, surface: pg.Surface, pro_positions=(0.125, 0.5, 0.875)) -> None:
        self.surface = surface
        self.surface_size = ary(self.surface.get_size())
        self.pro_positions = ary(pro_positions)
        x_pos = self.pro_positions * self.surface_size[0]
        y_pos = self.pro_positions * self.surface_size[1]
        self.targets = ary(list(itertools.product(x_pos, y_pos)))
        self.active = 0

    def shuffle(self, reset=True):
        if reset:
            self.reset()
        np.random.shuffle(self.targets)

    def active_pos(self) -> int:
        return self.targets[self.active]

    def reset(self) -> None:
        self.active = 0

    def n(self) -> int:
        return len(self.targets[:, 0])

    def remaining(self) -> int:
        return self.n() - self.active - 1

    def next(self) -> tuple:
        if self.remaining():
            this_target = self.targets[self.active]
            self.active += 1
            return True, this_target
        return False, None

    # Draw the full calibration grid and highlight the active target
    def draw(self) -> None:
        index = 0
        for target in self.targets:
            pos = list(map(int, target))
            color = self.active_color if index == self.active else self.color
            index += 1
            circle(self.surface, color, pos, self.radius, self.stroke)