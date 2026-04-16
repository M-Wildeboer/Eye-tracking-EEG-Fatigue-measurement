import csv
import os
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import serial

# Regular expression used to recognize one complete EEG sample record
DEFAULT_RECORD_RE = re.compile(
    r"(\d+),(\d+),"
    r"(-?\d+\.\d{3}),(-?\d+\.\d{3}),(-?\d+\.\d{3}),(-?\d+\.\d{3}),"
    r"(-?\d+\.\d{3}),(-?\d+\.\d{3}),(-?\d+\.\d{3}),(-?\d+\.\d{3})"
)

# Simple result object returned after one EEG measurement period ends
@dataclass
class EEGMeasureResult:
    eeg_file: str
    event_file: str
    merged_file: Optional[str] = None
    sample_count: int = 0


class EEGAPI:
    """
    Simple EEG logger API for use alongside libyeti.

    Typical use:
        eeg = EEGAPI(port="COM5", baud=2_000_000)
        eeg.start_measure(part_id, trial_index)
        ...
        result = eeg.stop_measure(eye_df=Yet.data)

    Output per measurement:
        - raw EEG csv
        - event csv
        - optional merged eye+EEG csv
    """

    # Column names for the saved EEG data and event log files
    eeg_columns = [
        "pc_time",
        "timestamp",
        "dev",
        "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8",
    ]

    event_columns = [
        "pc_time",
        "part_id",
        "trial_index",
        "stimulus",
        "event",
        "note",
    ]

    def __init__(
        self,
        port: str,
        baud: int = 2_000_000,
        output_dir: str = "Data_EEG",
        part_id: int = 0,
        record_re: re.Pattern = DEFAULT_RECORD_RE,
        serial_timeout: float = 0.02,
        flush_every: int = 100,
    ) -> None:
        self.port = port
        self.baud = baud
        self.output_dir = Path(output_dir + "/" + part_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.record_re = record_re
        self.serial_timeout = serial_timeout
        self.flush_every = flush_every

        self.ser = None
        self.eeg_file_handle = None
        self.event_file_handle = None
        self.eeg_writer = None
        self.event_writer = None

        self.buffer = ""
        self.last_record = None
        self.sample_count = 0

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.current_part_id = None
        self.current_trial_index = None
        self.current_stimulus = ""
        self.current_measure_started = None

        self.current_eeg_file = None
        self.current_event_file = None
        self.current_merged_file = None

    # ---------------------------
    # Public API
    # ---------------------------

    def start_measure(self, part_id: str, trial_index: int, stimulus: str = "") -> None:
        """
        Start EEG logging for one measurement period.
        Creates fresh EEG/event files and starts background serial polling.
        """
        if self.running:
            raise RuntimeError("EEG measurement already running. Stop it first.")

        # Store information about the current participant, trial, and stimulus
        self.current_part_id = str(part_id)
        self.current_trial_index = int(trial_index)
        self.current_stimulus = str(stimulus)
        self.current_measure_started = time.time()
        self.sample_count = 0
        self.buffer = ""
        self.last_record = None
        self.current_merged_file = None

        # Create output file paths for this measurement period
        base = f"eeg_{self.current_part_id}_{self.current_trial_index}"
        self.current_eeg_file = str(self.output_dir / f"{base}_raw.csv")
        self.current_event_file = str(self.output_dir / f"{base}_events.csv")

        self.eeg_file_handle = open(self.current_eeg_file, "w", newline="", encoding="utf-8")
        self.event_file_handle = open(self.current_event_file, "w", newline="", encoding="utf-8")

        self.eeg_writer = csv.writer(self.eeg_file_handle)
        self.event_writer = csv.writer(self.event_file_handle)

        self.eeg_writer.writerow(self.eeg_columns)
        self.event_writer.writerow(self.event_columns)

        # Open the serial connection and start reading EEG data in the background
        self.ser = serial.Serial(self.port, self.baud, timeout=self.serial_timeout)

        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

        self.log_event("measure_start", self.current_part_id, self.current_trial_index, self.current_stimulus)

    def stop_measure(
        self,
        eye_df: Optional[pd.DataFrame] = None,
        merge: bool = True,
        eye_time_col: str = "time",
    ) -> EEGMeasureResult:
        """
        Stop EEG logging and optionally create a merged eye+EEG csv.

        eye_df:
            pass Yet.data here if you want a merged file
        merge:
            if True and eye_df is provided, creates merged csv
        """
        if not self.running:
            raise RuntimeError("EEG measurement is not running.")

        self.log_event("measure_stop", self.current_part_id, self.current_trial_index, self.current_stimulus)

        # Stop the background polling thread and close all open files/connections
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        if self.eeg_file_handle is not None:
            self.eeg_file_handle.flush()
            self.eeg_file_handle.close()
            self.eeg_file_handle = None

        if self.event_file_handle is not None:
            self.event_file_handle.flush()
            self.event_file_handle.close()
            self.event_file_handle = None

        if self.ser is not None:
            self.ser.close()
            self.ser = None

        merged_file = None
        merge = False
        if merge and eye_df is not None:
            merged_file = self.merge_with_eye_data(eye_df=eye_df, eye_time_col=eye_time_col)
            self.current_merged_file = merged_file

        return EEGMeasureResult(
            eeg_file=self.current_eeg_file,
            event_file=self.current_event_file,
            merged_file=merged_file,
            sample_count=self.sample_count,
        )

    def log_event(
        self,
        event: str,
        part_id: Optional[str] = None,
        trial_index: Optional[int] = None,
        stimulus: str = "",
        note: str = "",
    ) -> None:
        # Save one event marker to the event log file
        if self.event_writer is None:
            return

        row = [
            time.time(),
            self.current_part_id if part_id is None else str(part_id),
            self.current_trial_index if trial_index is None else int(trial_index),
            stimulus if stimulus else self.current_stimulus,
            event,
            note,
        ]
        with self.lock:
            self.event_writer.writerow(row)
            self.event_file_handle.flush()

    def merge_with_eye_data(self, eye_df: pd.DataFrame, eye_time_col: str = "time") -> str:
        """
        Create a merged csv by nearest timestamp.
        Does NOT change the original eye-tracking csv format.
        """
        if eye_time_col not in eye_df.columns:
            raise ValueError(f"Eye dataframe must contain column '{eye_time_col}'.")

        eeg_df = pd.read_csv(self.current_eeg_file)
        if eeg_df.empty:
            raise RuntimeError("EEG file is empty; cannot merge.")

        eye_copy = eye_df.copy()
        eeg_copy = eeg_df.copy()

        eye_copy = eye_copy.sort_values(eye_time_col).reset_index(drop=True)
        eeg_copy = eeg_copy.sort_values("pc_time").reset_index(drop=True)

        # Match each eye-tracking sample to the nearest EEG timestamp
        merged = pd.merge_asof(
            eye_copy,
            eeg_copy,
            left_on=eye_time_col,
            right_on="pc_time",
            direction="nearest",
        )

        merged["eeg_eye_time_delta"] = (merged[eye_time_col] - merged["pc_time"]).abs()

        merged_file = str(self.output_dir / f"eeg_{self.current_part_id}_{self.current_trial_index}_merged.csv")
        merged.to_csv(merged_file, index=False)
        return merged_file

    def close(self) -> None:
        """
        Safe cleanup helper.
        """
        if self.running:
            try:
                self.stop_measure(merge=False)
            except Exception:
                pass

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _poll_loop(self) -> None:
        # Keep reading serial data while measurement is running
        while self.running:
            try:
                self._poll_once()
            except Exception:
                # keep background logger alive rather than crash the experiment
                time.sleep(0.05)

    def _poll_once(self) -> None:
        # Read one block of EEG data from the serial connection and process it
        if self.ser is None:
            return

        raw = self.ser.read(4096)
        if not raw:
            return

        text = raw.decode("utf-8", errors="replace")
        self.buffer += text

        # Handle complete EEG lines when line breaks are present
        if "\n" in self.buffer or "\r" in self.buffer:
            parts = re.split(r"[\r\n]+", self.buffer)
            self.buffer = parts.pop()

            for line in parts:
                line = line.strip()
                if not line:
                    continue

                fields = line.split(",")
                if len(fields) == 10:
                    self._write_if_new(fields)
        else:
            # If no line breaks are found, try to detect EEG records with regex
            last_end = 0
            for match in self.record_re.finditer(self.buffer):
                fields = list(match.groups())
                self._write_if_new(fields)
                last_end = match.end()

            if last_end:
                self.buffer = self.buffer[last_end:]
            else:
                self.buffer = self.buffer[-2000:]

    def _write_if_new(self, fields) -> None:
        # Write a new EEG sample unless it is a duplicate of the previous one
        current_record = tuple(fields)
        if current_record == self.last_record:
            return

        row = [time.time()] + fields
        with self.lock:
            self.eeg_writer.writerow(row)
            self.sample_count += 1

            if self.sample_count % self.flush_every == 0:
                self.eeg_file_handle.flush()

        self.last_record = current_record