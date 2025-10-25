import asyncio
import json
import time
import sys
import pickle
from collections import deque, Counter
from functools import partial

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy import signal

import websockets

# -------------------------
# --- CONFIG / CONSTANTS ---
# -------------------------
MODEL_PATH = "model.pkl"

WINDOW_LENGTH = 50
SAMPLES_PER_WINDOW = 128
OVERLAP_SAMPLES = 0
GYRO_BUFFER = 360
GYRO_CALIB_DURATION = 2.0
EEG_CONFIDENCE_THRESHOLD = 0.75

POSITION_THRESHOLD = 7.0
HOLD_TIME = 0.1
GYRO_PULL_MAX = 50
SLEEP_DELAY = 0

WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

# -------------------------
# --- FEATURE EXTRACTION ---
# -------------------------
class FeatureExtractor:
    def __init__(self, sfreq=256):
        self.sfreq = sfreq

    def extract_features(self, data):
        n_samples, n_channels = data.shape
        features = []
        for ch in range(n_channels):
            d = data[:, ch]
            features.extend([
                float(np.mean(d)),
                float(np.std(d)),
                float(np.max(d) - np.min(d)),
            ])
            freqs, psd = signal.welch(d, fs=self.sfreq, nperseg=min(64, n_samples))
            for band in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]:
                features.append(float(self._band_power(freqs, psd, *band)))
            # last three appended: delta, theta, alpha, beta, gamma -> alpha index = -3
            # safe indexing assuming bands appended in fixed order
            alpha = features[-3]
            beta = features[-2]
            theta = features[-4]
            features.append(float(beta / alpha) if alpha > 0 else 0.0)
            features.append(float(theta / alpha) if alpha > 0 else 0.0)
        return np.array(features)

    def _band_power(self, freqs, psd, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0

# -------------------------
# --- LSL SETUP HELPERS ---
# -------------------------
def setup_lsl_inlet(stream_type, timeout=3.0, buf=WINDOW_LENGTH):
    print(f"Looking for a {stream_type} stream...")
    streams = resolve_byprop("type", stream_type, 1, timeout)
    if not streams:
        raise RuntimeError(f"Unable to find {stream_type} stream after {timeout}s.")
    inlet = StreamInlet(streams[0], max_buflen=buf)
    print(f"{stream_type} stream found (buf={buf}s).")
    return inlet

def calibrate_gyro_blocking(inlet, duration=GYRO_CALIB_DURATION):
    """Blocking calibration used in executor."""
    print(f"Calibrating gyro... keep head still for {duration:.1f}s")
    start = time.time()
    collected = []
    while time.time() - start < duration:
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=100)
        if chunk:
            collected.extend(chunk)
        time.sleep(0.005)
    if not collected:
        raise RuntimeError("No gyro samples during calibration.")
    bias = np.mean(np.array(collected), axis=0)
    print(f"Calibration complete. Bias = {bias}")
    return bias

# -------------------------
# --- WEBSOCKET BROADCAST ---
# -------------------------
class Broadcaster:
    def __init__(self):
        self.connections = set()
        self._lock = asyncio.Lock()

    async def register(self, websocket):
        async with self._lock:
            self.connections.add(websocket)

    async def unregister(self, websocket):
        async with self._lock:
            self.connections.discard(websocket)

    async def broadcast(self, message_json):
        """Send message_json (string) to all connected websockets; remove dead ones."""
        async with self._lock:
            conns = list(self.connections)
        if not conns:
            return
        # send concurrently
        await asyncio.gather(*[self._safe_send(ws, message_json) for ws in conns])

    async def _safe_send(self, ws, msg):
        try:
            await ws.send(msg)
        except Exception:
            # Bad connection -> drop it
            async with self._lock:
                self.connections.discard(ws)

broadcaster = Broadcaster()

# -------------------------
# --- WEBSOCKET HANDLER ---
# -------------------------
async def ws_handler(websocket):
    await broadcaster.register(websocket)
    try:
        async for _ in websocket:
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        await broadcaster.unregister(websocket)

# -------------------------
# --- PROCESSING TASK (async) ---
# -------------------------
async def processing_loop(loop):
    # Load model (blocking) in executor
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        return

    # Setup streams (blocking) - run in executor to avoid blocking asyncio loop
    inlet_creator = partial(setup_lsl_inlet, "EEG", WINDOW_LENGTH)
    eeg_inlet = await loop.run_in_executor(None, inlet_creator)
    gyro_inlet = await loop.run_in_executor(None, partial(setup_lsl_inlet, "GYRO", GYRO_BUFFER))

    # Calibrate gyro (blocking) in executor
    bias = await loop.run_in_executor(None, partial(calibrate_gyro_blocking, gyro_inlet))

    # EEG info
    eeg_info = eeg_inlet.info()
    sfreq = int(eeg_info.nominal_srate())
    n_channels = eeg_info.channel_count()
    extractor = FeatureExtractor(sfreq=sfreq)

    eeg_window_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    prediction_buffer = deque(maxlen=5)
    prob_buffer = deque(maxlen=5)

    # GYRO state
    yaw_pos = 0.0
    pitch_pos = 0.0
    yaw_state = "center"
    pitch_state = "center"
    yaw_candidate = "center"
    pitch_candidate = "center"
    yaw_state_start = time.time()
    pitch_state_start = time.time()
    last_time = time.time()

    print("\nStreaming EEG + GYRO (websocket). Ctrl+C to quit.\n")

    try:
        while True:
            # --- EEG (non-blocking call run inside executor) ---
            eeg_pull = partial(eeg_inlet.pull_chunk, timeout=0.05, max_samples=SAMPLES_PER_WINDOW)
            eeg_samples, _ = await loop.run_in_executor(None, eeg_pull)

            eeg_status = "EEG: --"
            eeg_majority = 0
            eeg_conf_val = 0.0

            if eeg_samples:
                for s in eeg_samples:
                    eeg_window_buffer.append(s)

                if len(eeg_window_buffer) == SAMPLES_PER_WINDOW:
                    window = np.array(list(eeg_window_buffer))
                    # feature extraction is CPU-bound: run in executor
                    features = await loop.run_in_executor(None, partial(extractor.extract_features, window))

                    # predictions are often blocking -> run in executor
                    try:
                        pred = await loop.run_in_executor(None, partial(model.predict, np.array([features])))
                        pred = int(pred[0])
                        prob_vec = await loop.run_in_executor(None, partial(model.predict_proba, np.array([features])))
                        prob_vec = prob_vec[0]
                    except Exception:
                        pred = 0
                        prob_vec = np.ones(1)

                    prediction_buffer.append(int(pred))
                    prob_buffer.append(prob_vec)

                    if len(prediction_buffer) == prediction_buffer.maxlen:
                        counts = Counter(prediction_buffer)
                        majority = counts.most_common(1)[0][0]
                        # calc avg confidence for chosen class safely
                        confidences = []
                        for p in prob_buffer:
                            if len(p) > majority:
                                confidences.append(p[majority])
                            else:
                                confidences.append(0.0)
                        avg_conf = float(np.mean(confidences) if confidences else 0.0)
                        if avg_conf < EEG_CONFIDENCE_THRESHOLD:
                            majority = 0
                        eeg_majority = int(majority)
                        eeg_conf_val = float(avg_conf)
                        eeg_status = f"EEG: {eeg_majority} ({eeg_conf_val:.2f})"
                    else:
                        conf = float(prob_vec[int(pred)]) if len(prob_vec) > int(pred) else 1.0
                        if conf < EEG_CONFIDENCE_THRESHOLD:
                            pred = 0
                        eeg_majority = int(pred)
                        eeg_conf_val = conf
                        eeg_status = f"EEG: {int(pred)} ({conf:.2f})"


            gyro_pull = partial(gyro_inlet.pull_chunk, timeout=0.0, max_samples=GYRO_PULL_MAX)
            gyro_samples, _ = await loop.run_in_executor(None, gyro_pull)

            if gyro_samples:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                gyro_arr = np.array(gyro_samples)
                if gyro_arr.ndim == 1:
                    gyro_arr = gyro_arr[np.newaxis, :]
                if gyro_arr.shape[1] >= 3:
                    avg = np.mean(gyro_arr, axis=0) - bias
                    pitch_rate = float(avg[1])
                    yaw_rate = float(avg[2])

                    yaw_pos += yaw_rate * dt
                    pitch_pos += pitch_rate * dt

                    if abs(yaw_rate) < 1.0:
                        yaw_pos *= 0.95
                    if abs(pitch_rate) < 1.0:
                        pitch_pos *= 0.98

                    new_yaw_candidate = (
                        "left" if yaw_pos > POSITION_THRESHOLD else
                        "right" if yaw_pos < -POSITION_THRESHOLD else
                        "center"
                    )
                    if new_yaw_candidate != yaw_candidate:
                        yaw_candidate = new_yaw_candidate
                        yaw_state_start = current_time
                    elif current_time - yaw_state_start >= HOLD_TIME:
                        yaw_state = yaw_candidate

                    # pitch
                    new_pitch_candidate = (
                        "down" if pitch_pos > POSITION_THRESHOLD else
                        "up" if pitch_pos < -POSITION_THRESHOLD else
                        "center"
                    )
                    if new_pitch_candidate != pitch_candidate:
                        pitch_candidate = new_pitch_candidate
                        pitch_state_start = current_time
                    elif current_time - pitch_state_start >= HOLD_TIME:
                        pitch_state = pitch_candidate

            else:
                last_time = time.time()

            # --- Prepare payload and broadcast ---
            move_x = -1 if yaw_state == "left" else (1 if yaw_state == "right" else 0)
            move_y = 0  # always zero per your spec

            jump = pitch_state == "up"
            punch = eeg_majority == 3
            kick = eeg_majority == 2
            flash_step = eeg_majority == 1

            payload = {
                "move": [move_x, move_y],
                "jump": jump,
                "punch": punch,
                "kick": kick,
                "flash_step": flash_step
            }
            # broadcast as json string
            await broadcaster.broadcast(json.dumps(payload))

            # optional local console output (kept brief)
            print(
                f"\r{payload} | "
                f"yaw: {yaw_state:<6} ({yaw_pos:6.1f}°) | "
                f"pitch: {pitch_state:<6} ({pitch_pos:6.1f}°)"
            )

            await asyncio.sleep(SLEEP_DELAY)

    except asyncio.CancelledError:
        print("processing loop cancelled.")
        return

# -------------------------
# --- entrypoint / run ---
# -------------------------
async def main_async():
    loop = asyncio.get_running_loop()
    # start websocket server
    server = await websockets.serve(ws_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)
    print(f"websocket server listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    # run processing loop concurrently
    proc_task = asyncio.create_task(processing_loop(loop))

    # keep running until cancelled / ctrl-c
    try:
        await proc_task
    except KeyboardInterrupt:
        proc_task.cancel()
    finally:
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nstopped by user.")
