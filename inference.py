from picamera2 import Picamera2
import cv2, time, numpy as np, pathlib, os, sys
from tflite_runtime.interpreter import Interpreter       # tflite‑runtime 2.14

# ── SETTINGS ────────────────────────────────────────────────────────────────
MODEL_PATH = pathlib.Path("best_float16.tflite")          # adjust if needed
LABELS     = ["awake", "drowsy"]
THRESH     = 0.70                                         # trigger level
NUM_THREADS = os.cpu_count() or 4                         # Pi 4 → 4

# ── LOAD TFLITE ─────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    sys.exit(f"✗ model not found: {MODEL_PATH}")

interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=NUM_THREADS)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
H, W = inp["shape"][1:3]
print("✓ model loaded:", MODEL_PATH.name, "| input", inp["shape"])

# ── CAMERA SETUP ────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}))
picam2.start()
print("Camera started, press q to quit")

# ── MAIN LOOP ───────────────────────────────────────────────────────────────
t0, frame_count = time.time(), 0
try:
    while True:
        frame = picam2.capture_array()                  # 640×480 BGRA
        frame = frame[:, :, :3]                         # drop alpha → BGR
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb   = cv2.resize(rgb, (W, H), cv2.INTER_AREA)
        inp_img = (rgb.astype(np.float32) / 255.0)[np.newaxis]

        interpreter.set_tensor(inp["index"], inp_img)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"])[0]   # [awake, drowsy]
        awake, drowsy = probs

        # FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - t0)
            t0 = time.time()

        # Overlay text
        cv2.putText(frame, f"awake {awake:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"drowsy {drowsy:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"FPS {fps:.1f}", (520, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if drowsy > THRESH:
            cv2.putText(frame, "DROWSY!", (220, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)
            # os.system("aplay alert.wav")  # uncomment to play a wav alarm

        cv2.imshow("PiCam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop(); picam2.close()
    cv2.destroyAllWindows()
