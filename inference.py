# Prathik Narsetty
# this is the main inference loop on my raspberry PI
import os
import sys
import time
import pathlib
import numpy as np
import cv2
import subprocess
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# setup
MODEL_PATH  = pathlib.Path("best_float16.tflite")   # path to tflite
ALERT_SOUND = "alert.wav"  # my Audio Alert FIle
LABELS = ["awake", "drowsy"]
THRESH = 0.60
NUM_THREADS = os.cpu_count() or 4


if not MODEL_PATH.exists():
    sys.exit(f"Model not found at {MODEL_PATH}")

## setup Model
interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=NUM_THREADS)
interpreter.allocate_tensors()
## use this information to reshape camera frames
inp  = interpreter.get_input_details()[0]
outp = interpreter.get_output_details()[0]
H, W = inp["shape"][1:3]

print("Model loaded:", MODEL_PATH.name, "| input shape", inp["shape"])

### Setting up camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "XRGB8888"}
)
picam2.configure(preview_config)
picam2.start()
print("Camera started – press q to quit")

fps = 0.0
t0, frame_count = time.time(), 0

try:
    while True:
        # grab frame, convert to BGR
        frame = picam2.capture_array()[:, :, :3]

        # prep for model
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        inp_img = (rgb.astype(np.float32) / 255.0)[np.newaxis]

        # inference
        interpreter.set_tensor(inp["index"], inp_img)
        interpreter.invoke()
        probs = interpreter.get_tensor(outp["index"])[0]
        awake, drowsy = float(probs[0]), float(probs[1])

        # update FPS every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / (time.time() - t0)
            t0 = time.time()

        # overlay results
        cv2.putText(frame, f"awake  {awake:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"drowsy {drowsy:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"FPS {fps:.1f}", (520, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # audio alert
        if drowsy > THRESH:
            cv2.putText(frame, "⚠ DROWSY ⚠", (180, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)
            # fire off aplay (non‑blocking)
            subprocess.Popen(["aplay", ALERT_SOUND],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

        # show preview & check for quit
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
