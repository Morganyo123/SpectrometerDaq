#!/usr/bin/env python3
import sys
import struct
import serial
import cv2
import time
import numpy as np
import binascii
from picamera2 import Picamera2

frameWidth = 800
frameHeight = 600

PORT = "/dev/ttyGS0"
BAUD = 115200

START = b'\xAA\x55'
END = b'\x55\xAA'

ser = serial.Serial(PORT, BAUD, timeout=1)

DTYPE_TO_CODE = {
    np.dtype(np.uint8):   1,
    np.dtype(np.int16):   2,
    np.dtype(np.int32):   3,
    np.dtype(np.float32): 4,
    np.dtype(np.float64): 5,
}

def build_packet(array, dtype=np.uint8):
    arr = np.asarray(array, dtype=dtype)
    payload = arr.tobytes()
    type_code = DTYPE_TO_CODE.get(arr.dtype, 1)

    crc_payload = struct.pack("B", type_code) + payload
    crc = binascii.crc_hqx(crc_payload, 0xFFFF)

    return (
        START +
        struct.pack("B", type_code) +
        struct.pack("<H", len(payload)) +
        payload +
        struct.pack("<H", crc) +
        END
    )

# --- STARTUP HANDSHAKE ---
print("Waiting for startup configuration from PC...")
picamGain = 40.0  # Default fallback gain

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith("START_GAIN:"):
            try:
                new_gain = float(line.split(":")[1])
                picamGain = min(max(new_gain, 1.0), 50.0)
                print(f"Startup gain set to: {picamGain}")
            except ValueError:
                pass
            
            # Send ACK and allow buffer to transmit fully (no aggressive reset)
            ser.write(b"ACK\n")
            ser.flush()
            time.sleep(0.5)
            break
    time.sleep(0.05)

# --- CAMERA SETUP ---
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"format": 'RGB888', "size": (frameWidth, frameHeight)}, 
    controls={"FrameDurationLimits": (33333, 33333)}
)
picam2.configure(video_config)
picam2.start()
picam2.set_controls({"AnalogueGain": picamGain})

intensity = np.zeros(frameWidth, dtype=np.uint8)

print("Starting continuous stream.")
try:
    while True:
        frame = picam2.capture_array()
        y = int((frameHeight / 2) - 40)
        cropped = frame[y:y + 80, 0:frameWidth]
        bwimage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        rows, cols = bwimage.shape
        halfway = int(rows / 2)

        for i in range(cols):
            dataminus1 = bwimage[halfway - 1, i]
            datazero = bwimage[halfway, i]
            dataplus1 = bwimage[halfway + 1, i]
            data = (int(dataminus1) + int(datazero) + int(dataplus1)) / 3
            intensity[i] = np.uint8(data)

        builded_packet = build_packet(intensity, dtype=np.uint8)
        ser.write(builded_packet)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    picam2.stop()
    ser.close()