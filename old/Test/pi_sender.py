#!/usr/bin/env python3

'''
PySpectrometer2 Les Wright 2022
https://www.youtube.com/leslaboratory
https://github.com/leswright1977

This project is a follow on from: https://github.com/leswright1977/PySpectrometer 

This is a more advanced, but more flexible version of the original program. Tk Has been dropped as the GUI to allow fullscreen mode on Raspberry Pi systems and the iterface is designed to fit 800*480 screens, which seem to be a common resolutin for RPi LCD's, paving the way for the creation of a stand alone benchtop instrument.

Whats new:
Higher resolution (800px wide graph)
3 row pixel averaging of sensor data
Fullscreen option for the Spectrometer graph
3rd order polymonial fit of calibration data for accurate measurement.
Improved graph labelling
Labelled measurement cursors
Optional waterfall display for recording spectra changes over time.
Key Bindings for all operations

All old features have been kept, including peak hold, peak detect, Savitsky Golay filter, and the ability to save graphs as png and data as CSV.

For instructions please consult the readme!
'''
import sys
import struct
import serial
import cv2
import time
import numpy as np
from specFunctions import readcal,writecal
import base64
import argparse
import binascii



from picamera2 import Picamera2

frameWidth = 800
frameHeight = 600
unit = '/cm'

PORT = "/dev/ttyGS0"      # Change if needed
BAUD = 115200

START = b'\xAA\x55'
END = b'\x55\xAA'

ser = serial.Serial(PORT, BAUD, timeout=1)

# Dictionary mapping NumPy dtypes to a 1-byte integer code
DTYPE_TO_CODE = {
    np.dtype(np.uint8):   1,
    np.dtype(np.int16):   2,
    np.dtype(np.int32):   3,
    np.dtype(np.float32): 4,
    np.dtype(np.float64): 5,
}

# Reverse lookup dictionary for the receiver
CODE_TO_DTYPE = {code: dtype for dtype, code in DTYPE_TO_CODE.items()}


def build_packet(array, dtype=np.uint8):
    # Convert array to desired NumPy dtype
    arr = np.asarray(array, dtype=dtype)
    payload = arr.tobytes()

    # Get the 1-byte code for this dtype (default to uint8 = 1 if unknown)
    type_code = DTYPE_TO_CODE.get(arr.dtype, 1)

    # Calculate CRC over both type_code and payload to ensure data integrity
    crc_payload = struct.pack("B", type_code) + payload
    crc = binascii.crc_hqx(crc_payload, 0xFFFF)

    return (
        START +
        struct.pack("B", type_code) +          # 1 byte for data type ID
        struct.pack("<H", len(payload)) +      # 2 bytes for payload length
        payload +                              # Raw bytes
        struct.pack("<H", crc) +               # 2 bytes CRC
        END                                    # End marker
    )



#Change analog gain
#picam2.set_controls({"AnalogueGain": 10.0}) #Default 1
#picam2.set_controls({"Brightness": 0.2}) #Default 0 range -1.0 to +1.0
#picam2.set_controls({"Contrast": 1.8}) #Default 1 range 0.0-32.0


# calibrate = False

# intensity = [0] * frameWidth #array for intensity data...full of zeroes

# #Go grab the computed calibration data
# caldata = readcal(frameWidth)
# wavelengthData = caldata[0]
# calmsg1 = caldata[1]
# calmsg2 = caldata[2]
# calmsg3 = caldata[3]


intensity = np.zeros(frameWidth, dtype=np.uint8)  # Preallocate intensity array with zeros

# --- STARTUP HANDSHAKE ---
print("Waiting for startup configuration from PC...")
while True:
    # Block and wait for a command from the serial port
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    
    if line.startswith("START_GAIN:"):
        try:
            # Extract the number from the command string
            new_gain = float(line.split(":")[1])
            picamGain = min(max(new_gain, 1.0), 50.0)
            
            # Apply the new gain to the camera
            
            print(f"Startup gain set to: {picamGain}")
            
            # Send Acknowledgment back to PC
            ser.write(b"ACK\n")
            ser.flush()

            time.sleep(1)
            ser.reset_output_buffer()
            
            break  # Exit the setup loop and start streaming
            
        except ValueError:
            pass
# -------------------------

picam2 = Picamera2()
#need to spend more time at: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
#but this will do for now!
#min and max microseconds per frame gives framerate.
#30fps (33333, 33333)
#25fps (40000, 40000)
print("Picam started")


video_config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (frameWidth, frameHeight)}, controls={"FrameDurationLimits": (33333, 33333)})
picam2.configure(video_config)
picam2.start()
picam2.set_controls({"AnalogueGain": picamGain})


running = True
print("Starting continuous stream.")

try:
    while running:
        # Capture frame-by-frame and stream data continuously
        frame = picam2.capture_array()
        y = int((frameHeight / 2) - 40)  # origin of the vertical crop
        # y=200 	#origin of the vert crop
        x = 0  # origin of the horiz crop
        h = 80  # height of the crop
        w = frameWidth  # width of the crop
        cropped = frame[y:y + h, x:x + w]
        bwimage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        rows, cols = bwimage.shape
        halfway = int(rows / 2)

        # Now process the intensity data and display it 
        # intensity = []
        for i in range(cols):
            # data = bwimage[halfway,i] #pull the pixel data from the halfway mark
            # print(type(data)) #numpy.uint8
            # average the data of 3 rows of pixels:
            dataminus1 = bwimage[halfway - 1, i]
            datazero = bwimage[halfway, i]  # pull the pixel data from the halfway mark
            dataplus1 = bwimage[halfway + 1, i]
            data = (int(dataminus1) + int(datazero) + int(dataplus1)) / 3
            data = np.uint8(data)
            intensity[i] = int(data)

    
        builded_packet = build_packet(intensity,dtype=np.uint8)
        ser.write(builded_packet)
        ser.flush()
        time.sleep(0.2)  # small delay to ensure data is sent before next frame

except KeyboardInterrupt: #
    print("Manually interrupted via SSH (Ctrl+C).") #

finally: #
    # clean exit block: always guarantees hardware resources are released
    print("Cleaning up resources...") #
    try: #
        picam2.stop() #
    except Exception: #
        pass #
        
    try: #
        ser.close() #
    except Exception: #
        pass #
        
    print("Pi script exited cleanly.") #