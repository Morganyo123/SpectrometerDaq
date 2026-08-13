import serial
import struct
import binascii
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.signal import savgol_filter

START = b'\xAA\x55'
END = b'\x55\xAA'

PORT = "COM4"  # Update to match your COM port
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

CODE_TO_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
}

def read_exact(ser, n):
    data = b''
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise TimeoutError("Timeout reading packet data.")
        data += chunk
    return data

def read_packet(ser):
    # 1. Search for START marker
    while True:
        if ser.read(1) == b'\xAA':
            if ser.read(1) == b'\x55':
                break

    # 2. Read Type Code (1 byte) & Payload Length (2 bytes)
    header = read_exact(ser, 3)
    type_code = header[0]
    length = struct.unpack("<H", header[1:3])[0]

    # Sanity check payload length
    if length > 2000:
        ser.reset_input_buffer()
        return None

    # 3. Read Payload, CRC, and End Marker
    payload = read_exact(ser, length)
    received_crc = struct.unpack("<H", read_exact(ser, 2))[0]
    end_marker = read_exact(ser, 2)

    # 4. Check End Marker; if invalid, flush buffer to force resynchronization
    if end_marker != END:
        print("Bad end marker detected - flushing buffer to resync...")
        ser.reset_input_buffer()
        return None

    # 5. Verify CRC
    calculated_crc = binascii.crc_hqx(struct.pack("B", type_code) + payload, 0xFFFF)
    if calculated_crc != received_crc:
        print("CRC Mismatch - flushing buffer to resync...")
        ser.reset_input_buffer()
        return None

    target_dtype = CODE_TO_DTYPE.get(type_code, np.uint8)
    return np.frombuffer(payload, dtype=target_dtype), target_dtype


# --- HANDSHAKE PHASE ---
desired_start_gain = 40.0
print(f"Sending startup gain configuration: {desired_start_gain}...")

# Clear existing buffers
ser.reset_input_buffer()
ser.reset_output_buffer()

# Send gain command until ACK received
while True:
    ser.write(f"START_GAIN:{desired_start_gain}\n".encode())
    ser.flush()
    time.sleep(0.2)
    
    if ser.in_waiting > 0:
        resp = ser.readline().decode('utf-8', errors='ignore').strip()
        if "ACK" in resp:
            print("Handshake successful!")
            break

# Pause briefly and clear out ALL leftover text residue from the serial buffer
time.sleep(0.5)
ser.reset_input_buffer()
print("Serial buffer purged. Starting plot streaming...")


# --- MATPLOTLIB ANIMATION SETUP ---
wavelength = np.array([380.0 + i * 0.46 for i in range(800)])  # 800 sample wavelength map
intensity = np.zeros(800, dtype=np.uint8)

fig, ax = plt.subplots()
line, = ax.plot(wavelength, intensity)
ax.set_xlabel("Wavelength")
ax.set_ylabel("Intensity")
ax.set_title("Spectrometer Data")
ax.set_ylim(0, 255)
ax.set_xlim(wavelength[0], wavelength[-1])

def anim_init():
    line.set_data(wavelength, intensity)
    return (line,)

def anim_update(frame):
    global intensity
    try:
        packet = read_packet(ser)
        if packet is not None:
            data, dtype = packet
            if dtype == np.uint8 and len(data) == len(wavelength):
                intensity = data
    except Exception as e:
        pass

    y = savgol_filter(intensity, 51, 3)
    line.set_data(wavelength, y)
    return (line,)

try:
    ani = animation.FuncAnimation(fig, anim_update, init_func=anim_init, interval=30, blit=False)
    plt.show()
finally:
    ser.close()