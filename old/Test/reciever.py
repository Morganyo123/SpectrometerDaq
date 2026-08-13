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


PORT = "COM4"      
BAUD = 115200


CODE_TO_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
}


def read_exact(n):
    data = b''
    while len(data) < n:
        chunk = ser2.read(n - len(data))
        if not chunk:
            raise TimeoutError("Timeout reading packet data.")
        data += chunk
    return data



def handshake_with_pi(desired_start_gain=30.0):
    

    setup_specmeter2()  # Perform the setup and launch the Pi sender script
    time.sleep(15)
    #Perform the Startup Handshake
    desired_start_gain = 30.0
    print(f"Sending startup gain configuration: {desired_start_gain}...")
    ser.write(f"START_GAIN:{desired_start_gain}\n".encode())
    print(ser.readline())  # Read the echoed command from the Pi
    
    time.sleep(0.2)

    ser.close()
    




def read_packet():
    # 1. Search for START marker
    timeout_start = time.time()
    while True:
        if time.time() - timeout_start > 5:  # 5 seconds timeout
            print("Error: Timeout waiting for START marker")
            
            handshake_with_pi(desired_start_gain=30.0)  # Perform the handshake with the Pi
            return None, None
        read = ser2.read(1)
        if read == b'\xAA':
            if ser2.read(1) == b'\x55':
                break

    # 2. Read Type Code (1 byte) & Payload Length (2 bytes)
    type_code = struct.unpack("B", read_exact(1))[0]
    length = struct.unpack("<H", read_exact(2))[0]

    # 3. Read Payload
    payload = read_exact(length)

    # 4. Read CRC (2 bytes) & End Marker (2 bytes)
    received_crc = struct.unpack("<H", read_exact(2))[0]
    end_marker = read_exact(2)

    if end_marker != END:
        print("Error: Bad end marker")
        return None ,None

    # 5. Verify CRC
    calculated_crc = binascii.crc_hqx(struct.pack("B", type_code) + payload, 0xFFFF)
    if calculated_crc != received_crc:
        print("Error: CRC Check Failed")
        return None,None

    # 6. Look up NumPy dtype and restore array automatically
    target_dtype = CODE_TO_DTYPE.get(type_code, np.uint8)
    return np.frombuffer(payload, dtype=target_dtype), target_dtype



def setup_specmeter2(username='bvandenb', password='Raspibvandenb',
                     path='/home/bvandenb/Documents/PySpectrometerMorgan',
                     exec='pi_sender.py'):
    
    ser.write(b'\n')
    time.sleep(5)
    resp = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
    print(f"Initial state: {resp!r}")

    if 'login:' in resp.lower():
        print("Login prompt detected, sending credentials...")
        ser.write(f'{username}\n'.encode())
        time.sleep(1.5)
        ser.read(ser.in_waiting)  # clear buffer
        ser.write(f'{password}\n'.encode())
        time.sleep(1.5)
        ser.read(ser.in_waiting)
    else:
        print("Already logged in — skipping credentials")

    ser.write(f"cd {path}\n".encode())
    time.sleep(1)
    print("cd:", ser.read(ser.in_waiting).decode('utf-8', errors='replace'))

    ser.write(f"./{exec}\n".encode())
    time.sleep(2)
    launch = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
    print(f"Launch response: {launch!r}")

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("start complete")

def setup_specmeter(username='bvandenb', password='Raspibvandenb', path='/home/bvandenb/Documents/PySpectrometerMorgan', exec='pi_sender.py'):
        ser.write(b'\n')
        ser.write(f'{username}\n'.encode())
        time.sleep(1)
        ser.write(f'{password}\n'.encode())
        time.sleep(1)
        ser.write(f"cd {path}\n".encode())
        time.sleep(1)
        print('cd complete')
        ser.write(f"./{exec}\n".encode())
        print('command complete')
        time.sleep(16)
        #print(ser.readlines())  # Read the first line of output from the command

        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print('start complete')


wavelength = np.loadtxt(r"Morgan Code\Test\calibrationData.csv", delimiter=",")  # Load wavelength data from file
intensity = np.zeros(800, dtype=np.uint8)

fig, ax = plt.subplots() 
line, = ax.plot(wavelength, intensity) 
ax.set_xlabel("Wavelength") 
ax.set_ylabel("Intensity") 
ax.set_title("Spectrometer Data") 
ax.set_ylim(0, 255)  # Set y-axis limits for uint8 intensity
ax.set_xlim(wavelength[0], wavelength[-1])  # Set x-axis limits based on wavelength range

def on_key_press(event): 
    if event.key == 'q': 
        # close the local plot
        print("Closing UI...")
        plt.close()  
        ser2.close()  # Close the serial port
# Connect the key press event to the Matplotlib figure
fig.canvas.mpl_connect('key_press_event', on_key_press) 

def anim_init(): 
    line.set_data(wavelength, intensity) 
    return (line,) 

def anim_update(frame): 
    global wavelength, intensity 
    try: #
        packet = read_packet() 
        if packet is not None: 
            data, dtype = packet 
            if dtype == np.uint8: 
                intensity = data 
    except Exception as e: 
        print("Error reading packet:", e) 


    # update plot data; guard lengths
    x = np.asarray(wavelength) 
    y = np.asarray(savgol_filter(intensity, 51, 3))  # Apply Savitzky-Golay filter
    
    # if lengths mismatch, truncate to min length
    if x.shape[0] != y.shape[0]: 
        m = min(x.shape[0], y.shape[0]) 
        x = x[:m] 
        y = y[:m] 

    line.set_data(x, y) 
    return (line,) 



ser = serial.Serial(PORT, BAUD, timeout=2)


handshake_with_pi(desired_start_gain=30.0)  # Perform the handshake with the Pi

try: 
    ser2 = serial.Serial(PORT, BAUD, timeout=2)  # Reopen the serial port for continuous reading
    # Start the continuous read loop 
    ani = animation.FuncAnimation(fig, anim_update, init_func=anim_init, interval=50, blit=False) 
    plt.show() 
    
except KeyboardInterrupt: 
    ser2.close() 

