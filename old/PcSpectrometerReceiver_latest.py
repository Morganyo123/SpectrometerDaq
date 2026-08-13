import serial
import struct
import binascii
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.signal import savgol_filter, find_peaks



class PiSerial():
    def __init__(self, port = "COM4", baud = 115200):
        self.CODE_TO_DTYPE = {
                                1: np.uint8,
                                2: np.int16,
                                3: np.int32,
                                4: np.float32,
                                5: np.float64,
                            }
        self.READY_TOKEN  = 'waiting for startup configuration'
        self.ACK_TOKENS   = ['startup gain set to', 'ack']
        self.STREAM_TOKEN = 'starting continuous stream'

        self.LOGIN_TOKEN    = 'login:'
        self.PASSWORD_TOKEN = 'password:'
        self.FAIL_TOKENS    = ['login incorrect', 'incorrect', 'timed out']
        self.SHELL_TOKENS   = ['$', '#']

        self.START = b'\xAA\x55'
        self.END   = b'\x55\xAA'

        self.PORT = port
        self.BAUD = baud
        self.MAX_PAYLOAD = 4096



        print("Checking for existing data stream...")
        if self.stream_detected(seconds=3):
            print("Data already streaming from the Pi - skipping handshake.")
        else:
            print("No live stream - checking console state...")
            if self.sender_already_running(30.0):
                print("Sender was already up - configured it.")
                if not self.stream_detected(seconds=5):
                    print("No bytes after configuring - running full handshake...")
                    self.bring_up_pi(desired_start_gain=30.0)
            else:
                print("Running full handshake...")
                self.bring_up_pi(desired_start_gain=30.0)

    def open_data(self):
        global ser
        self.ser = serial.Serial(self.PORT, self.BAUD, timeout=2)


    def close_data(self):
        global ser
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


    def wait_for(self, console, tokens, timeout=20):
        """Read until one of tokens appears in the accumulated buffer.
        Returns (found_token_or_None, buf)."""
        buf = ''
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = console.read(console.in_waiting or 1)
            if chunk:
                buf = (buf + chunk.decode('utf-8', errors='replace'))[-2000:]
                low = buf.lower()
                for t in tokens:
                    if t in low:
                        return t, buf
            time.sleep(0.1)
        return None, buf


    def classify_console(self, console, timeout=8):
        """Work out what is on the far end BEFORE typing anything at it.
        A bare newline is safe everywhere: at a getty it submits an empty username
        (recoverable via reset_to_login), at a shell it reprints the prompt, and the
        sender's readline loop ignores a blank line."""
        if console.in_waiting:
            console.read(console.in_waiting)
        console.write(b'\n')

        tokens = ([self.STREAM_TOKEN, self.READY_TOKEN, self.PASSWORD_TOKEN, self.LOGIN_TOKEN]
                + self.FAIL_TOKENS + self.SHELL_TOKENS)
        found, buf = self.wait_for(console, tokens, timeout=timeout)
        low = buf.lower()

        # The sender's own banners are the truth if present.
        if self.STREAM_TOKEN in low:
            state = 'streaming'
        elif self.READY_TOKEN in low:
            state = 'sender_ready'
        else:
            # Otherwise the LAST prompt in the buffer is the current state, so a
            # buffer like "Login incorrect ... login:" classifies as 'login'.
            marks = []
            if self.PASSWORD_TOKEN in low:
                marks.append((low.rfind(self.PASSWORD_TOKEN), 'password'))
            if self.LOGIN_TOKEN in low:
                marks.append((low.rfind(self.LOGIN_TOKEN), 'login'))
            for t in self.FAIL_TOKENS:
                if t in low:
                    marks.append((low.rfind(t), 'login'))
            for t in self.SHELL_TOKENS:
                if t in low:
                    marks.append((low.rfind(t), 'shell'))
            state = max(marks)[1] if marks else 'blank'

        print(f"Console state: {state} {buf!r}")
        return state, buf


    def reset_to_login(self, console, tries=3):
        """Get a stale/poisoned getty back to a clean login: prompt.
        An empty entry makes it fail fast and reprint the prompt."""
        buf = ''
        for _ in range(tries):
            console.write(b'\n')
            found, buf = self.wait_for(console, [self.LOGIN_TOKEN], timeout=15)
            if found:
                print("Console reset to a clean login prompt.")
                return True
        print(f"Could not reach a login prompt: {buf!r}")
        return False


    def stream_detected(self, seconds=3):
        """Open the port briefly and check whether framed packets are arriving."""

        self.seconds = seconds
        buf = b''
        deadline = time.time() + self.seconds
        try:
            probe = serial.Serial(self.PORT, self.BAUD, timeout=0.2)
            try:
                while time.time() < deadline:
                    chunk = probe.read(64)
                    if chunk:
                        buf = (buf + chunk)[-64:]
                        if self.START in buf:
                            return True
            finally:
                probe.close()
        except Exception as e:
            print(f"stream_detected: could not open port ({e})")
        return False


    def send_gain(self, console, gain):
        """Write START_GAIN and wait for the sender's ACK line.

        ONLY call this when classify_console() reported 'sender_ready' or 'blank'.
        At a login or shell prompt the text is entered as a username/password,
        which never ACKs and leaves the console in a broken state."""
        # Drain any pending RX bytes before writing
        if console.in_waiting:
            console.read(console.in_waiting)

        cmd = f"START_GAIN:{gain}\n"
        console.write(cmd.encode())

        found, buf = self.wait_for(console, self.ACK_TOKENS, timeout=15)

        if found:
            # Remove the echoed command (tty echo) so we only accept the real reply
            echo_stripped = buf.lower().replace(cmd.lower().strip(), '')
            still_found = any(t in echo_stripped for t in self.ACK_TOKENS)
            if still_found:
                print(f"Gain acknowledged: {buf.strip()!r}")
                return True
            print(f"Only echo found, no real ACK. Buffer: {buf!r}")
            return False

        print(f"No gain ACK. Buffer: {buf!r}")
        return False


    def wait_for_stream_banner(self, console, timeout=40):
        """Wait on the already-open console for STREAM_TOKEN.
        Call after the gain ACK - this covers the Picamera2 init delay."""
        found, buf = self.wait_for(console, [self.STREAM_TOKEN], timeout=timeout)
        if found:
            print("Sender is streaming - verifying bytes...")
            return True
        print("Warning: stream banner not seen - will probe for bytes anyway.")
        return False


    def sender_already_running(self, desired_start_gain=30.0):
        """Sender already up but unconfigured (console blank / at READY banner).
        Classifies FIRST so it never types a gain at a login or shell prompt."""
        try:
            console = serial.Serial(self.PORT, self.BAUD, timeout=2)
            try:
                state, _ = self.classify_console(console)
                if state == 'streaming':
                    return True
                if state not in ('sender_ready', 'blank'):
                    print(f"Console is at '{state}' - needs a login, not a gain.")
                    return False
                if self.send_gain(console, desired_start_gain):
                    self.wait_for_stream_banner(console, timeout=40)
                    return True
                return False
            finally:
                console.close()
        except Exception as e:
            print(f"sender_already_running: {e}")
            return False


    def login_and_launch(self, console, state, username='bvandenb', password='Raspibvandenb',
                        path='/home/bvandenb/Documents/PySpectrometerMorgan',
                        script='pi_sender.py'):
        """Log in (if needed) and launch the sender. `state` comes from
        classify_console() - this function never re-probes the console."""
        if state == 'password':
            print("Stale password prompt - resetting console...")
            if not self.reset_to_login(console):
                return False
            state = 'login'

        if state == 'login':
            print("Login prompt detected, sending credentials...")
            console.write(f'{username}\n'.encode())
            found, buf = self.wait_for(console, [self.PASSWORD_TOKEN], timeout=20)
            if not found:
                print(f"No password prompt: {buf!r}")
                return False
            console.write(f'{password}\n'.encode())
            found, buf = self.wait_for(console, self.SHELL_TOKENS + [self.LOGIN_TOKEN] + self.FAIL_TOKENS,
                                timeout=25)
            if found is None or found == self.LOGIN_TOKEN or found in self.FAIL_TOKENS:
                print(f"Login failed: {buf!r}")
                return False
            print("Logged in.")
        elif state == 'shell':
            print("Shell prompt detected - already logged in.")
        else:
            # 'sender_ready' / 'blank' - a program owns the console, don't type creds
            return 'unknown'

        console.write(f"cd {path}\n".encode())
        self.wait_for(console, self.SHELL_TOKENS, timeout=15)

        console.write(f"./{script}\n".encode())

        # The sender prints its READY banner immediately, so this is deterministic
        found, buf = self.wait_for(console, [self.READY_TOKEN], timeout=30)
        low = buf.lower()
        if self.LOGIN_TOKEN in low or 'command not found' in low or 'no such file' in low:
            print(f"Launch failed: {buf!r}")
            return False
        if found:
            print(f"Sender ready: {buf.strip()!r}")
        else:
            print("Ready banner not seen after launch - falling back to 15 s wait.")
            time.sleep(15)
        return True


    def bring_up_pi(self, desired_start_gain=30.0, max_attempts=None):
        attempt = 0
        while True:
            attempt += 1
            if max_attempts is not None and attempt > max_attempts:
                print(f"Giving up after {max_attempts} attempt(s).")
                return False
            print(f"Handshake attempt {attempt}...")

            gained = False
            banner_seen = False
            streaming = False
            try:
                console = serial.Serial(self.PORT, self.BAUD, timeout=2)
                try:
                    state, _ = self.classify_console(console)

                    if state == 'streaming':
                        print("Pi is already streaming.")
                        streaming = True
                    elif state in ('sender_ready', 'blank'):
                        # Sender owns the console and is waiting for its config
                        gained = self.send_gain(console, desired_start_gain)
                    else:
                        ok = self.login_and_launch(console, state)
                        if ok is False:
                            raise RuntimeError("login_and_launch failed")
                        gained = self.send_gain(console, desired_start_gain)

                    if gained:
                        # Covers Picamera2 init before we let go of the port
                        banner_seen = self.wait_for_stream_banner(console, timeout=40)
                finally:
                    # Console MUST be closed before we probe the data stream
                    console.close()
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(2)
                continue

            if streaming:
                return True

            if not gained:
                print(f"Attempt {attempt}: gain not acknowledged, retrying...")
                time.sleep(2)
                continue

            if not banner_seen:
                print("Stream banner missed - short settle before byte probe.")

            time.sleep(0.5)
            if self.stream_detected(seconds=5):
                print("Handshake successful - data stream is live.")
                return True
            print(f"Attempt {attempt}: no data bytes detected, retrying...")
            time.sleep(2)


    def read_exact(self, n):
        data = b''
        while len(data) < n:
            chunk = self.ser.read(n - len(data))
            if not chunk:
                raise TimeoutError("Timeout reading packet data.")
            data += chunk
        return data


    def read_packet(self):
        timeout_start = time.time()
        while True:
            if time.time() - timeout_start > 5:
                print("Timeout waiting for START marker - attempting recovery...")
                self.close_data()
                if not self.bring_up_pi(max_attempts=3):
                    print("Recovery failed - is the Pi alive?")
                self.open_data()
                return None, None
            byte = self.ser.read(1)
            if byte == b'\xAA' and self.ser.read(1) == b'\x55':
                break

        type_code = struct.unpack("B", self.read_exact(1))[0]
        length    = struct.unpack("<H", self.read_exact(2))[0]
        if length > self.MAX_PAYLOAD:
            print(f"Error: bogus length {length}, resyncing")
            return None, None
        payload      = self.read_exact(length)
        received_crc = struct.unpack("<H", self.read_exact(2))[0]
        end_marker   = self.read_exact(2)

        if end_marker != self.END:
            print("Error: Bad end marker")
            return None, None

        calculated_crc = binascii.crc_hqx(struct.pack("B", type_code) + payload, 0xFFFF)
        if calculated_crc != received_crc:
            print("Error: CRC check failed")
            return None, None

        target_dtype = self.CODE_TO_DTYPE.get(type_code, np.uint8)
        return np.frombuffer(payload, dtype=target_dtype), target_dtype


wavelength = np.loadtxt(r"Morgan Code\Test\calibrationData.csv", delimiter=",")
intensity  = np.zeros(800, dtype=np.uint8)

fig, ax = plt.subplots()
line, = ax.plot(wavelength, intensity)
ax.set_xlabel("Wavelength")
ax.set_ylabel("Intensity")
ax.set_title("Spectrometer Data")
ax.set_ylim(0, 255)
ax.set_xlim(wavelength[0], wavelength[-1])


def on_key_press(event):
    if event.key == 'q':
        print("Closing UI...")
        serialPi.close_data()
        plt.close()


fig.canvas.mpl_connect('key_press_event', on_key_press)


def anim_init():
    line.set_data(wavelength, intensity)
    return (line,)


def anim_update(frame):
    global intensity
    
    try:
        data, dtype = serialPi.read_packet()
        if data is not None and dtype == np.uint8:
            intensity = data
    except Exception as e:
        print("Error reading packet:", e)

    x = np.asarray(wavelength)
    if len(intensity) >= 51:
        y = np.asarray(savgol_filter(intensity, 51, 3))
    else:
        y = np.asarray(intensity)
    if x.shape[0] != y.shape[0]:
        m = min(x.shape[0], y.shape[0])
        x, y = x[:m], y[:m]

    # Clear previous annotations so they do not accumulate over frames
    for txt in ax.texts[:]:
        txt.remove()

    peaks, _ = find_peaks(y, height=10,  distance=5, threshold=5, prominence=5)  # Example: find peaks with height > 10
    if len(peaks) > 0:
        for peak in peaks:
            ax.annotate(f'Peak: {y[peak]:.1f}', xy=(x[peak], y[peak]), xytext=(x[peak], y[peak]+10),
                        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=8)
    line.set_data(x, y)
    return (line,)


serialPi = PiSerial()

try:
    serialPi.open_data()
    ani = animation.FuncAnimation(fig, anim_update, init_func=anim_init,
                                  interval=50, blit=False, cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    print("Interrupted.")
finally:
    serialPi.close_data()

