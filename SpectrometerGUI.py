import os
import sys
import csv
import time
import struct
import binascii
import serial
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QPlainTextEdit, QGridLayout, QSizePolicy, QLineEdit,
                             QInputDialog,QMessageBox)

classes = ['logs', 'livePlotter', 'PiSerial', 'specGUI']


class logs(QWidget):
    def __init__(self, title='Logs', parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setStyleSheet('font-weight:bold;Padding:4px;')
        layout.addWidget(title_label)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(1000)
        self.text.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 6px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
            """
        )
        layout.addWidget(self.text)

    def add_message(self, text, show_time=True):
        if show_time:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {text}"
        else:
            line = text

        self.text.appendPlainText(line)
        scrollbar = self.text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Repaint now so handshake messages appear while the port is blocking
        QApplication.processEvents()

    def clear_messages(self):
        self.text.clear()


class livePlotter(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3), facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid()
        for spine in self.ax.spines.values():
            spine.set_color('white')

        self._width = 50
        self._mindist = 50

        self.x = []
        self.y = []
        self.line, = self.ax.plot(self.x, self.y, 'r-')
        self.p = self.ax.plot([], [], 'wo')[0]
        self.ax.set_xlabel("Wavelength [nm]")
        self.ax.set_ylabel("Intensity")
        self.ax.set_ylim(0, 255)
    def update_plot(self, x, y, show_peaks):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # Truncate to the shorter of the two so a short packet cannot break the plot
        if self.x.shape[0] != self.y.shape[0]:
            m = min(self.x.shape[0], self.y.shape[0])
            self.x, self.y = self.x[:m], self.y[:m]
        if self.x.size == 0:
            return

        self.line.set_data(self.x, self.y)


        for txt in self.ax.texts[:]:
                        txt.remove()

        if show_peaks:
            self.peakFinder()
        else:
            self.p.set_data([], [])

        self.ax.set_xlim(np.min(self.x), np.max(self.x))
        
        
        self.draw()

    def peakFinder(self):
        peaks, _ = find_peaks(self.y, distance=self._mindist, width=self._width,height=10)
        self.p.set_data(self.x[peaks], self.y[peaks])
        for peak in peaks:
            self.ax.annotate(f'{self.x[peak]:.1f} nm', xy=(self.x[peak], self.y[peak]), xytext=(self.x[peak], self.y[peak]+5),
                         fontsize=10,color="white")  
                  


class PiSerial():
    def __init__(self, port="COM4", baud=115200, logger=print, autostart=True,default_start_gain=30):
        self.CODE_TO_DTYPE = {
                                1: np.uint8,
                                2: np.int16,
                                3: np.int32,
                                4: np.float32,
                                5: np.float64,
                            }
        #self.default_start_gain = default_start_gain
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

        self.ser = None          # no data connection open yet
        self.log = logger        # GUI passes log_box.add_message here

        if autostart:
            self.connect(default_start_gain)

    def connect(self, default_start_gain):
        """Get the Pi streaming. Was the body of __init__ - split out so the GUI
        can run it after the window is up and route the prints to the log box."""
        self.log("Checking for existing data stream...")
        if self.stream_detected(seconds=3):
            self.log("Data already streaming from the Pi - skipping handshake. ")
            
            return True

        self.log("No live stream - checking console state...")

        user_start_gain = window.gain_popup()

        if user_start_gain is not None:
            default_start_gain = user_start_gain
        
        if self.sender_already_running(default_start_gain):
            self.log("Sender was already up - configured it.")
            if self.stream_detected(seconds=5):
                return True
            self.log("No bytes after configuring - running full handshake...")
            return self.bring_up_pi(default_start_gain=default_start_gain)

        self.log("Running full handshake...")
        return self.bring_up_pi(default_start_gain=default_start_gain)

    def open_data(self):
        self.ser = serial.Serial(self.PORT, self.BAUD, timeout=2)

    def close_data(self):
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

        self.log(f"Console state: {state}")
        return state, buf

    def reset_to_login(self, console, tries=3):
        """Get a stale/poisoned getty back to a clean login: prompt.
        An empty entry makes it fail fast and reprint the prompt."""
        buf = ''
        for _ in range(tries):
            console.write(b'\n')
            found, buf = self.wait_for(console, [self.LOGIN_TOKEN], timeout=15)
            if found:
                self.log("Console reset to a clean login prompt.")
                return True
        self.log(f"Could not reach a login prompt: {buf!r}")
        return False

    def stream_detected(self, seconds=3):
        """Open the port briefly and check whether framed packets are arriving."""
        buf = b''
        deadline = time.time() + seconds
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
            self.log(f"stream_detected: could not open port ({e})")
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
                self.log(f"Gain acknowledged: {buf.strip()!r}")
                return True
            self.log(f"Only echo found, no real ACK.")
            return False

        self.log(f"No gain ACK. Buffer: {buf!r}")
        return False

    def wait_for_stream_banner(self, console, timeout=40):
        """Wait on the already-open console for STREAM_TOKEN.
        Call after the gain ACK - this covers the Picamera2 init delay."""
        found, buf = self.wait_for(console, [self.STREAM_TOKEN], timeout=timeout)
        if found:
            self.log("Sender is streaming - verifying bytes...")
            return True
        self.log("Warning: stream banner not seen - will probe for bytes anyway.")
        return False

    def sender_already_running(self, default_start_gain=30.0):
        """Sender already up but unconfigured (console blank / at READY banner).
        Classifies FIRST so it never types a gain at a login or shell prompt."""
        try:
            console = serial.Serial(self.PORT, self.BAUD, timeout=2)
            try:
                state, _ = self.classify_console(console)
                if state == 'streaming':
                    return True
                if state not in ('sender_ready', 'blank'):
                    self.log(f"Console is at '{state}' - needs a login, not a gain.")
                    return False
                if self.send_gain(console, default_start_gain):
                    self.wait_for_stream_banner(console, timeout=40)
                    return True
                return False
            finally:
                console.close()
        except Exception as e:
            self.log(f"sender_already_running: {e}")
            return False

    def login_and_launch(self, console, state, username='bvandenb', password='Raspibvandenb',
                        path='/home/bvandenb/Documents/PySpectrometerMorgan',
                        script='pi_sender.py'):
        """Log in (if needed) and launch the sender. `state` comes from
        classify_console() - this function never re-probes the console."""
        if state == 'password':
            self.log("Stale password prompt - resetting console...")
            if not self.reset_to_login(console):
                return False
            state = 'login'

        if state == 'login':
            self.log("Login prompt detected, sending credentials...")
            console.write(f'{username}\n'.encode())
            found, buf = self.wait_for(console, [self.PASSWORD_TOKEN], timeout=20)
            if not found:
                self.log(f"No password prompt: {buf!r}")
                return False
            console.write(f'{password}\n'.encode())
            found, buf = self.wait_for(console, self.SHELL_TOKENS + [self.LOGIN_TOKEN] + self.FAIL_TOKENS,
                                timeout=25)
            if found is None or found == self.LOGIN_TOKEN or found in self.FAIL_TOKENS:
                self.log(f"Login failed: {buf!r}")
                return False
            self.log("Logged in.")
        elif state == 'shell':
            self.log("Shell prompt detected - already logged in.")
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
            self.log(f"Launch failed: {buf!r}")
            return False
        if found:
            self.log("Sender ready - waiting for gain config.")
        else:
            self.log("Ready banner not seen after launch - falling back to 15 s wait.")
            time.sleep(15)
        return True

    def bring_up_pi(self, default_start_gain=30.0, max_attempts=None):
        attempt = 0
        while True:
            attempt += 1
            if max_attempts is not None and attempt > max_attempts:
                self.log(f"Giving up after {max_attempts} attempt(s).")
                return False
            self.log(f"Handshake attempt {attempt}...")

            gained = False
            banner_seen = False
            streaming = False
            try:
                console = serial.Serial(self.PORT, self.BAUD, timeout=2)
                try:
                    state, _ = self.classify_console(console)

                    if state == 'streaming':
                        self.log("Pi is already streaming.")
                        streaming = True
                    elif state in ('sender_ready', 'blank'):
                        # Sender owns the console and is waiting for its config
                        gained = self.send_gain(console, default_start_gain)
                    else:
                        ok = self.login_and_launch(console, state)
                        if ok is False:
                            raise RuntimeError("login_and_launch failed")
                        gained = self.send_gain(console, default_start_gain)

                    if gained:
                        # Covers Picamera2 init before we let go of the port
                        banner_seen = self.wait_for_stream_banner(console, timeout=40)
                finally:
                    # Console MUST be closed before we probe the data stream
                    console.close()
            except Exception as e:
                self.log(f"Attempt {attempt} failed: {e}")
                time.sleep(2)
                continue

            if streaming:
                return True

            if not gained:
                self.log(f"Attempt {attempt}: gain not acknowledged, retrying...")
                time.sleep(2)
                continue

            if not banner_seen:
                self.log("Stream banner missed - short settle before byte probe.")

            time.sleep(0.5)
            if self.stream_detected(seconds=5):
                self.log("Handshake successful - data stream is live.")
                return True
            self.log(f"Attempt {attempt}: no data bytes detected, retrying...")
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
                self.log("Timeout waiting for START marker - attempting recovery...")
                self.close_data()
                if not self.bring_up_pi(max_attempts=3):
                    self.log("Recovery failed - is the Pi alive?")
                self.open_data()
                return None, None
            byte = self.ser.read(1)
            if byte == b'\xAA' and self.ser.read(1) == b'\x55':
                break

        type_code = struct.unpack("B", self.read_exact(1))[0]
        length    = struct.unpack("<H", self.read_exact(2))[0]
        if length > self.MAX_PAYLOAD:
            self.log(f"Error: bogus length {length}, resyncing")
            return None, None
        payload      = self.read_exact(length)
        received_crc = struct.unpack("<H", self.read_exact(2))[0]
        end_marker   = self.read_exact(2)

        if end_marker != self.END:
            self.log("Error: Bad end marker")
            return None, None

        calculated_crc = binascii.crc_hqx(struct.pack("B", type_code) + payload, 0xFFFF)
        if calculated_crc != received_crc:
            self.log("Error: CRC check failed")
            return None, None

        target_dtype = self.CODE_TO_DTYPE.get(type_code, np.uint8)
        return np.frombuffer(payload, dtype=target_dtype), target_dtype



class specGUI(QMainWindow):
    def __init__(self, port='COM4', rep_rate=10, testing=False,
                 cal_file="calData.csv", default_start_gain=30.0):
        super().__init__()
        self.setWindowTitle("Spectrometer Serial")
        self.resize(1500, 800)
        self.rep_rate = rep_rate
        self.testing = testing
        self.cal_file = cal_file
        self.default_start_gain = default_start_gain
        self.pfind_state = False
        self.smooth_state = True
        self.calibrating = False
        self.packet_count = 0
        self.bad_count = 0
        self.wl_data = []
        self.in_data = []

        self.status_msg = 'Testing' if testing else 'Not connected'
        self.init_ui()
        self.load_calibration(self.cal_file)

        if testing:
            self.pi = None
            check = self.check_testing
            self.status_label.setText('Status: Testing (no serial)')
            self.start_btn.setEnabled(True)
        else:
            self.pi = PiSerial(port=port, logger=self.log_box.add_message,
                               autostart=False, default_start_gain=default_start_gain)
            check = self.read_data
            # Handshake after the window is painted, so the log is visible
            QTimer.singleShot(100, self.connect_pi)

        self.live_plotter.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.live_plotter.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(check)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        self.status_label = QLabel(f'Status: {self.status_msg}')
        self.status_label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.status_label, 0, 0,
                         alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.data_label = QLabel("Packets: 0  dropped: 0  samples: 0")
        self.data_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.data_label.setMinimumWidth(450)
        layout.addWidget(self.data_label, 0, 1,
                         alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Reading")
        self.start_btn.setStyleSheet("background-color: green; color: white; padding: 8px;")
        self.start_btn.setEnabled(False)          # enabled once the Pi is up
        self.start_btn.clicked.connect(self.start_loop)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Reading")
        self.stop_btn.setStyleSheet("background-color: red; color: white; padding: 8px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_loop)
        btn_layout.addWidget(self.stop_btn)

        self.peakfind_btn = QPushButton("Enable peak finder")
        self.peakfind_btn.setStyleSheet("background-color: gray; color: white; padding: 8px;")
        self.peakfind_btn.clicked.connect(self.pfinder)
        btn_layout.addWidget(self.peakfind_btn)

        self.smooth_btn = QPushButton("Disable smoothing")
        self.smooth_btn.setStyleSheet("background-color: gray; color: white; padding: 8px;")
        self.smooth_btn.clicked.connect(self.toggle_smoothing)
        btn_layout.addWidget(self.smooth_btn)

        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setStyleSheet("background-color: #555; color: white; padding: 8px;")
        self.calibrate_btn.clicked.connect(self.calibrate)
        btn_layout.addWidget(self.calibrate_btn)


        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet("background-color: gray; color: white; padding: 8px;")
        self.save_btn.clicked.connect(self.save_data)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout, 4, 0)


        self.live_plotter = livePlotter(self)
        self.live_plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.live_plotter, 1, 0, 2, 4)

        self.width_box = QLineEdit()
        self.width_box.setPlaceholderText(f"Width is {self.live_plotter._width}")
        self.width_box.setFixedWidth(150)
        self.width_box.returnPressed.connect(self.handle_width)
        layout.addWidget(self.width_box, 3, 1)

        self.mindist_box = QLineEdit()
        self.mindist_box.setPlaceholderText(f"MinDist is {self.live_plotter._mindist}")
        self.mindist_box.setFixedWidth(150)
        self.mindist_box.returnPressed.connect(self.handle_mindist)
        layout.addWidget(self.mindist_box, 3, 2)

        self.cursor_pos = QLabel("")
        self.cursor_pos.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.cursor_pos.setFixedWidth(150)
        layout.addWidget(self.cursor_pos, 3, 3)

        self.log_box = logs(title='Logs')
        self.log_box.setFixedWidth(280)
        layout.addWidget(self.log_box, 0, 4, 4, 1)

    def gain_popup(self):
        """Prompt user to enter picam gain"""
        gain, ok = QInputDialog.getInt(
                            self, "Enter Gain", " 1-50")
        if not ok:
            return None

        #clamp to allowed values
        gain = max(1,gain)
        gain = min(50,gain)

        return gain
    

    def load_calibration(self, path, width=800):
        """Fit a wavelength axis from calibration points (pixel,wavelength CSV,
        as written by on_click). 3 points -> 2nd order fit, >3 -> 3rd order
        with an R^2 check. Falls back to pixel index if data is missing,
        malformed, or too short."""
        try:
            pixels, wavelengths = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
            pixels, wavelengths = np.atleast_1d(pixels), np.atleast_1d(wavelengths)
            if pixels.size < 3:
                raise ValueError("need at least 3 calibration points")
        except Exception as e:
            pixels = np.array([0, 400, 800])
            wavelengths = np.array([380, 560, 750])

            self.log_box.add_message(f"No calibration data ({e})")
            

        order = 2 if pixels.size == 3 else 3
        coeffs = np.polyfit(pixels, wavelengths, order)
        self.wavelength = np.round(np.polyval(coeffs, np.arange(width)), 6)

        if order == 3:
            r_sq = np.corrcoef(wavelengths, np.polyval(coeffs, pixels))[0, 1] ** 2
            self.log_box.add_message(
                f"Calibrated: {pixels.size} points, 3rd order fit (R^2={r_sq:.5f})")
        else:
            self.log_box.add_message(
                f"Calibrated: 3 points, 2nd order fit (add more points for accuracy)")

    def connect_pi(self):
        self.status_label.setText("Status: connecting...")
        self.log_box.add_message("Bringing up the Pi...")
        try:
            ok = self.pi.connect(default_start_gain=self.default_start_gain)
        except Exception as e:
            ok = False
            self.log_box.add_message(f"Handshake error: {e}")

        if ok:
            self.status_label.setText("Status: Pi streaming - press Start Reading")
            self.status_label.setStyleSheet("color: green; font-size: 14px;")
            self.start_btn.setEnabled(True)
        else:
            self.status_label.setText("Status: handshake failed")
            self.status_label.setStyleSheet("color: red; font-size: 14px;")

    def start_loop(self):
        if self.serial_timer.isActive():
            return
        if not self.testing:
            try:
                self.pi.open_data()          # only open the data port while reading
            except Exception as e:
                self.log_box.add_message(f"Could not open data port: {e}")
                return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: reading...")
        self.status_label.setStyleSheet("color: green; font-size: 14px;")
        self.log_box.add_message('Reading started')
        self.serial_timer.start(int(1000 / self.rep_rate))

    def stop_loop(self):
        if not self.serial_timer.isActive():
            return
        self.serial_timer.stop()
        if not self.testing:
            self.pi.close_data()             # release the port for the console
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: stopped")
        self.status_label.setStyleSheet("color: red; font-size: 14px;")
        self.log_box.add_message('Reading stopped')

    def pfinder(self):
        self.pfind_state = not self.pfind_state
        if self.pfind_state:
            self.peakfind_btn.setText("Disable peak finder")
            self.peakfind_btn.setStyleSheet("background-color: #0078d4; color: white; padding: 8px;")
            self.log_box.add_message('Peak finder enabled')
        else:
            self.peakfind_btn.setText("Enable peak finder")
            self.peakfind_btn.setStyleSheet("background-color: gray; color: white; padding: 8px;")
            self.log_box.add_message('Peak finder disabled')

    def toggle_smoothing(self):
        self.smooth_state = not self.smooth_state
        self.smooth_btn.setText("Disable smoothing" if self.smooth_state else "Enable smoothing")
        self.log_box.add_message(f"Savgol smoothing {'on' if self.smooth_state else 'off'}")

    def calibrate(self):
        self.calibrating = not self.calibrating
        if self.calibrating:
            self.calibrate_btn.setText("Stop Calibration")
            self.calibrate_btn.setStyleSheet("background-color: #0078d4; color: white; padding: 8px;")
            self.log_box.add_message('Calibration mode on - click a peak on the plot')
        else:
            self.calibrate_btn.setText("Calibrate")
            self.calibrate_btn.setStyleSheet("background-color: #555; color: white; padding: 8px;")
            self.log_box.add_message('Calibration mode off')

    def save_data(self):
        
        fname, ok = QInputDialog.getText(
                    self, "Enter Filename", "")
        if not ok:
            return
        
        self.save_path = os.path.dirname(os.path.abspath(__file__))

        df = pd.DataFrame({'wn':self.wl_data,'int':self.in_data})
        df.to_csv(os.path.join(self.save_path, f'{fname}.csv'))
        self.log_box.add_message(f'Data saved to {fname}') 
        

    def on_click(self, event):
        if not self.calibrating or event.inaxes != self.live_plotter.ax:
            return

        if len(self.wl_data) == 0:
            self.log_box.add_message("No data plotted yet - can't calibrate")
            return

        axis = np.asarray(self.wl_data)
        pixel = int(np.argmin(np.abs(axis - event.xdata)))

        wavelength, ok = QInputDialog.getDouble(
            self, "Calibration point", f"Wavelength for pixel {pixel}:",
            decimals=2)
        if not ok:
            return

        write_header = not os.path.exists(self.cal_file)
        with open(self.cal_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['pixel', 'wavelength'])
            writer.writerow([pixel, wavelength])

        self.log_box.add_message(f"Added cal point: pixel {pixel} -> {wavelength} nm")
        self.load_calibration(self.cal_file, width=len(self.wl_data))

    def handle_width(self):
        try:
            self.live_plotter._width = float(self.width_box.text())
        except ValueError:
            self.log_box.add_message('Width must be a number')
            return
        self.width_box.clear()
        self.width_box.setPlaceholderText(f"Width is {self.live_plotter._width}")
        self.log_box.add_message(f"Width changed to {self.live_plotter._width}")

    def handle_mindist(self):
        try:
            self.live_plotter._mindist = float(self.mindist_box.text())
        except ValueError:
            self.log_box.add_message('MinDist must be a number')
            return
        self.mindist_box.clear()
        self.mindist_box.setPlaceholderText(f"MinDist is {self.live_plotter._mindist}")
        self.log_box.add_message(f"Mindist changed to {self.live_plotter._mindist}")

    def on_hover(self, event):
        if event.inaxes == self.live_plotter.ax:
            x, y = event.xdata, event.ydata
            self.cursor_pos.setText(f'({x:.2f},{y:.2f})')
        else:
            self.cursor_pos.setText('')

    def smooth(self, y):
        if self.smooth_state and len(y) >= 51:
            return np.asarray(savgol_filter(y, 51, 3))
        return np.asarray(y)

    def read_data(self):
        try:
            data, dtype = self.pi.read_packet()
        except Exception as e:
            self.bad_count += 1
            self.log_box.add_message(f"Read error: {e}")
            return

        if data is None:
            self.bad_count += 1
            return

        self.packet_count += 1
        self.in_data = np.asarray(data)
        self.wl_data = self.wavelength
        self.data_label.setText(
            f"Packets: {self.packet_count}  dropped: {self.bad_count}  samples: {len(self.in_data)}")
        self.live_plotter.update_plot(self.wl_data, self.smooth(self.in_data), self.pfind_state)

    def check_testing(self):
        self.wl_data = np.linspace(350, 750, 800)

        def gauss(x, mu, sigma, A):
            return A * np.exp(-((x - mu) / sigma) * ((x - mu) / sigma))

        self.in_data = (gauss(self.wl_data, np.random.normal(550, 80),
                              np.random.normal(20, 20), np.random.normal(60, 10))
                        + np.random.normal(10, 3, 800))
        self.packet_count += 1
        self.data_label.setText(
            f"Packets: {self.packet_count}  dropped: {self.bad_count}  samples: {len(self.in_data)}")
        self.live_plotter.update_plot(self.wl_data, self.smooth(self.in_data), self.pfind_state)

    def closeEvent(self, event):
        if self.serial_timer.isActive():
            self.serial_timer.stop()
        if self.pi is not None:
            self.pi.close_data()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = specGUI(port='COM4', rep_rate=10, testing=False,default_start_gain=40.0)
    window.show()
    sys.exit(app.exec())