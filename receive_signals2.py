#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG GUI (ESP32-C3 + ADS1292R)  ---  v9.6.1  (EMG stable + IMU 100Hz + Mag calibration + 9-axis fusion)

Changes vs v8.x:
  - HPF / LPF cutoff are editable (Hz), not just ON/OFF
  - Device setting apply feedback via CFG?/IMUCFG?/GLITCH? readback + OK/ERR lines
  - IMU: raw 6-axis + mag + temperature received at 100Hz, host-side magnetometer calibration + Madgwick 9-axis fusion -> quaternion
  - CSV: adds IMU timestamp(ms), accel/gyro/mag/temp, fused quaternion (qw,qx,qy,qz)
  - Spike reject (uV abs / step): used both for device GLITCHABS/GLITCHSTEP and host-side guard (prevents huge abnormal spikes)

Python: 3.9+
Packages expected: numpy, pyqtgraph, (PySide6 or PyQt5), pyopengl (optional for 3D)

Protocol:
  - EMG packets unchanged
  - IMU packets: seq(u32) + ms(u32) + 10 floats + flags(u8)  => 49 bytes payload
"""

import sys
import time
import math
import socket
import struct
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

# ======================= UDP Discovery (for router STA mode) =======================
# Firmware will listen on DISCOVERY_PORT and reply to DISCOVERY_MAGIC with a one-line ASCII response:
#   EMG_HERE_V1 id=.. label=.. mac=.. ip=.. tcp=3333 rssi=.. fw=..
DISCOVERY_PORT = 3334
DISCOVERY_MAGIC = b"EMG_DISCOVER_V1"
DISCOVERY_REPLY_PREFIX = "EMG_HERE_V1"
DISCOVERY_TIMEOUT_S = 0.6

# ---- Qt binding ----
QT_LIB = None
try:
    from PySide6 import QtCore, QtWidgets, QtGui

    QT_LIB = "PySide6"
except Exception:
    from PyQt5 import QtCore, QtWidgets, QtGui  # type: ignore

    QT_LIB = "PyQt5"

import pyqtgraph as pg

# ---- pyqtgraph performance defaults ----
# 关闭抗锯齿能显著降低CPU占用（尤其是1000SPS+多曲线实时绘制时）
try:
    pg.setConfigOptions(antialias=False)
except Exception:
    pass

# Optional OpenGL 3D
HAS_GL = False
try:
    import pyqtgraph.opengl as gl  # type: ignore

    HAS_GL = True
except Exception:
    HAS_GL = False

# ---------------- Constants ----------------
VREF_V = 2.418  # ADS1292R internal reference typical (V)
# ADS1292R: 24-bit, output is signed, with full-scale = +/-Vref/gain
# uV = counts * Vref * 1e6 / (gain * 2^23)
ADC_FULL_SCALE = float(1 << 23)


def adc_to_uv(counts: np.ndarray, gain: int) -> np.ndarray:
    # counts: int32 array
    return counts.astype(np.float64) * (VREF_V * 1e6) / (float(gain) * ADC_FULL_SCALE)


def ADS1292R_uV_per_count(gain: int) -> float:
    """Return microvolts per ADC count for ADS1292R (24-bit, signed).

    Firmware sends raw ADC counts (int32). Convert to uV:
        uV = counts * (VREF_V * 1e6) / (gain * 2^23)

    This helper returns the multiplicative factor (uV per count).
    """
    try:
        g = int(gain)
    except Exception:
        g = 1
    if g <= 0:
        g = 1
    return (VREF_V * 1e6) / (float(g) * ADC_FULL_SCALE)



# ---------------- Simple IIR biquad filters ----------------
class Biquad:
    __slots__ = ("b0", "b1", "b2", "a1", "a2", "z1", "z2")

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float):
        self.b0 = float(b0);
        self.b1 = float(b1);
        self.b2 = float(b2)
        self.a1 = float(a1);
        self.a2 = float(a2)
        self.z1 = 0.0;
        self.z2 = 0.0

    def reset(self) -> None:
        self.z1 = 0.0;
        self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        # Direct Form II Transposed
        y = np.empty_like(x, dtype=np.float64)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(x.size):
            xi = float(x[i])
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y


def _norm_f0(fs: float, f0: float) -> float:
    if fs <= 0:
        return 0.0
    return max(1e-6, min(0.499, f0 / fs))


def design_hpf(fs: float, fc: float, q: float = 0.707) -> Biquad:
    f = _norm_f0(fs, fc)
    w0 = 2 * math.pi * f
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b0 = (1 + cosw) / 2
    b1 = -(1 + cosw)
    b2 = (1 + cosw) / 2
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def design_lpf(fs: float, fc: float, q: float = 0.707) -> Biquad:
    f = _norm_f0(fs, fc)
    w0 = 2 * math.pi * f
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b0 = (1 - cosw) / 2
    b1 = 1 - cosw
    b2 = (1 - cosw) / 2
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def design_notch(fs: float, f0: float, q: float = 30.0) -> Biquad:
    f = _norm_f0(fs, f0)
    w0 = 2 * math.pi * f
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b0 = 1
    b1 = -2 * cosw
    b2 = 1
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


# ---------------- IMU utilities ----------------
@dataclass
class ImuSample:
    seq: int
    ms: int
    ax: float;
    ay: float;
    az: float  # g
    gx: float;
    gy: float;
    gz: float  # dps
    mx: float;
    my: float;
    mz: float  # uT
    tempC: float
    flags: int  # bit0=ICM ok, bit1=MMC ok


class AxisMap:
    """
    Map a 3D vector using a permutation + sign flips.

    cfg = ((src_idx, sign), (src_idx, sign), (src_idx, sign)) for output (x,y,z)
    src_idx in {0,1,2}, sign in {+1,-1}
    """

    def __init__(self):
        self.cfg = ((0, +1), (1, +1), (2, +1))

    def set_cfg(self, cfg: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> None:
        self.cfg = cfg

    def apply(self, v: np.ndarray) -> np.ndarray:
        (sx, sgnx), (sy, sgny), (sz, sgnz) = self.cfg
        return np.array([sgnx * v[sx], sgny * v[sy], sgnz * v[sz]], dtype=np.float64)


class MagCal:
    """
    Simple hard-iron + per-axis soft-iron scaling:
      m_cal = (m_raw - offset) * scale
    """

    def __init__(self):
        self.offset = np.zeros(3, dtype=np.float64)
        self.scale = np.ones(3, dtype=np.float64)
        self.valid = False

    def apply(self, m: np.ndarray) -> np.ndarray:
        return (m - self.offset) * self.scale

    def solve_minmax(self, samples: np.ndarray) -> None:
        # samples: Nx3
        if samples.shape[0] < 50:
            raise ValueError("Need more samples (>=50) for mag calibration.")
        mn = samples.min(axis=0)
        mx = samples.max(axis=0)
        self.offset = (mx + mn) / 2.0
        half = (mx - mn) / 2.0
        avg = float(np.mean(half))
        # Avoid divide-by-zero
        self.scale = np.where(half > 1e-9, avg / half, 1.0)
        self.valid = True

    def to_dict(self) -> dict:
        return {"offset": self.offset.tolist(), "scale": self.scale.tolist(), "valid": bool(self.valid)}

    def from_dict(self, d: dict) -> None:
        self.offset = np.array(d.get("offset", [0, 0, 0]), dtype=np.float64)
        self.scale = np.array(d.get("scale", [1, 1, 1]), dtype=np.float64)
        self.valid = bool(d.get("valid", False))


class MadgwickAHRS:
    """
    Madgwick AHRS (9-axis when mag is present, IMU-only fallback).
    Quaternion convention: q = [w, x, y, z]
    """

    def __init__(self, beta: float = 0.1):
        self.beta = float(beta)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self) -> None:
        self.q[:] = (1.0, 0.0, 0.0, 0.0)

    def update_imu(self, gx: float, gy: float, gz: float, ax: float, ay: float, az: float, dt: float) -> None:
        # Gyro in rad/s, accel in any units
        q1, q2, q3, q4 = self.q  # w,x,y,z

        # Normalise accel
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-9:
            return
        ax /= norm;
        ay /= norm;
        az /= norm

        # Gradient descent corrective step
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q4 = 2.0 * q4
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _4q3 = 4.0 * q3
        _8q2 = 8.0 * q2
        _8q3 = 8.0 * q3
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        q4q4 = q4 * q4

        s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s2 = _4q2 * q4q4 - _2q4 * ax + 4.0 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
        s4 = 4.0 * q2q2 * q4 - _2q2 * ax + 4.0 * q3q3 * q4 - _2q3 * ay
        norm_s = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
        if norm_s < 1e-9:
            return
        s1 /= norm_s;
        s2 /= norm_s;
        s3 /= norm_s;
        s4 /= norm_s

        # qDot = 0.5*q*omega - beta*s
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt
        q4 += qDot4 * dt

        norm_q = math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        if norm_q < 1e-9:
            return
        self.q[:] = (q1 / norm_q, q2 / norm_q, q3 / norm_q, q4 / norm_q)

    def update(self, gx: float, gy: float, gz: float,
               ax: float, ay: float, az: float,
               mx: float, my: float, mz: float,
               dt: float) -> None:
        # Gyro in rad/s, accel/mag in any units
        q1, q2, q3, q4 = self.q

        # Normalise accel
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-9:
            return
        ax /= norm;
        ay /= norm;
        az /= norm

        # Normalise mag
        normm = math.sqrt(mx * mx + my * my + mz * mz)
        if normm < 1e-9:
            # no mag => fallback
            self.update_imu(gx, gy, gz, ax, ay, az, dt)
            return
        mx /= normm;
        my /= normm;
        mz /= normm

        _2q1 = 2.0 * q1;
        _2q2 = 2.0 * q2;
        _2q3 = 2.0 * q3;
        _2q4 = 2.0 * q4
        _2q1q3 = 2.0 * q1 * q3
        _2q3q4 = 2.0 * q3 * q4
        q1q1 = q1 * q1;
        q1q2 = q1 * q2;
        q1q3 = q1 * q3;
        q1q4 = q1 * q4
        q2q2 = q2 * q2;
        q2q3 = q2 * q3;
        q2q4 = q2 * q4
        q3q3 = q3 * q3;
        q3q4 = q3 * q4
        q4q4 = q4 * q4

        # Reference direction of Earth's magnetic field
        hx = mx * (q1q1 + q2q2 - q3q3 - q4q4) + my * (2.0 * (q2q3 - q1q4)) + mz * (2.0 * (q2q4 + q1q3))
        hy = mx * (2.0 * (q2q3 + q1q4)) + my * (q1q1 - q2q2 + q3q3 - q4q4) + mz * (2.0 * (q3q4 - q1q2))
        _2bx = math.sqrt(hx * hx + hy * hy)
        _2bz = mx * (2.0 * (q2q4 - q1q3)) + my * (2.0 * (q3q4 + q1q2)) + mz * (q1q1 - q2q2 - q3q3 + q4q4)
        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        # Gradient descent step
        s1 = (-_2q3 * (2.0 * (q2q4 - q1q3) - ax) +
              _2q2 * (2.0 * (q1q2 + q3q4) - ay) -
              _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) +
              (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) +
              _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s2 = (_2q4 * (2.0 * (q2q4 - q1q3) - ax) +
              _2q1 * (2.0 * (q1q2 + q3q4) - ay) -
              4.0 * q2 * (1.0 - 2.0 * (q2q2 + q3q3) - az) +
              _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) +
              (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) +
              (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s3 = (-_2q1 * (2.0 * (q2q4 - q1q3) - ax) +
              _2q4 * (2.0 * (q1q2 + q3q4) - ay) -
              4.0 * q3 * (1.0 - 2.0 * (q2q2 + q3q3) - az) +
              (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) +
              (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) +
              (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s4 = (_2q2 * (2.0 * (q2q4 - q1q3) - ax) +
              _2q3 * (2.0 * (q1q2 + q3q4) - ay) +
              (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) +
              (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) +
              _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        norm_s = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
        if norm_s < 1e-9:
            return
        s1 /= norm_s;
        s2 /= norm_s;
        s3 /= norm_s;
        s4 /= norm_s

        # Quaternion rate from gyros
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt
        q4 += qDot4 * dt

        norm_q = math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        if norm_q < 1e-9:
            return
        self.q[:] = (q1 / norm_q, q2 / norm_q, q3 / norm_q, q4 / norm_q)


def quat_to_axis_angle(q: np.ndarray) -> Tuple[float, float, float, float]:
    # returns angle_deg, ax, ay, az
    q = q.astype(np.float64)
    w, x, y, z = q
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return 0.0, 1.0, 0.0, 0.0
    ax = x / s;
    ay = y / s;
    az = z / s
    return math.degrees(angle), ax, ay, az


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product: q = q1 * q2
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


# ---------------- Receiver ----------------

class Receiver(QtCore.QObject):
    """
    More robust TCP 接收器（修复“Connectafter无法再次Connect/连上无数据”的common issues）：
      - 不再把 QObject moveToThread(QThread)（这是导致重连不稳定的主要原因之一）
      - 改为使用 Python threading.Thread + socket timeout，使得Disconnect/重连更可靠
    """
    text_sig = QtCore.Signal(str) if QT_LIB == "PySide6" else QtCore.pyqtSignal(str)
    stat_sig = QtCore.Signal(dict) if QT_LIB == "PySide6" else QtCore.pyqtSignal(dict)
    bat_sig = QtCore.Signal(dict) if QT_LIB == "PySide6" else QtCore.pyqtSignal(dict)

    # Packet types（固件协议保持不变）
    PKT_EMG = ord('E')
    PKT_IMU = ord('I')
    PKT_STAT = ord('S')
    PKT_TEXT = ord('T')
    PKT_BAT = ord('B')

    def __init__(self):
        super().__init__()
        self.sock: Optional[socket.socket] = None
        self.buf = bytearray()
        self.running = False
        self.q_emg = deque()
        self.q_imu = deque()
        self.connected = False
        self.last_rx_time = 0.0

        self._rx_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._tx_lock = threading.Lock()

    def connect_to(self, host: str, port: int) -> bool:
        self.disconnect()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((host, int(port)))
            # 关键：设置一个较小的timeout，方便退出线程、支持快速重连
            s.settimeout(0.25)
            self.sock = s
            self.connected = True
            self.running = True
            self.last_rx_time = time.time()
            self.buf.clear()
            self._stop_evt.clear()

            th = threading.Thread(target=self._rx_loop, args=(s,), daemon=True)
            self._rx_thread = th
            th.start()
            return True
        except Exception as e:
            self.text_sig.emit(f"[ERR] connect: {e}")
            try:
                s.close()
            except Exception:
                pass
            self.connected = False
            self.running = False
            self.sock = None
            return False

    def disconnect(self) -> None:
        # 标记停止
        self.running = False
        self.connected = False
        self._stop_evt.set()

        s = self.sock
        self.sock = None

        # 关闭socket以打断 recv
        if s:
            try:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                s.close()
            except Exception:
                pass

        # 等待接收线程结束（短等待，避免卡死）
        th = self._rx_thread
        self._rx_thread = None
        if th and th.is_alive():
            try:
                th.join(timeout=0.6)
            except Exception:
                pass

    def send_line(self, line: str) -> None:
        if not line:
            return
        if not self.sock:
            return
        if not line.endswith("\n"):
            line = line + "\n"
        data = line.encode("utf-8", errors="ignore")
        try:
            with self._tx_lock:
                if self.sock:
                    self.sock.sendall(data)
        except Exception:
            # 发送失败通常意味着已断开
            pass

    def send_lines(self, lines: List[str]) -> None:
        if not lines:
            return
        for ln in lines:
            self.send_line(ln)

    def _rx_loop(self, s: socket.socket) -> None:
        try:
            while (not self._stop_evt.is_set()) and self.running and (self.sock is s):
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    continue
                except Exception:
                    break
                if not chunk:
                    break
                self.last_rx_time = time.time()
                self.buf.extend(chunk)
                self._parse_buf()
        except Exception:
            pass
        # 线程退出
        self.running = False
        self.connected = False
        self.text_sig.emit("[INFO] Disconnected")

    def _parse_buf(self) -> None:
        # Frame: 0xA5 0x5A [type u8] [len u16 LE] [payload] [crc16 LE]
        while True:
            if len(self.buf) < 7:
                return
            # sync
            if self.buf[0] != 0xA5 or self.buf[1] != 0x5A:
                # resync
                try:
                    i = self.buf.index(0xA5)
                    self.buf = self.buf[i:]
                except ValueError:
                    self.buf.clear()
                    return
                if len(self.buf) < 7:
                    return
                if self.buf[0] != 0xA5 or self.buf[1] != 0x5A:
                    self.buf = self.buf[1:]
                    continue
            ptype = self.buf[2]
            plen = self.buf[3] | (self.buf[4] << 8)
            frame_len = 2 + 1 + 2 + plen + 2
            if len(self.buf) < frame_len:
                return
            payload = self.buf[5:5 + plen]
            # skip CRC check (firmware already stable, and python CRC cost is high)
            self.buf = self.buf[frame_len:]

            if ptype == self.PKT_TEXT:
                try:
                    s = payload.decode("utf-8", errors="replace").strip()
                    if s:
                        self.text_sig.emit(s)
                except Exception:
                    pass

            elif ptype == self.PKT_STAT:
                if plen == 15:
                    ms, err, rec, code, detail, flags = struct.unpack_from("<III3B", payload, 0)
                    d = {"ms": ms, "err": err, "rec": rec, "code": code, "detail": detail, "flags": flags}
                    self.stat_sig.emit(d)

            elif ptype == self.PKT_EMG:
                # start_idx(u32) gain(u8) flags(u8) fs(u16) + 40*i32(CH1) [+ 40*i32(CH2)]
                # 兼容：旧单通道固件可能只发送 CH1（40 samples）
                if plen >= 8 + 40 * 4:
                    start_idx = struct.unpack_from("<I", payload, 0)[0]
                    gain = payload[4]
                    flags = payload[5]
                    fs = struct.unpack_from("<H", payload, 6)[0]
                    off = 8
                    ch1 = np.frombuffer(payload, dtype=np.int32, count=40, offset=off).copy()
                    off += 40 * 4
                    if plen >= 8 + 40 * 4 * 2:
                        ch2 = np.frombuffer(payload, dtype=np.int32, count=40, offset=off).copy()
                    else:
                        ch2 = np.zeros(40, dtype=np.int32)
                    self.q_emg.append((start_idx, gain, flags, fs, ch1, ch2))

            elif ptype == self.PKT_IMU:
                # v9 firmware: seq(u32) + ms(u32) + 10 floats + flags(u8) => 49 bytes payload
                if plen == 49:
                    seq, ms = struct.unpack_from("<II", payload, 0)
                    vals = struct.unpack_from("<10f", payload, 8)
                    flags = payload[48]
                    imu = ImuSample(seq=seq, ms=ms,
                                    ax=vals[0], ay=vals[1], az=vals[2],
                                    gx=vals[3], gy=vals[4], gz=vals[5],
                                    mx=vals[6], my=vals[7], mz=vals[8],
                                    tempC=vals[9],
                                    flags=flags)
                    self.q_imu.append(imu)
                # backward compat (old): 45 bytes
                elif plen == 45:
                    seq = struct.unpack_from("<I", payload, 0)[0]
                    vals = struct.unpack_from("<10f", payload, 4)
                    flags = payload[44]
                    imu = ImuSample(seq=seq, ms=0,
                                    ax=vals[0], ay=vals[1], az=vals[2],
                                    gx=vals[3], gy=vals[4], gz=vals[5],
                                    mx=vals[6], my=vals[7], mz=vals[8],
                                    tempC=vals[9],
                                    flags=flags)
                    self.q_imu.append(imu)

            elif ptype == self.PKT_BAT:
                # u32 ms + u16 vbat_mV + u8 pct + u16 adc_mV + u8 reason => 10 bytes (v9.4+)
                if plen >= 7:
                    ms = struct.unpack_from("<I", payload, 0)[0]
                    vbat_mV = struct.unpack_from("<H", payload, 4)[0] if plen >= 6 else 0
                    pct = payload[6] if plen >= 7 else 0
                    adc_mV = struct.unpack_from("<H", payload, 7)[0] if plen >= 9 else 0
                    reason = payload[9] if plen >= 10 else 0
                    self.bat_sig.emit({"ms": ms, "vbat_mV": int(vbat_mV), "pct": int(pct),
                                       "adc_mV": int(adc_mV), "reason": int(reason)})


# ---------------- CSV Logger ----------------
class CsvLogger:
    """
    CSV Logging（分两个文件）：
      - EMG：1000Hz（每行一个采样）
      - IMU：100Hz（每条IMU一行；by imu_ms deduplicate，保证不写重复行）

    Time戳（by你的要求尽量简单）：
      - IMU1_RefTime：from device imu_ms(毫s) converted to "HH:MM:SS.mmm"
                     这是“IMU acquisition time（relative since power-on Time）”，用于after续轨迹重建。
      - EMG_t_s：由采样序号/Sample rateconverted to的s（since this CSVStart计时）
      - Unix_Time_s: 绝对系统时间戳，用于多模态（视频等）数据对齐
    """

    def __init__(self):
        self.emg_f = None
        self.imu_f = None
        self.enabled = False
        self.emg_path = ""
        self.imu_path = ""
        self._last_imu_ms_written = None

        # EMG time base (from first sample index)
        self._emg_base_idx = None
        self._emg_base_fs = None
        
        # Absolute alignment clock
        self._unix_start_time = 0.0
        self._first_imu_ms = None

    @staticmethod
    def _fmt_hmsms_from_ms(ms: int) -> str:
        """ms -> 'HH:MM:SS.mmm' (time since boot)"""
        if ms is None:
            ms = 0
        try:
            ms = int(ms)
        except Exception:
            ms = 0
        if ms < 0:
            ms = 0
        hh = ms // 3600000
        mm = (ms % 3600000) // 60000
        ss = (ms % 60000) // 1000
        mmm = ms % 1000
        return f"'{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"  # 前置单引号，防止Excel吞掉末尾0

    def start(self, emg_path: str, imu_path: str, meta_lines: Optional[List[str]] = None) -> None:
        self.stop()
        self.emg_f = open(emg_path, "w", encoding="utf-8-sig", newline="")
        self.imu_f = open(imu_path, "w", encoding="utf-8-sig", newline="")
        self.emg_path = emg_path
        self.imu_path = imu_path
        self.enabled = True
        self._last_imu_ms_written = None
        self._emg_base_idx = None
        self._emg_base_fs = None
        
        # Sync to system clock at recording trigger
        self._unix_start_time = time.time()
        self._first_imu_ms = None

        # meta
        if meta_lines:
            for ln in meta_lines:
                if not ln.startswith("#"):
                    ln = "# " + ln
                self.emg_f.write(ln.rstrip() + "\n")
                self.imu_f.write(ln.rstrip() + "\n")

        # a short note so you know what the timestamp is
        self.imu_f.write("# IMU1_RefTime = format(imu_ms) = relative since power-on Time (IMU acquisition time)\n")
        self.imu_f.write(f"# Unix_Time_s_at_Start = {self._unix_start_time:.6f} (System time when start button was pressed)\n")

        # EMG columns note
        self.emg_f.write(
            "# EMG column notes: Unix_Time_s=aligned with video; EMG_t_s=seconds since this Start; sample_idx=device sample index (ADS frame index); fs=sample rate (SPS); gain=gain; flags=0x01(TEST) 0x02(SHORT) 0x04(RLD); emg_valid=1 valid / 0 invalid (spike-guard decision); ch1_raw_uV=raw uV (after spike guard); ch1_filt_uV=filtered uV; ch1_env_uV=envelope (RMS) uV\n")

        # headers
        self.emg_f.write(",".join([
            "Unix_Time_s", "EMG_t_s", "sample_idx", "fs", "gain", "flags",
            "emg_valid",
            "ch1_raw_uV", "ch1_filt_uV", "ch1_env_uV",
            "ch2_raw_uV", "ch2_filt_uV", "ch2_env_uV"
        ]) + "\n")

        self.imu_f.write(",".join([
            "Unix_Time_s",
            "IMU1_RefTime",
            "ax_g", "ay_g", "az_g",
            "gx_dps", "gy_dps", "gz_dps",
            "mx_uT", "my_uT", "mz_uT",
            "temp_C",
            "qw", "qx", "qy", "qz"
        ]) + "\n")

        self.emg_f.flush()
        self.imu_f.flush()

    def stop(self) -> None:
        for f in (self.emg_f, self.imu_f):
            if f:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
        self.emg_f = None
        self.imu_f = None
        self.enabled = False
        self.emg_path = ""
        self.imu_path = ""
        self._last_imu_ms_written = None
        self._emg_base_idx = None
        self._emg_base_fs = None

    def write_imu(self, imu: Optional[dict], q: Optional[np.ndarray]) -> None:
        """Write一条IMU（100Hz）。by imu_ms deduplicate：same ms do not write twice。"""
        if not self.enabled or not self.imu_f or imu is None or q is None:
            return
        try:
            ms = int(imu.get("ms", 0))
        except Exception:
            ms = 0

        # de-dup by imu_ms
        if self._last_imu_ms_written is not None and ms == self._last_imu_ms_written:
            return
        self._last_imu_ms_written = ms

        # Alignment calculation
        if self._first_imu_ms is None:
            self._first_imu_ms = ms
        unix_ts = self._unix_start_time + (ms - self._first_imu_ms) / 1000.0

        ref_time = self._fmt_hmsms_from_ms(ms)

        def _f(x, nd=6):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return ""

        line = [
            f"{unix_ts:.6f}",
            ref_time,
            _f(imu.get("ax_g", ""), 6), _f(imu.get("ay_g", ""), 6), _f(imu.get("az_g", ""), 6),
            _f(imu.get("gx_dps", ""), 6), _f(imu.get("gy_dps", ""), 6), _f(imu.get("gz_dps", ""), 6),
            _f(imu.get("mx_uT", ""), 6), _f(imu.get("my_uT", ""), 6), _f(imu.get("mz_uT", ""), 6),
            _f(imu.get("temp_C", ""), 3),
            f"{float(q[0]):.8f}", f"{float(q[1]):.8f}", f"{float(q[2]):.8f}", f"{float(q[3]):.8f}",
        ]
        self.imu_f.write(",".join(line) + "\n")

    def write_emg_block(self,
                        start_idx: int, fs: int, gain: int, flags: int,
                        ch1_raw_uv: np.ndarray, ch1_filt_uv: np.ndarray, ch1_env_uv: np.ndarray,
                        valid_mask: np.ndarray, blank_invalid: bool,
                        ch2_raw_uv: Optional[np.ndarray] = None,
                        ch2_filt_uv: Optional[np.ndarray] = None,
                        ch2_env_uv: Optional[np.ndarray] = None) -> None:
        """Write一段 EMG（支持双通道）

        CSV列：
          Unix_Time_s,EMG_t_s,sample_idx,fs,gain,flags,emg_valid,
          ch1_raw_uV,ch1_filt_uV,ch1_env_uV,
          ch2_raw_uV,ch2_filt_uV,ch2_env_uV
        """
        if self.emg_f is None:
            return

        try:
            n = int(ch1_raw_uv.size)
        except Exception:
            return
        if n <= 0:
            return

        # base timeline (keep stable across packets)
        if self._emg_base_idx is None:
            self._emg_base_idx = int(start_idx)
        if self._emg_base_fs is None:
            self._emg_base_fs = float(fs) if fs > 0 else 1.0

        base_idx = int(self._emg_base_idx)
        base_fs = float(self._emg_base_fs) if self._emg_base_fs else (float(fs) if fs > 0 else 1.0)
        if base_fs <= 0:
            base_fs = 1.0

        # CH2 presence check
        has_ch2 = False
        if (ch2_raw_uv is not None) and (ch2_filt_uv is not None) and (ch2_env_uv is not None):
            try:
                has_ch2 = (int(ch2_raw_uv.size) == n and int(ch2_filt_uv.size) == n and int(ch2_env_uv.size) == n)
            except Exception:
                has_ch2 = False

        lines = []
        for i in range(n):
            sample_idx = int(start_idx) + int(i)
            t_s = (sample_idx - base_idx) / base_fs
            unix_ts = self._unix_start_time + t_s # Aligned absolute time
            emg_valid = 1 if bool(valid_mask[i]) else 0

            if (emg_valid == 0) and blank_invalid:
                ch1r = ""
                ch1f = ""
                ch1e = ""
                ch2r = ""
                ch2f = ""
                ch2e = ""
            else:
                ch1r = f"{float(ch1_raw_uv[i]):.3f}"
                ch1f = f"{float(ch1_filt_uv[i]):.3f}"
                ch1e = f"{float(ch1_env_uv[i]):.3f}"

                if has_ch2:
                    ch2r = f"{float(ch2_raw_uv[i]):.3f}"
                    ch2f = f"{float(ch2_filt_uv[i]):.3f}"
                    ch2e = f"{float(ch2_env_uv[i]):.3f}"
                else:
                    ch2r = ""
                    ch2f = ""
                    ch2e = ""

            lines.append(",".join([
                f"{unix_ts:.6f}",
                f"{t_s:.6f}",
                str(sample_idx),
                f"{float(fs):.2f}",
                str(int(gain)),
                str(int(flags)),
                str(int(emg_valid)),
                ch1r, ch1f, ch1e,
                ch2r, ch2f, ch2e,
            ]))

        self.emg_f.write("\n".join(lines) + "\n")


import cv2
import time
import csv
import os

import cv2
import time
import csv

class VideoRecorder(QtCore.QThread):
    def __init__(self, camera_idx=0, save_path="tactile_video.avi"):
        super().__init__()
        self.camera_idx = camera_idx
        self.save_path = save_path.replace(".mp4", ".avi") 
        self.log_path = self.save_path.replace(".avi", "_timestamps.csv")
        self.running = False
        self.cap = None
        self.out = None

    def run(self):
        """方案 A：动态帧率适配版录制主循环"""
        # 1. 使用 DSHOW 后端并强制设置分辨率以降低带宽占用
        self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"[ERR] 无法打开相机 {self.camera_idx}")
            return

        # 显式限制分辨率至 640x480 以确保 USB 稳定性
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 2. 初始化缓冲：连续读取 5 帧丢弃，让硬件传输稳定下来
        for _ in range(5):
            self.cap.read()
        
        # --- 核心修改部分：动态获取相机物理帧率 ---
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 如果相机未返回有效帧率（某些驱动会返回 0），则指定一个标准值（如 30.0）
        if fps <= 0 or fps > 120:
            fps = 60.0
            print(f"[INFO] 无法获取硬件帧率，使用默认值: {fps}")
        else:
            print(f"[INFO] 检测到相机物理帧率: {fps}")
        # ----------------------------------------------------

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 使用获取到的动态 fps 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
        
        try:
            # 开启行缓冲 (buffering=1) 确保时间戳实时写入硬盘
            with open(self.log_path, mode='w', newline='', buffering=1) as f:
                log_writer = csv.writer(f)
                log_writer.writerow(["Frame_Index", "System_Time_s", "Human_Readable"])
                
                frame_count = 0
                self.running = True
                fail_count = 0 # 用于统计连续失败帧数
                
                while self.running:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        # 如果读取失败，尝试重试（最多连续 5 次）
                        fail_count += 1
                        if fail_count > 5:
                            print("[ERR] 连续 5 帧读取失败，录制终止")
                            break
                        time.sleep(0.01)
                        continue
                    
                    fail_count = 0 # 读取成功则重置失败计数
                    
                    # 获取高精度系统时间戳
                    curr_t = time.time()
                    t_str = time.strftime("%H:%M:%S", time.localtime(curr_t)) + f".{int((curr_t%1)*1000):03d}"
                    
                    # 在画面压入可视化时间戳
                    cv2.putText(frame, t_str, (10, height - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    self.out.write(frame)
                    log_writer.writerow([frame_count, curr_t, t_str])
                    f.flush() # 强制刷新 CSV 缓冲区
                    frame_count += 1
                    
        except Exception as e:
            print(f"[ERR] 录制中发生异常: {e}")
        finally:
            self._release_resources()

    def _release_resources(self):
        """安全释放相机和文件句柄"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.out is not None:
            self.out.release()
            self.out = None

    def stop(self):
        """停止逻辑，防止界面卡死"""
        self.running = False
        if not self.wait(1000):
            print("[WARN] 视频线程挂起，强制释放资源")
            self._release_resources()
            self.terminate()
            self.wait()
        

class DeviceWidget(QtWidgets.QWidget):
    def __init__(self, slot_index: int = 1):
        super().__init__()
        self.slot_index = int(slot_index)
        # 约定：设备ID/窗口号 1-11 为单通道；12-17 为双通道（CH1+CH2）
        self.dual_mode = (self.slot_index >= 12)
        # 通道使能（双通道设备支持 CH1/CH2 开关；默认双通道 CH1/CH2 都开启）
        self.ch1_enabled = True
        self.ch2_enabled = bool(self.dual_mode)
        # 仅当前选中的设备页绘图，其他页只处理数据不绘图（降低CPU占用，支持17台同时在线（含双通道12-17））
        self.draw_enabled = False
        # UDP发现信息（用于总览页显示/自动连接）
        self.discovered_info = {}
        self.last_discovery_time = 0.0
        self.connecting = False
        self.setWindowTitle(f"Device {self.slot_index:02d} - EMG + IMU Acquisition v9.6.1 ({QT_LIB})")

        self.rx = Receiver()
        self.rx.text_sig.connect(self._on_text)
        self.rx.stat_sig.connect(self._on_stat)
        self.rx.bat_sig.connect(self._on_bat)

        self.csv = CsvLogger()

        # Device state
        self.cur_fs = 1000
        self.cur_gain = 12
        self.connected = False
        self.streaming = False

        # Device version string (from VER?)
        self.dev_fw: str = ""

        # For “应用成功/失败”反馈：保存上一次请求的设备参数，并在 CFG?/IMUCFG?/GLITCH? 回读后自动对比
        self._pending_req: Optional[dict] = None
        self._pending_req_time: float = 0.0
        self._last_verified_cfg: Optional[dict] = None

        # Host-side filter change tracking (avoid log spam)
        self._last_hostflt_desc: str = ""

        # X轴时间基准：使用 ADS start_idx 计算，避免受线程/网络抖动影响
        self._base_start_idx: Optional[int] = None
        # Host-side spike guard
        self.last_good_uv_ch1 = 0.0
        self.last_good_uv_ch2 = 0.0

        # IMU state
        self.latest_imu: Optional[ImuSample] = None
        self.latest_q: Optional[np.ndarray] = None
        self._imu_last_ms: Optional[int] = None
        self.axis_ag = AxisMap()
        self.axis_mag = AxisMap()
        self.mag_cal = MagCal()
        self.fusion = MadgwickAHRS(beta=0.1)
        # 输出四元数以“参考姿态”为零点：q_out = q_ref * q_abs，其中 q_ref 在“复位姿态”时设置为 conj(q_abs)
        self.q_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # IMU UI 更新节流（避免100Hz频繁setText/更新3D导致卡顿）
        self._imu_last_ui_update = 0.0
        self._imu_last_cube_update = 0.0
        self._imu_dt_hist = deque(maxlen=200)  # 用于估计IMU接收频率(从imu.ms差分)
        self._imu_rate_last_ms: Optional[int] = None

        # Battery UI smoothing (host-side)
        self._bat_ema_v: Optional[float] = None
        self._bat_ema_pct: Optional[float] = None
        self.mag_cal_samples: List[np.ndarray] = []
        self.mag_cal_running = False

        # Filters
        self.hpf: Optional[Biquad] = None
        self.lpf: Optional[Biquad] = None
        self.notch: Optional[Biquad] = None

        # Plot window length (seconds) - needs to exist before UI build
        self.win_sec = 5.0

        self._build_ui()
        self._rebuild_filters()

        # Plot buffers

        maxlen = int(self.win_sec * max(1, self.cur_fs))
        self.buf_raw = deque(maxlen=maxlen)
        self.buf_filt = deque(maxlen=maxlen)
        self.buf_env = deque(maxlen=maxlen)
        self.buf_t = deque(maxlen=maxlen)
        # CH2 buffers（双通道设备才会填充；单通道设备保持空即可）
        self.buf_raw2 = deque(maxlen=maxlen)
        self.buf_filt2 = deque(maxlen=maxlen)
        self.buf_env2 = deque(maxlen=maxlen)
        self.t0_plot = time.time()

        # 自动缩放Y轴：使用分位数估计，并做平滑（避免画面抖动）
        self._auto_y_emg: Optional[float] = None
        self._auto_y_env: Optional[float] = None

        # Envelope (RMS) state
        self._env_sq = deque()
        self._env_sum = 0.0
        # CH2 envelope (RMS) state
        self._env_sq2 = deque()
        self._env_sum2 = 0.0
        try:
            ms = float(self.sp_env_ms.value())
        except Exception:
            ms = 50.0
        self._env_n = max(1, int(round(ms / 1000.0 * float(self.cur_fs if self.cur_fs > 0 else 1000))))

        # timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(33)  # ~30Hz UI刷新，避免过高刷新导致卡顿

    # ---------- UI ----------

    def _build_ui(self) -> None:
        main = QtWidgets.QHBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(8)

        # ---------------- Left: tabbed control panel ----------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(360)
        main.addWidget(self.tabs, 0)

        # ===== Tab 1: 采集 =====
        tab_acq_sa = QtWidgets.QScrollArea()
        tab_acq_sa.setWidgetResizable(True)
        tab_acq = QtWidgets.QWidget()
        tab_acq_sa.setWidget(tab_acq)
        v_acq = QtWidgets.QVBoxLayout(tab_acq)
        v_acq.setContentsMargins(6, 6, 6, 6)
        v_acq.setSpacing(8)

        # 连接        # 连接
        g_conn = QtWidgets.QGroupBox("Connect")
        l_conn = QtWidgets.QGridLayout(g_conn)
        self.ed_ip = QtWidgets.QLineEdit("192.168.4.1")
        self.ed_port = QtWidgets.QLineEdit("3333")
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.btn_disc = QtWidgets.QPushButton("Disconnect")
        self.btn_disc.setEnabled(False)

        # STA(路由器)模式下 IP 会变：提供 UDP 扫描，避免手动查路由器 DHCP 列表
        self.btn_scan = QtWidgets.QPushButton("Scan devices")
        self.cb_scan = QtWidgets.QComboBox()
        self.cb_scan.addItem('(Click "Scan devices")')
        self.cb_scan.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        l_conn.addWidget(QtWidgets.QLabel("IP address"), 0, 0);
        l_conn.addWidget(self.ed_ip, 0, 1)
        l_conn.addWidget(QtWidgets.QLabel("Port"), 1, 0);
        l_conn.addWidget(self.ed_port, 1, 1)
        l_conn.addWidget(self.btn_conn, 2, 0);
        l_conn.addWidget(self.btn_disc, 2, 1)
        l_conn.addWidget(self.btn_scan, 3, 0);
        l_conn.addWidget(self.cb_scan, 3, 1)

        self.btn_conn.clicked.connect(self._on_connect)
        self.btn_disc.clicked.connect(self._on_disconnect)
        self.btn_scan.clicked.connect(self._on_scan_devices)
        self.cb_scan.currentIndexChanged.connect(self._on_scan_select)

        self._found_devices = []
        v_acq.addWidget(g_conn)

        # -------- 网络/标识配置（扩展：运行时修改 WiFi / TCP / ID / LABEL） --------
        g_net = QtWidgets.QGroupBox("Network/ID config (advanced)")
        l_net = QtWidgets.QGridLayout(g_net)

        self.ed_net_ssid = QtWidgets.QLineEdit("")
        self.ed_net_pass = QtWidgets.QLineEdit("")
        self.ed_net_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_net_tcp = QtWidgets.QLineEdit("")
        self.ed_net_id = QtWidgets.QLineEdit("")
        self.ed_net_label = QtWidgets.QLineEdit("")

        self.btn_net_query = QtWidgets.QPushButton("Query (NETCFG?)")
        self.btn_net_set_wifi = QtWidgets.QPushButton("Write WiFi/Port (SET_WIFI)")
        self.btn_net_set_dev = QtWidgets.QPushButton("Write ID/Label (SET_DEVICE)")

        self.lb_net_hint = QtWidgets.QLabel(
            "Note: After SET_WIFI or TCPPORT, the device will reboot and switch network/port; the TCP connection will drop.\n"
            "After switching, make sure the PC is on the same Wi‑Fi/router network, then click “Scan devices” to rediscover and connect."
        )
        self.lb_net_hint.setWordWrap(True)

        l_net.addWidget(QtWidgets.QLabel("WiFi SSID"), 0, 0)
        l_net.addWidget(self.ed_net_ssid, 0, 1)
        l_net.addWidget(self.btn_net_query, 0, 2)

        l_net.addWidget(QtWidgets.QLabel("WiFi password"), 1, 0)
        l_net.addWidget(self.ed_net_pass, 1, 1)
        l_net.addWidget(self.btn_net_set_wifi, 1, 2)

        l_net.addWidget(QtWidgets.QLabel("TCP port"), 2, 0)
        l_net.addWidget(self.ed_net_tcp, 2, 1)

        l_net.addWidget(QtWidgets.QLabel("Device ID"), 3, 0)
        l_net.addWidget(self.ed_net_id, 3, 1)
        l_net.addWidget(self.btn_net_set_dev, 3, 2)

        l_net.addWidget(QtWidgets.QLabel("LABEL"), 4, 0)
        l_net.addWidget(self.ed_net_label, 4, 1)

        l_net.addWidget(self.lb_net_hint, 5, 0, 1, 3)

        self.btn_net_query.clicked.connect(self._on_net_query)
        self.btn_net_set_wifi.clicked.connect(self._on_net_set_wifi)
        self.btn_net_set_dev.clicked.connect(self._on_net_set_device)

        v_acq.addWidget(g_net)

        # -------- 电池 --------
        g_bat = QtWidgets.QGroupBox("Battery")
        l_bat = QtWidgets.QGridLayout(g_bat)
        self.lb_bat = QtWidgets.QLabel("—")
        self.lb_bat.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.pb_bat = QtWidgets.QProgressBar()
        self.pb_bat.setRange(0, 100)
        self.pb_bat.setValue(0)
        self.pb_bat.setFormat("%p%")
        l_bat.addWidget(QtWidgets.QLabel("Voltage"), 0, 0)
        l_bat.addWidget(self.lb_bat, 0, 1)
        l_bat.addWidget(self.pb_bat, 1, 0, 1, 2)
        v_acq.addWidget(g_bat)

        # 设备设置
        g_dev = QtWidgets.QGroupBox("Device settings (ADS1292R / IMU)")
        l_dev = QtWidgets.QGridLayout(g_dev)

        self.cb_fs = QtWidgets.QComboBox()
        self.cb_fs.addItems(["500", "1000", "2000"])
        self.cb_fs.setCurrentText("1000")

        self.cb_gain = QtWidgets.QComboBox()
        self.cb_gain.addItems(["1", "2", "3", "4", "6", "8", "12"])
        self.cb_gain.setCurrentText("12")

        self.ck_test = QtWidgets.QCheckBox("Internal test (TEST)")
        self.ck_short = QtWidgets.QCheckBox("Input short (SHORT)")
        self.ck_rld = QtWidgets.QCheckBox("RLD")
        self.ck_imu = QtWidgets.QCheckBox("IMU data")
        self.ck_imu.setChecked(True)

        self.cb_imu_acc = QtWidgets.QComboBox()
        self.cb_imu_acc.addItems(["2", "4", "8", "16"])
        self.cb_imu_acc.setCurrentText("16")

        self.cb_imu_gyr = QtWidgets.QComboBox()
        self.cb_imu_gyr.addItems(["250", "500", "1000", "2000"])
        self.cb_imu_gyr.setCurrentText("2000")

        self.btn_apply = QtWidgets.QPushButton("Apply (don't start)")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        row = 0
        l_dev.addWidget(QtWidgets.QLabel("Sample rate (FS / SPS)"), row, 0);
        l_dev.addWidget(self.cb_fs, row, 1);
        row += 1
        l_dev.addWidget(QtWidgets.QLabel("Gain (GAIN)"), row, 0);
        l_dev.addWidget(self.cb_gain, row, 1);
        row += 1
        l_dev.addWidget(self.ck_test, row, 0);
        l_dev.addWidget(self.ck_short, row, 1);
        row += 1
        l_dev.addWidget(self.ck_rld, row, 0);
        l_dev.addWidget(self.ck_imu, row, 1);
        row += 1
        # 双通道设备：CH1/CH2 开关（仅对 12-17 设备显示）
        if self.dual_mode:
            self.ck_ch1_en = QtWidgets.QCheckBox("Enable CH1")
            self.ck_ch1_en.setChecked(True)
            self.ck_ch2_en = QtWidgets.QCheckBox("Enable CH2")
            self.ck_ch2_en.setChecked(True)
            self.ck_ch1_en.toggled.connect(self._on_ch_en_ui_changed)
            self.ck_ch2_en.toggled.connect(self._on_ch_en_ui_changed)
            l_dev.addWidget(self.ck_ch1_en, row, 0);
            l_dev.addWidget(self.ck_ch2_en, row, 1);
            row += 1
        else:
            self.ck_ch1_en = None
            self.ck_ch2_en = None

        l_dev.addWidget(QtWidgets.QLabel("Accel range (±g)"), row, 0);
        l_dev.addWidget(self.cb_imu_acc, row, 1);
        row += 1
        l_dev.addWidget(QtWidgets.QLabel("Gyro range (±dps)"), row, 0);
        l_dev.addWidget(self.cb_imu_gyr, row, 1);
        row += 1

        l_dev.addWidget(self.btn_apply, row, 0);
        l_dev.addWidget(self.btn_start, row, 1);
        row += 1
        l_dev.addWidget(self.btn_stop, row, 0, 1, 2);
        row += 1

        self.lb_dev_status = QtWidgets.QLabel("Device: -")
        self.lb_dev_status.setWordWrap(True)
        l_dev.addWidget(self.lb_dev_status, row, 0, 1, 2)

        v_acq.addWidget(g_dev)

        # 查询 / 调试
        g_dbg = QtWidgets.QGroupBox("Query / Debug")
        l_dbg = QtWidgets.QGridLayout(g_dbg)
        self.btn_cfg = QtWidgets.QPushButton("Read config (CFG?)")
        self.btn_reg = QtWidgets.QPushButton("Read registers (REG?)")
        self.btn_glitch = QtWidgets.QPushButton("Read spike thresholds (GLITCH?)")
        self.btn_clr = QtWidgets.QPushButton("Clear counters (CLR)")
        self.btn_cfg.clicked.connect(lambda: self._send_lines_cmd(["CFG?", "IMUCFG?", "GLITCH?"]))
        self.btn_reg.clicked.connect(lambda: self._send_line_cmd("REG?"))
        self.btn_glitch.clicked.connect(lambda: self._send_line_cmd("GLITCH?"))
        self.btn_clr.clicked.connect(lambda: self._send_line_cmd("CLR"))
        l_dbg.addWidget(self.btn_cfg, 0, 0, 1, 2)
        l_dbg.addWidget(self.btn_reg, 1, 0)
        l_dbg.addWidget(self.btn_glitch, 1, 1)
        l_dbg.addWidget(self.btn_clr, 2, 0, 1, 2)
        v_acq.addWidget(g_dbg)

        v_acq.addStretch(1)
        self.tabs.addTab(tab_acq_sa, "Acquisition")

        # ===== Tab 2: 滤波 =====
        tab_flt = QtWidgets.QScrollArea()
        tab_flt.setWidgetResizable(True)
        tab_flt_w = QtWidgets.QWidget()
        v_flt = QtWidgets.QVBoxLayout(tab_flt_w)
        v_flt.setContentsMargins(6, 6, 6, 6)
        v_flt.setSpacing(8)

        g_flt = QtWidgets.QGroupBox("Host-side digital filtering (display/analysis only)")
        l_flt = QtWidgets.QGridLayout(g_flt)

        self.ck_hpf = QtWidgets.QCheckBox("High-pass (HPF)")
        self.sp_hpf = QtWidgets.QDoubleSpinBox()
        self.sp_hpf.setRange(0.1, 300.0)
        self.sp_hpf.setValue(20.0)
        self.sp_hpf.setDecimals(1)
        self.sp_hpf.setSuffix(" Hz")

        self.ck_lpf = QtWidgets.QCheckBox("Low-pass (LPF)")
        self.sp_lpf = QtWidgets.QDoubleSpinBox()
        self.sp_lpf.setRange(10.0, 1000.0)
        self.sp_lpf.setValue(450.0)
        self.sp_lpf.setDecimals(1)
        self.sp_lpf.setSuffix(" Hz")

        self.ck_notch = QtWidgets.QCheckBox("Notch")
        self.cb_notch_f = QtWidgets.QComboBox()
        self.cb_notch_f.addItems(["50", "60", "Custom"])
        self.cb_notch_f.setCurrentText("50")

        self.sp_notch = QtWidgets.QDoubleSpinBox()
        self.sp_notch.setRange(1.0, 500.0)
        self.sp_notch.setValue(50.0)
        self.sp_notch.setDecimals(1)
        self.sp_notch.setSuffix(" Hz")
        self.sp_notch.setEnabled(False)

        self.sp_notch_q = QtWidgets.QDoubleSpinBox()
        self.sp_notch_q.setRange(2.0, 200.0)
        self.sp_notch_q.setValue(30.0)
        self.sp_notch_q.setDecimals(1)

        # 一键自动检测工频峰值并设置陷波中心频率
        self.btn_auto_notch = QtWidgets.QPushButton("Auto-detect mains frequency and set")
        self.btn_auto_notch.clicked.connect(self._on_auto_notch)

        self.ck_hpf.setChecked(True)
        self.ck_lpf.setChecked(True)
        self.ck_notch.setChecked(True)

        self.cb_notch_f.currentTextChanged.connect(self._on_notch_sel)
        for w in (self.ck_hpf, self.ck_lpf, self.ck_notch, self.sp_hpf, self.sp_lpf, self.sp_notch, self.sp_notch_q):
            if isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self._rebuild_filters)
            else:
                w.valueChanged.connect(self._rebuild_filters)

        row = 0
        l_flt.addWidget(self.ck_hpf, row, 0);
        l_flt.addWidget(self.sp_hpf, row, 1);
        row += 1
        l_flt.addWidget(self.ck_lpf, row, 0);
        l_flt.addWidget(self.sp_lpf, row, 1);
        row += 1
        l_flt.addWidget(self.ck_notch, row, 0);
        l_flt.addWidget(self.cb_notch_f, row, 1);
        row += 1
        l_flt.addWidget(QtWidgets.QLabel("Notch center f0"), row, 0);
        l_flt.addWidget(self.sp_notch, row, 1);
        row += 1
        l_flt.addWidget(QtWidgets.QLabel("Notch Q"), row, 0);
        l_flt.addWidget(self.sp_notch_q, row, 1);
        row += 1
        l_flt.addWidget(self.btn_auto_notch, row, 0, 1, 2);
        row += 1

        # 当前滤波参数显示（便于确认是否生效）
        self.lb_hostflt_status = QtWidgets.QLabel("Current filter: -")
        self.lb_hostflt_status.setWordWrap(True)
        l_flt.addWidget(self.lb_hostflt_status, row, 0, 1, 2);
        row += 1
        self.lb_curve_hint = QtWidgets.QLabel("Curve colors: gray=raw  green=filtered  yellow=envelope (RMS)")
        self.lb_curve_hint.setWordWrap(True)
        l_flt.addWidget(self.lb_curve_hint, row, 0, 1, 2);
        row += 1
        v_flt.addWidget(g_flt)

        g_env = QtWidgets.QGroupBox("Envelope (RMS)")
        l_env = QtWidgets.QGridLayout(g_env)
        self.ck_env_show = QtWidgets.QCheckBox("Show envelope")
        self.ck_env_show.setChecked(True)
        self.ck_env_show.toggled.connect(self._on_env_show_changed)
        self.sp_env_ms = QtWidgets.QSpinBox()
        self.sp_env_ms.setRange(5, 500)
        self.sp_env_ms.setValue(50)
        self.sp_env_ms.setSuffix(" ms")
        self.sp_env_ms.valueChanged.connect(self._on_env_param_changed)
        l_env.addWidget(self.ck_env_show, 0, 0, 1, 2)
        l_env.addWidget(QtWidgets.QLabel("Window"), 1, 0);
        l_env.addWidget(self.sp_env_ms, 1, 1)
        v_flt.addWidget(g_env)

        # 显示（滚动窗口）
        g_view = QtWidgets.QGroupBox("Display (scrolling window)")
        l_view = QtWidgets.QGridLayout(g_view)
        self.sp_win_sec = QtWidgets.QDoubleSpinBox()
        self.sp_win_sec.setRange(1.0, 30.0)
        self.sp_win_sec.setDecimals(1)
        self.sp_win_sec.setValue(self.win_sec)
        self.sp_win_sec.setSuffix(" s")

        self.ck_autoscroll = QtWidgets.QCheckBox("Auto-scroll (no manual dragging needed)")
        self.ck_autoscroll.setChecked(True)

        self.ck_autoy = QtWidgets.QCheckBox("Auto-scale Y axis")
        self.ck_autoy.setChecked(True)
        self.ck_autoy.toggled.connect(self._reset_auto_y)

        l_view.addWidget(QtWidgets.QLabel("Window length"), 0, 0)
        l_view.addWidget(self.sp_win_sec, 0, 1)
        l_view.addWidget(self.ck_autoscroll, 1, 0, 1, 2)
        l_view.addWidget(self.ck_autoy, 2, 0, 1, 2)

        self.sp_win_sec.valueChanged.connect(self._on_win_sec_changed)

        v_flt.addWidget(g_view)

        v_flt.addStretch(1)
        tab_flt.setWidget(tab_flt_w)
        self.tabs.addTab(tab_flt, "Filtering")

        # ===== Tab 3: 尖峰 =====
        tab_spk = QtWidgets.QScrollArea()
        tab_spk.setWidgetResizable(True)
        tab_spk_w = QtWidgets.QWidget()
        v_spk = QtWidgets.QVBoxLayout(tab_spk_w)
        v_spk.setContentsMargins(6, 6, 6, 6)
        v_spk.setSpacing(8)

        g_gl = QtWidgets.QGroupBox("Spike threshold (device + host)")
        l_gl = QtWidgets.QGridLayout(g_gl)

        self.sp_glitch_abs = QtWidgets.QSpinBox()
        self.sp_glitch_abs.setRange(0, 500000)
        self.sp_glitch_abs.setValue(20000)
        self.sp_glitch_abs.setSuffix(" uV")

        self.sp_glitch_step = QtWidgets.QSpinBox()
        self.sp_glitch_step.setRange(0, 500000)
        self.sp_glitch_step.setValue(10000)
        self.sp_glitch_step.setSuffix(" uV")

        self.btn_apply_glitch = QtWidgets.QPushButton("Apply spike threshold only")
        self.btn_apply_glitch.clicked.connect(self._on_apply_glitch)

        row = 0
        l_gl.addWidget(QtWidgets.QLabel("Absolute threshold (GLITCHABS)"), row, 0);
        l_gl.addWidget(self.sp_glitch_abs, row, 1);
        row += 1
        l_gl.addWidget(QtWidgets.QLabel("Step threshold (GLITCHSTEP)"), row, 0);
        l_gl.addWidget(self.sp_glitch_step, row, 1);
        row += 1
        l_gl.addWidget(self.btn_apply_glitch, row, 0, 1, 2)

        v_spk.addWidget(g_gl)

        g_spk = QtWidgets.QGroupBox("Spike guard (host display/log)")
        l_spk2 = QtWidgets.QGridLayout(g_spk)
        self.ck_spike = QtWidgets.QCheckBox("Enable spike guard (replace points over threshold with last valid value)")
        self.ck_spike.setChecked(True)
        self.ck_blank_csv = QtWidgets.QCheckBox("Keep rows in CSV but mark emg_valid=0 (for post-processing)")
        self.ck_blank_csv.setChecked(False)
        l_spk2.addWidget(self.ck_spike, 0, 0, 1, 2)
        l_spk2.addWidget(self.ck_blank_csv, 1, 0, 1, 2)
        v_spk.addWidget(g_spk)

        v_spk.addStretch(1)
        tab_spk.setWidget(tab_spk_w)
        self.tabs.addTab(tab_spk, "Spikes")

        # ===== Tab 4: IMU =====
        tab_imu = QtWidgets.QScrollArea()
        tab_imu.setWidgetResizable(True)
        tab_imu_w = QtWidgets.QWidget()
        v_imu = QtWidgets.QVBoxLayout(tab_imu_w)
        v_imu.setContentsMargins(6, 6, 6, 6)
        v_imu.setSpacing(8)

        g_imu = QtWidgets.QGroupBox("IMU (100 Hz) - Magnetometer calibration + 9-axis fusion")
        l_imu = QtWidgets.QGridLayout(g_imu)

        self.sp_beta = QtWidgets.QDoubleSpinBox()
        self.sp_beta.setRange(0.001, 2.0)
        self.sp_beta.setSingleStep(0.01)
        self.sp_beta.setValue(0.10)

        self.btn_reset_q = QtWidgets.QPushButton("Reset orientation")
        self.btn_mag_start = QtWidgets.QPushButton("Start mag calibration")
        self.btn_mag_stop = QtWidgets.QPushButton("Stop & solve")
        self.btn_mag_save = QtWidgets.QPushButton("Save calibration")
        self.btn_mag_load = QtWidgets.QPushButton("Load calibration")
        self.lb_mag_info = QtWidgets.QLabel("Mag calibration: not calibrated")
        self.lb_mag_info.setWordWrap(True)

        self.btn_reset_q.clicked.connect(self._on_reset_q)
        self.btn_mag_start.clicked.connect(self._on_mag_start)
        self.btn_mag_stop.clicked.connect(self._on_mag_stop)
        self.btn_mag_save.clicked.connect(self._on_mag_save)
        self.btn_mag_load.clicked.connect(self._on_mag_load)
        self.sp_beta.valueChanged.connect(self._on_beta_changed)

        # Axis mapping UI
        def _axis_row(title: str):
            w = QtWidgets.QWidget()
            glay = QtWidgets.QGridLayout(w)
            glay.setContentsMargins(0, 0, 0, 0)
            glay.addWidget(QtWidgets.QLabel(title), 0, 0, 1, 6)
            combos = []
            flips = []
            for j, axname in enumerate(["X", "Y", "Z"]):
                glay.addWidget(QtWidgets.QLabel(axname + " <-"), 1, j * 2)
                cb = QtWidgets.QComboBox()
                cb.addItems(["X", "Y", "Z"])
                glay.addWidget(cb, 1, j * 2 + 1)
                ck = QtWidgets.QCheckBox("-")
                flips.append(ck)
                combos.append(cb)
                glay.addWidget(ck, 2, j * 2 + 1)
            return w, combos, flips

        self.w_map_ag, self.map_ag_cbs, self.map_ag_flips = _axis_row("Accel/Gyro axis mapping")
        self.w_map_m, self.map_m_cbs, self.map_m_flips = _axis_row("Magnetometer axis mapping")

        for cb in self.map_ag_cbs + self.map_m_cbs:
            cb.currentTextChanged.connect(self._on_axis_map_changed)
        for ck in self.map_ag_flips + self.map_m_flips:
            ck.toggled.connect(self._on_axis_map_changed)

        row = 0
        l_imu.addWidget(QtWidgets.QLabel("Madgwick β"), row, 0);
        l_imu.addWidget(self.sp_beta, row, 1);
        l_imu.addWidget(self.btn_reset_q, row, 2);
        row += 1
        l_imu.addWidget(self.btn_mag_start, row, 0);
        l_imu.addWidget(self.btn_mag_stop, row, 1);
        row += 1
        l_imu.addWidget(self.btn_mag_save, row, 0);
        l_imu.addWidget(self.btn_mag_load, row, 1);
        row += 1
        l_imu.addWidget(self.lb_mag_info, row, 0, 1, 3);
        row += 1
        l_imu.addWidget(self.w_map_ag, row, 0, 1, 3);
        row += 1
        l_imu.addWidget(self.w_map_m, row, 0, 1, 3);
        row += 1

        self.lb_imu_vals = QtWidgets.QLabel("IMU：-")
        self.lb_imu_vals.setWordWrap(True)
        l_imu.addWidget(self.lb_imu_vals, row, 0, 1, 3)

        v_imu.addWidget(g_imu)
        v_imu.addStretch(1)
        tab_imu.setWidget(tab_imu_w)
        self.tabs.addTab(tab_imu, "IMU")

        # ===== Tab 5: CSV / 日志 =====
        tab_log = QtWidgets.QWidget()
        v_log = QtWidgets.QVBoxLayout(tab_log)
        v_log.setContentsMargins(6, 6, 6, 6)
        v_log.setSpacing(8)

        g_csv = QtWidgets.QGroupBox("CSV Logging")
        l_csv = QtWidgets.QGridLayout(g_csv)
        self.ed_csv = QtWidgets.QLineEdit("emg_log.csv")
        self.btn_csv_on = QtWidgets.QPushButton("Start saving CSV")
        self.btn_csv_off = QtWidgets.QPushButton("Stop saving CSV")
        self.btn_csv_off.setEnabled(False)
        self.btn_csv_on.clicked.connect(self._on_csv_on)
        self.btn_csv_off.clicked.connect(self._on_csv_off)
        l_csv.addWidget(self.ed_csv, 0, 0, 1, 2)
        l_csv.addWidget(self.btn_csv_on, 1, 0)
        l_csv.addWidget(self.btn_csv_off, 1, 1)
        v_log.addWidget(g_csv)
        # 多设备版本：CSV统一在“总览/录制”页面集中控制，这里隐藏
        g_csv.setVisible(False)

        g_log = QtWidgets.QGroupBox("Log")
        l_log = QtWidgets.QVBoxLayout(g_log)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        l_log.addWidget(self.log)
        v_log.addWidget(g_log, 1)

        self.tabs.addTab(tab_log, "Log")

        # ---------------- Right: plots + 3D (splitter) ----------------
        right_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main.addWidget(right_split, 1)

        # EMG plot（原始 + 滤波后）
        self.plot_emg = pg.PlotWidget()
        self.plot_emg.showGrid(x=True, y=True)
        try:
            self.plot_emg.setLabel('bottom', 'Time', units='s')
            self.plot_emg.setLabel('left', 'Voltage', units='uV')
        except Exception:
            pass
        try:
            self.plot_emg.addLegend()
        except Exception:
            pass

        # 颜色区分：
        #   - CH1：灰(原始) / 绿(滤波后)
        #   - CH2：深灰(原始) / 蓝(滤波后)   <-- 仅双通道设备(12-17)显示
        self.curve_raw = self.plot_emg.plot(pen=pg.mkPen(color=(180, 180, 180), width=1), name="Raw (CH1)")
        self.curve_filt = self.plot_emg.plot(pen=pg.mkPen(color=(0, 255, 0), width=2), name="Filtered (CH1)")

        if self.dual_mode:
            self.curve_raw2 = self.plot_emg.plot(pen=pg.mkPen(color=(120, 120, 120), width=1), name="Raw (CH2)")
            self.curve_filt2 = self.plot_emg.plot(pen=pg.mkPen(color=(0, 120, 255), width=2), name="Filtered (CH2)")
        else:
            self.curve_raw2 = None
            self.curve_filt2 = None
        right_split.addWidget(self.plot_emg)

        # 包络线（单独窗口）
        self.plot_env = pg.PlotWidget()
        self.plot_env.showGrid(x=True, y=True)
        self.plot_env.setXLink(self.plot_emg)
        try:
            self.plot_env.setLabel('bottom', 'Time', units='s')
            self.plot_env.setLabel('left', 'Envelope', units='uV_RMS')
        except Exception:
            pass
        try:
            self.plot_env.addLegend()
        except Exception:
            pass
        # 包络线颜色：CH1=黄；CH2=紫（仅双通道设备显示）
        self.curve_env = self.plot_env.plot(pen=pg.mkPen(color=(255, 200, 0), width=2), name="Envelope (CH1, RMS)")
        if self.dual_mode:
            self.curve_env2 = self.plot_env.plot(pen=pg.mkPen(color=(200, 0, 255), width=2), name="Envelope (CH2, RMS)")
        else:
            self.curve_env2 = None
        # 绘图性能优化：自动下采样 + 仅绘制可视区域（可显著降低CPU占用）
        curves_for_perf = [self.curve_raw, self.curve_filt, self.curve_env]
        if self.curve_raw2 is not None: curves_for_perf.append(self.curve_raw2)
        if self.curve_filt2 is not None: curves_for_perf.append(self.curve_filt2)
        if self.curve_env2 is not None: curves_for_perf.append(self.curve_env2)
        for _c in curves_for_perf:
            try:
                _c.setDownsampling(auto=True, mode='peak')
                _c.setClipToView(True)
            except Exception:
                pass
        right_split.addWidget(self.plot_env)

        # 3D view
        if HAS_GL:
            self.glw = gl.GLViewWidget()
            self.glw.opts['distance'] = 5
            try:
                self.glw.setBackgroundColor((15, 15, 15))
            except Exception:
                pass
            try:
                self.glw.opts['center'] = QtGui.QVector3D(0, 0, 0)
            except Exception:
                pass

            # 网格
            gx = gl.GLGridItem()
            gx.scale(1, 1, 1)
            self.glw.addItem(gx)

            # 坐标轴（红/绿/蓝，固定使用自定义线条：不同pyqtgraph版本的GLAxisItem颜色/粗细不一致且不够醒目）
            try:
                axis_len = 2.0
                axis_w = 4
                # X(red), Y(green), Z(blue)
                pts = np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=float)
                self.glw.addItem(gl.GLLinePlotItem(pos=pts, color=(1, 0, 0, 1), width=axis_w, antialias=True))
                pts = np.array([[0, 0, 0], [0, axis_len, 0]], dtype=float)
                self.glw.addItem(gl.GLLinePlotItem(pos=pts, color=(0, 1, 0, 1), width=axis_w, antialias=True))
                pts = np.array([[0, 0, 0], [0, 0, axis_len]], dtype=float)
                self.glw.addItem(gl.GLLinePlotItem(pos=pts, color=(0, 0, 1, 1), width=axis_w, antialias=True))
                # 文字标签（可选：某些版本可能没有GLTextItem）
                try:
                    txt = gl.GLTextItem(pos=(axis_len, 0, 0), text='X', color=(255, 80, 80, 255))
                    self.glw.addItem(txt)
                    txt = gl.GLTextItem(pos=(0, axis_len, 0), text='Y', color=(80, 255, 80, 255))
                    self.glw.addItem(txt)
                    txt = gl.GLTextItem(pos=(0, 0, axis_len), text='Z', color=(80, 80, 255, 255))
                    self.glw.addItem(txt)
                except Exception:
                    pass
            except Exception:
                pass

            # 方向方块
            self.cube = gl.GLBoxItem(size=QtGui.QVector3D(1, 1, 1))
            self.glw.addItem(self.cube)
            right_split.addWidget(self.glw)
        else:
            self.glw = None
            self.cube = None
            right_split.addWidget(QtWidgets.QLabel("pyqtgraph.opengl not installed - 3D view unavailable"))

        # initial sizes (user can drag)
        try:
            right_split.setSizes([450, 180, 330])
        except Exception:
            pass

    # ---------- Callbacks ----------
    def _log(self, s: str) -> None:
        t = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{t}] {s}")

    def _parse_kv_line(self, line: str) -> dict:
        # Parse "CFG: k=v k=v ..." or similar.
        # Returns dict with numeric values when possible.
        out = {}
        try:
            # remove prefix
            if ":" in line:
                _, rest = line.split(":", 1)
            else:
                rest = line
            parts = rest.strip().split()
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                k = k.strip()
                v = v.strip()
                # strip unit suffix like 'g' or 'dps'
                v_num = v
                for suf in ("dps", "Hz", "g", "uV", "mV"):
                    if v_num.endswith(suf):
                        v_num = v_num[: -len(suf)]
                        break
                # try int/float
                try:
                    if "." in v_num:
                        out[k] = float(v_num)
                    else:
                        out[k] = int(v_num)
                except Exception:
                    out[k] = v
        except Exception:
            pass
        return out


    def _parse_kv_line_raw(self, line: str) -> dict:
        # Parse "XXX: k=v k=v ..." but keep values as raw strings (不要把纯数字密码/SSID自动转成 int)
        out = {}
        try:
            if ":" in line:
                _, rest = line.split(":", 1)
            else:
                rest = line
            parts = rest.strip().split()
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
        except Exception:
            pass
        return out


    def _compare_cfg(self, req: dict, cfg: dict) -> Tuple[bool, str]:
        # Compare requested device config against readback CFG.
        keys = [
            ("fs", "fs"),
            ("gain", "gain"),
            ("test", "test"),
            ("short", "short"),
            ("rld", "rld"),
            ("ch1_en", "ch1_en"),
            ("ch2_en", "ch2_en"),
            ("imu", "imu"),
            ("imu_rate", "imu_rate"),
            ("acc_fs", "acc_fs"),
            ("gyro_fs", "gyro_fs"),
            ("glitch_abs_uV", "glitch_abs_uV"),
            ("glitch_step_uV", "glitch_step_uV"),
        ]
        mism = []
        for rk, ck in keys:
            if rk not in req:
                continue
            rv = req.get(rk)
            cv = cfg.get(ck)
            if cv is None:
                # 兼容旧固件：可能没有回读 ch1_en/ch2_en
                if rk in ("ch1_en", "ch2_en"):
                    continue
                mism.append(f"{rk}=? (expected {rv})")
                continue
            if str(rv) != str(cv):
                mism.append(f"{rk}={cv} (expected {rv})")
        ok = (len(mism) == 0)
        if ok:
            return True, f"FS={req.get('fs')} Gain={req.get('gain')} TEST={req.get('test')} SHORT={req.get('short')} RLD={req.get('rld')} IMU={req.get('imu')}({req.get('imu_rate')}Hz) ACC={req.get('acc_fs')}g GYR={req.get('gyro_fs')}dps"
        else:
            return False, "；".join(mism)

    def _on_text(self, s: str) -> None:
        # Firmware already sends lines without brackets
        self._log(s)

        # 记录固件版本（VER? 返回）
        if "EMG_FW" in s:
            self.dev_fw = s.strip()
            try:
                self.lb_dev_status.setText(f"Device: {self.dev_fw}")
            except Exception:
                pass

        # Auto update / verify status from CFG/IMUCFG/GLITCH lines
        if s.startswith("CFG:"):
            cfg = self._parse_kv_line(s)
            self._last_verified_cfg = cfg
            # 在界面上用更直观的中文显示
            try:
                fs = cfg.get("fs", "?")
                gain = cfg.get("gain", "?")
                test = cfg.get("test", "?")
                short = cfg.get("short", "?")
                rld = cfg.get("rld", "?")
                stream = cfg.get("stream", "?")
                imu = cfg.get("imu", "?")
                imu_rate = cfg.get("imu_rate", "?")
                acc_fs = cfg.get("acc_fs", "?")
                gyro_fs = cfg.get("gyro_fs", "?")
                ga = cfg.get("glitch_abs_uV", cfg.get("glitch_abs", "?"))
                gs = cfg.get("glitch_step_uV", cfg.get("glitch_step", "?"))
                self.lb_dev_status.setText(
                    f"Device config: FS={fs} SPS  Gain={gain}  TEST={test}  SHORT={short}  RLD={rld}  STREAM={stream}\n"
                    f"IMU={imu}  IMU_RATE={imu_rate}Hz  ACC={acc_fs}g  GYR={gyro_fs}dps  GLITCHABS={ga}uV  GLITCHSTEP={gs}uV"
                )
                # 同步通道使能状态（新固件才会返回 ch1_en/ch2_en）
                if self.dual_mode:
                    ch1_en = cfg.get("ch1_en", None)
                    ch2_en = cfg.get("ch2_en", None)

                    if ch1_en is not None:
                        try:
                            self.ch1_enabled = bool(int(ch1_en))
                            if self.ck_ch1_en is not None:
                                self.ck_ch1_en.blockSignals(True)
                                self.ck_ch1_en.setChecked(self.ch1_enabled)
                                self.ck_ch1_en.blockSignals(False)
                        except Exception:
                            pass

                    if ch2_en is not None:
                        try:
                            self.ch2_enabled = bool(int(ch2_en))
                            if self.ck_ch2_en is not None:
                                self.ck_ch2_en.blockSignals(True)
                                self.ck_ch2_en.setChecked(self.ch2_enabled)
                                self.ck_ch2_en.blockSignals(False)
                        except Exception:
                            pass

                    # 立即把“CH2关闭”反映到显示
                    try:
                        self._apply_ch_enable_state_to_view(rebuild_filters=False)
                    except Exception:
                        pass

                    # 在状态栏追加显示（避免看不到当前通道状态）
                    try:
                        if (ch1_en is not None) or (ch2_en is not None):
                            self.lb_dev_status.setText(self.lb_dev_status.text() + f"\nCH1_EN={ch1_en}  CH2_EN={ch2_en}")
                    except Exception:
                        pass

            except Exception:
                # fallback raw
                try:
                    self.lb_dev_status.setText(s)
                except Exception:
                    pass

            # 对比“刚刚请求的参数” -> 给出成功/失败提示
            if self._pending_req is not None and (time.time() - self._pending_req_time) < 5.0:
                ok, report = self._compare_cfg(self._pending_req, cfg)
                if ok:
                    self._log("✅ Device settings applied:" + report)
                else:
                    self._log("⚠️ Device settings not fully applied:" + report)
                # 清掉 pending，避免重复提示
                self._pending_req = None

        elif s.startswith("NETCFG:"):
            # 解析并回填到“网络/标识配置（扩展）”区域
            d = self._parse_kv_line_raw(s)
            try:
                if hasattr(self, "ed_net_ssid"):
                    self.ed_net_ssid.setText(d.get("ssid", ""))
                    self.ed_net_pass.setText(d.get("pass", ""))
                    self.ed_net_tcp.setText(d.get("tcp", ""))
                    self.ed_net_id.setText(d.get("id", ""))
                    self.ed_net_label.setText(d.get("label", ""))
            except Exception:
                pass

        elif s.startswith("IMUCFG:") or s.startswith("GLITCH:"):
            # 也显示在设备状态栏（便于查看）
            try:
                # 追加显示（不覆盖 CFG 的两行信息）
                cur = self.lb_dev_status.text() if hasattr(self, 'lb_dev_status') else ""
                if s.startswith("IMUCFG:"):
                    self.lb_dev_status.setText(cur + "\n" + s)
                elif s.startswith("GLITCH:"):
                    self.lb_dev_status.setText(cur + "\n" + s)
            except Exception:
                pass

    def _on_stat(self, d: dict) -> None:
        # status packet
        pass

    def _on_bat(self, d: dict) -> None:
        """Battery packet from firmware.

        Firmware sends:
          ms(u32) + vbat_mV(u16) + pct(u8) + adc_mV(u16) + reason(u8)

        We apply an extra light EMA on host-side to make the UI less jumpy.
        """
        try:
            vbat_mV = int(d.get("vbat_mV", 0))
            pct_fw = int(d.get("pct", 0))
            adc_mV = int(d.get("adc_mV", 0))
        except Exception:
            return

        if vbat_mV <= 0:
            self._bat_ema_v = None
            self._bat_ema_pct = None
            self.lb_bat.setText("—")
            self.pb_bat.setValue(0)
            return

        v_raw = vbat_mV / 1000.0

        # Host-side EMA to reduce visible jitter (does not affect logged value)
        if self._bat_ema_v is None:
            self._bat_ema_v = v_raw
        else:
            self._bat_ema_v = self._bat_ema_v * 0.80 + v_raw * 0.20

        v_show = float(self._bat_ema_v)

        # Recompute percent from smoothed voltage, clamp to 0..100
        pct = int(round((v_show - 2.8) * 100.0 / (4.1 - 2.8)))
        if pct < 0:
            pct = 0
        if pct > 100:
            pct = 100

        # 保存到成员变量：供“总览/电量/录制”页显示
        if self._bat_ema_pct is None:
            self._bat_ema_pct = float(pct)
        else:
            self._bat_ema_pct = self._bat_ema_pct * 0.80 + float(pct) * 0.20

        self.lb_bat.setText(f"{v_show:.2f} V ({pct}%)")
        self.pb_bat.setValue(max(0, min(100, pct)))

        self.pb_bat.setToolTip(
            f"RAW: {v_raw:.3f} V (FW pct={pct_fw}%)\n"
            f"ADC: {adc_mV} mV\n"
            "Divider: 100k/100k → theoretical VBAT≈2×ADC\n"
            "Note: If you power via a 5V supply/long cable, the board voltage may read noticeably lower due to cable loss and load."
        )

    # ---------- UDP 扫描（用于 STA/路由器模式快速找到设备 IP） ----------
    def _discover_devices(self, timeout_s: float = DISCOVERY_TIMEOUT_S) -> List[dict]:
        # 说明：
        #  - 向 255.255.255.255:DISCOVERY_PORT 广播 DISCOVERY_MAGIC
        #  - 固件收到后会回一个 ASCII 行（带 id/label/mac/ip/tcp...）
        #  - 若扫描不到：检查 PC 防火墙是否拦截 UDP、路由器是否隔离客户端、或先手动填 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.15)
        try:
            s.bind(("", 0))
        except Exception:
            pass

        try:
            try:
                s.sendto(DISCOVERY_MAGIC, ("255.255.255.255", int(DISCOVERY_PORT)))
            except Exception:
                # 某些系统用 <broadcast> 更稳
                s.sendto(DISCOVERY_MAGIC, ("<broadcast>", int(DISCOVERY_PORT)))

            t0 = time.time()
            found = {}

            while (time.time() - t0) < float(timeout_s):
                try:
                    data, addr = s.recvfrom(512)
                except socket.timeout:
                    continue
                except Exception:
                    break

                txt = data.decode("utf-8", errors="ignore").strip()
                if not txt.startswith(DISCOVERY_REPLY_PREFIX):
                    continue

                d = {"ip": addr[0]}
                parts = txt.split()
                for token in parts[1:]:
                    if "=" in token:
                        k, v = token.split("=", 1)
                        d[k] = v

                key = d.get("mac") or d.get("ip")
                found[key] = d

            return list(found.values())
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _apply_scan_device(self, idx: int) -> None:
        if idx < 0:
            return
        if not hasattr(self, "_found_devices"):
            return
        if idx >= len(self._found_devices):
            return
        d = self._found_devices[idx]
        ip = d.get("ip", "")
        if ip:
            self.ed_ip.setText(str(ip))
        tcp = d.get("tcp", "")
        if tcp:
            self.ed_port.setText(str(tcp))

    def _on_scan_devices(self) -> None:
        try:
            devs = self._discover_devices(timeout_s=DISCOVERY_TIMEOUT_S)
        except Exception as e:
            self._log(f"Scan failed: {e}")
            devs = []

        self._found_devices = devs

        self.cb_scan.blockSignals(True)
        self.cb_scan.clear()
        if not devs:
            self.cb_scan.addItem("(No devices found; check firewall/subnet/router isolation)")
        else:
            for d in devs:
                label = d.get("label", "?")
                did = d.get("id", "?")
                ip = d.get("ip", "?")
                mac = d.get("mac", "")
                rssi = d.get("rssi", "")
                fw = d.get("fw", "")
                show = f"{label}  id={did}  ip={ip}  rssi={rssi}  mac={mac}  {fw}"
                self.cb_scan.addItem(show)

        self.cb_scan.blockSignals(False)

        if devs:
            self.cb_scan.setCurrentIndex(0)
            self._apply_scan_device(0)

        self._log(f"Scan complete: {len(devs)} devices")

    def _on_scan_select(self, idx: int) -> None:
        self._apply_scan_device(idx)

    def _on_connect(self) -> None:
        host = self.ed_ip.text().strip()
        try:
            port = int(self.ed_port.text().strip())
        except ValueError:
            port = 3333
        ok = self.rx.connect_to(host, port)
        if ok:
            self.connected = True
            self.btn_conn.setEnabled(False)
            self.btn_disc.setEnabled(True)
            self._log(f"Connected to {host}:{port}")
            # ask version
            self.rx.send_line("VER?")
            self.rx.send_line("CFG?")
            self.rx.send_line("IMUCFG?")
            self.rx.send_line("GLITCH?")
        else:
            self.connected = False

    def _on_disconnect(self) -> None:
        self._on_stop()
        self.rx.disconnect()
        self.connected = False
        self.btn_conn.setEnabled(True)
        self.btn_disc.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        try:
            self.lb_bat.setText("—")
            self.pb_bat.setValue(0)
        except Exception:
            pass
        self._log("Disconnected")

    def _send_line_cmd(self, line: str) -> None:
        if not self.connected:
            self._log("Not connected")
            return
        self.rx.send_line(line)

    # -------- 网络/标识配置（扩展命令） --------
    def _on_net_query(self) -> None:
        # 查询设备当前保存的 WiFi/端口/ID/LABEL（以及当前网络状态）
        self._send_line_cmd("NETCFG?")

    def _on_net_set_wifi(self) -> None:
        # 发送：SET_WIFI <ssid> <pass> [tcp=<port>]
        # 注意：ssid/pass 不支持包含空格（下位机按空格分隔参数）；空密码可用 "-"。
        if not self.connected:
            self._log("Not connected")
            return

        ssid = self.ed_net_ssid.text().strip()
        pwd = self.ed_net_pass.text()
        tcp = self.ed_net_tcp.text().strip()

        if not ssid:
            self._log("❌ SSID cannot be empty.")
            return
        if " " in ssid:
            self._log("❌ SSID must not contain spaces (current firmware commands are space-delimited).")
            return
        if " " in pwd:
            self._log("❌ Password contains spaces: firmware commands are space-delimited; avoid spaces in the password.")
            return

        pwd_send = "-" if pwd == "" else pwd
        cmd = f"SET_WIFI {ssid} {pwd_send}"
        if tcp:
            cmd += f" tcp={tcp}"

        self._log("⚠️ Sending SET_WIFI: the device will reboot/switch network; the TCP connection will drop. Please rescan/reconnect afterwards.")
        self.rx.send_line(cmd)

    def _on_net_set_device(self) -> None:
        # 发送：SET_DEVICE <id> <label>
        if not self.connected:
            self._log("Not connected")
            return

        did = self.ed_net_id.text().strip()
        label = self.ed_net_label.text().strip()

        if did == "" or label == "":
            self._log("❌ Device ID and LABEL must not be empty.")
            return
        if " " in label:
            self._log("❌ LABEL must not contain spaces (scan fields are space-delimited).")
            return

        cmd = f"SET_DEVICE {did} {label}"
        self.rx.send_line(cmd)
        # 读回验证
        self.rx.send_line("NETCFG?")

    def _send_lines_cmd(self, lines: List[str]) -> None:
        if not self.connected:
            self._log("Not connected")
            return
        self.rx.send_lines(lines)

    def _on_apply_glitch(self) -> None:
        if not self.connected:
            self._log("Not connected")
            return
        abs_uv = int(self.sp_glitch_abs.value())
        step_uv = int(self.sp_glitch_step.value())
        self._log(f"Applied spike thresholds: ABS={abs_uv}uV STEP={step_uv}uV")
        self.rx.send_lines([f"GLITCHABS {abs_uv}", f"GLITCHSTEP {step_uv}", "GLITCH?"])

    def _on_env_param_changed(self) -> None:
        # RMS window length changed -> reset envelope state
        self._update_env_n(reset=True)

    def _on_win_sec_changed(self) -> None:
        try:
            new_sec = float(self.sp_win_sec.value())
        except Exception:
            return
        self._set_window_seconds(new_sec)

    def _reset_auto_y(self) -> None:
        self._auto_y_emg = None
        self._auto_y_env = None

    def _set_window_seconds(self, win_sec: float) -> None:
        win_sec = float(max(1.0, min(30.0, win_sec)))
        if abs(win_sec - float(self.win_sec)) < 1e-6:
            return
        self.win_sec = win_sec
        maxlen = int(self.win_sec * max(1, self.cur_fs))
        # 重新设置缓存长度（尽量保留最近的数据）
        try:
            t_old = list(self.buf_t)
            raw_old = list(self.buf_raw)
            fil_old = list(self.buf_filt)
            env_old = list(self.buf_env)
        except Exception:
            t_old, raw_old, fil_old, env_old = [], [], [], []
        self.buf_t = deque(t_old[-maxlen:], maxlen=maxlen)
        self.buf_raw = deque(raw_old[-maxlen:], maxlen=maxlen)
        self.buf_filt = deque(fil_old[-maxlen:], maxlen=maxlen)
        self.buf_env = deque(env_old[-maxlen:], maxlen=maxlen)
        self._log(f"Display window set to {self.win_sec:.1f} s")

    def _on_env_show_changed(self, v: bool) -> None:
        try:
            self.plot_env.setVisible(bool(v))
        except Exception:
            pass

    def _update_env_n(self, reset: bool = False) -> None:
        try:
            ms = float(self.sp_env_ms.value())
        except Exception:
            ms = 50.0
        fs = float(self.cur_fs if self.cur_fs > 0 else 1000)
        n = max(1, int(round(ms / 1000.0 * fs)))
        if reset or n != getattr(self, "_env_n", n):
            self._env_n = n
            try:
                self._env_sq.clear()
            except Exception:
                self._env_sq = deque()
            self._env_sum = 0.0
            # 同步重置 CH2 的包络状态
            try:
                self._env_sq2.clear()
            except Exception:
                self._env_sq2 = deque()
            self._env_sum2 = 0.0

    def _on_apply(self) -> None:
        if not self.connected:
            self._log("Not connected")
            return
        # 记录本次请求参数，并等待 CFG? 回读做确认
        req = self._current_device_req(stream_start=False)
        self._pending_req = req
        self._pending_req_time = time.time()
        self._log(
            f"Sent device settings:FS={req['fs']} SPS Gain={req['gain']} TEST={req['test']} SHORT={req['short']} RLD={req['rld']} IMU={req['imu']} ACC={req['acc_fs']}g GYR={req['gyro_fs']}dps GLITCHABS={req['glitch_abs_uV']}uV GLITCHSTEP={req['glitch_step_uV']}uV")

        cmds = self._build_apply_cmds(stream_start=False)
        self.rx.send_lines(cmds)

    def _on_start(self) -> None:
        if not self.connected:
            self._log("Not connected")
            return
        self.streaming = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.t0_plot = time.time()
        self._base_start_idx = None
        self.buf_raw.clear();
        self.buf_filt.clear();
        self.buf_env.clear();
        self.buf_t.clear()
        self.buf_raw2.clear();
        self.buf_filt2.clear();
        self.buf_env2.clear()
        try:
            self._env_sq.clear()
        except Exception:
            self._env_sq = deque()
        self._env_sum = 0.0
        try:
            self._env_sq2.clear()
        except Exception:
            self._env_sq2 = deque()
        self._env_sum2 = 0.0
        self.last_good_uv_ch1 = 0.0
        self.last_good_uv_ch2 = 0.0

        req = self._current_device_req(stream_start=True)
        self._pending_req = req
        self._pending_req_time = time.time()
        self._log(
            f"Starting acquisition, sent device settings:FS={req['fs']} SPS Gain={req['gain']} TEST={req['test']} SHORT={req['short']} RLD={req['rld']} IMU={req['imu']} ACC={req['acc_fs']}g GYR={req['gyro_fs']}dps GLITCHABS={req['glitch_abs_uV']}uV GLITCHSTEP={req['glitch_step_uV']}uV")

        cmds = self._build_apply_cmds(stream_start=True)
        self.rx.send_lines(cmds)

    def _on_stop(self) -> None:
        if not self.connected:
            self.streaming = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return
        self.rx.send_line("STREAM 0")
        self.streaming = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_csv_on(self) -> None:
        path = self.ed_csv.text().strip()
        if not path:
            path = "emg_log.csv"
            self.ed_csv.setText(path)

        meta = [
            f"FW/GUI: v9.6.1",
            f"DeviceFW: {self.dev_fw if self.dev_fw else 'unknown'}",
            f"VREF_V={VREF_V}",
            f"HostFilters: HPF={'ON' if self.ck_hpf.isChecked() else 'OFF'} fc={self.sp_hpf.value():.2f}Hz; "
            f"LPF={'ON' if self.ck_lpf.isChecked() else 'OFF'} fc={self.sp_lpf.value():.2f}Hz; "
            f"Notch={'ON' if self.ck_notch.isChecked() else 'OFF'} f0={self._get_notch_f0():.2f}Hz Q={self.sp_notch_q.value():.2f}",
            f"SpikeGuard: {'ON' if self.ck_spike.isChecked() else 'OFF'} abs_uV={self.sp_glitch_abs.value()} step_uV={self.sp_glitch_step.value()}",
            f"DeviceReq: fs={self.cb_fs.currentText()} gain={self.cb_gain.currentText()} test={int(self.ck_test.isChecked())} short={int(self.ck_short.isChecked())} rld={int(self.ck_rld.isChecked())} imu={int(self.ck_imu.isChecked())} acc_fs={self.cb_imu_acc.currentText()}g gyro_fs={self.cb_imu_gyr.currentText()}dps",
            f"Fusion: beta={self.fusion.beta:.3f}",
            f"QuatOut: relative_to_ref (q_out = q_ref * q_abs)",
            f"QuatRef(q_ref): {getattr(self, 'q_ref', np.array([1, 0, 0, 0])).tolist()}",
            f"AxisMap_AG: {self.axis_ag.cfg}",
            f"AxisMap_MAG: {self.axis_mag.cfg}",
            f"MagCal: valid={int(self.mag_cal.valid)} offset={self.mag_cal.offset.tolist()} scale={self.mag_cal.scale.tolist()}",
        ]
        emg_path = path
        imu_path = (path[:-4] + '_imu.csv') if path.lower().endswith('.csv') else (path + '_imu.csv')
        self.csv.start(emg_path, imu_path, meta_lines=meta)
        self.btn_csv_on.setEnabled(False)
        self.btn_csv_off.setEnabled(True)
        self._log(f"CSV saving enabled: EMG={emg_path}  IMU={imu_path}")

    def _on_csv_off(self) -> None:
        self.csv.stop()
        self.btn_csv_on.setEnabled(True)
        self.btn_csv_off.setEnabled(False)
        self._log("CSV logging OFF")

    def _on_notch_sel(self) -> None:
        if self.cb_notch_f.currentText() == "Custom":
            self.sp_notch.setEnabled(True)
        else:
            self.sp_notch.setEnabled(False)
            self.sp_notch.setValue(float(self.cb_notch_f.currentText()))
        self._rebuild_filters()

    def _on_auto_notch(self) -> None:
        """自动检测工频峰值，并把陷波中心频率 f0 设置为检测值。

        原理：对最近几s的“Raw (CH1)”做FFT，在 45~55Hz 或 55~65Hz 区间寻找最大峰值。
        只改变主机端显示/分析Filtering，不会改动下位机Acquisition数据。
        """
        try:
            fs = int(self.cur_fs) if self.cur_fs else 1000
        except Exception:
            fs = 1000
        fs = max(1, fs)

        raw = np.asarray(list(self.buf_raw), dtype=np.float64)
        if raw.size < int(2 * fs):
            self._log("Auto-detect mains frequency failed: not enough data (need at least 2 s of raw data). Please start acquisition first.")
            return

        # Use up to last 10 seconds for better frequency resolution
        n = min(raw.size, int(10 * fs))
        x = raw[-n:].copy()
        x -= float(np.mean(x))

        # Windowed FFT
        w = np.hanning(x.size)
        X = np.fft.rfft(x * w)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

        sel = self.cb_notch_f.currentText().strip()
        if sel == "60":
            f_lo, f_hi = 55.0, 65.0
        elif sel == "50":
            f_lo, f_hi = 45.0, 55.0
        else:
            f_lo, f_hi = 45.0, 65.0

        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            self._log("Auto-detect mains frequency failed: invalid spectrum search range.")
            return

        idxs = np.where(mask)[0]
        kk = idxs[int(np.argmax(np.abs(X[mask])))]
        f_peak = float(freqs[kk])

        # Amplitude estimate (uV peak) for log reference:
        # A ≈ 2*|X[k]| / sum(window)
        A_uVpk = float(2.0 * np.abs(X[kk]) / max(1e-9, np.sum(w)))

        f_set = float(round(f_peak, 1))

        # Apply to UI + rebuild
        try:
            self.cb_notch_f.setCurrentText("Custom")
        except Exception:
            pass
        self.ck_notch.setChecked(True)
        self.sp_notch.setEnabled(True)
        self.sp_notch.setValue(f_set)
        self._rebuild_filters()

        self._log(
            f"Auto-detected mains frequency: peak {f_peak:.2f} Hz (~{A_uVpk:.2f} µVpk). Set notch f0={f_set:.1f} Hz, Q={self.sp_notch_q.value():.1f}")

    def _on_beta_changed(self) -> None:
        self.fusion.beta = float(self.sp_beta.value())

    def _on_reset_q(self) -> None:
        # 复位“参考姿态”：不重置算法内部状态，只把当前姿态当作零点输出
        try:
            q_abs = self.fusion.q.copy()
            self.q_ref = quat_conj(q_abs)
            self.latest_q = quat_mul(self.q_ref, q_abs)
        except Exception:
            self.q_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.latest_q = self.q_ref.copy()
        self._log("IMU reference orientation reset (output quaternion now uses the current pose as zero).")

    def _on_mag_start(self) -> None:
        self.mag_cal_samples.clear()
        self.mag_cal_running = True
        self.lb_mag_info.setText("Mag cal: collecting... rotate device in all directions")
        self._log("Mag calibration: START (collecting)")

    def _on_mag_stop(self) -> None:
        self.mag_cal_running = False
        try:
            if len(self.mag_cal_samples) < 50:
                self.lb_mag_info.setText(f"Mag cal: not enough samples ({len(self.mag_cal_samples)})")
                self._log("Mag calibration: not enough samples")
                return
            samples = np.stack(self.mag_cal_samples, axis=0)
            self.mag_cal.solve_minmax(samples)
            self.lb_mag_info.setText(
                f"Mag cal OK. offset={self.mag_cal.offset.round(3).tolist()} scale={self.mag_cal.scale.round(3).tolist()} samples={len(self.mag_cal_samples)}"
            )
            self._log("Mag calibration: SOLVED and applied")
        except Exception as e:
            self.lb_mag_info.setText(f"Mag cal failed: {e}")
            self._log(f"Mag calibration failed: {e}")

    def _on_mag_save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mag calibration", "mag_cal.json", "JSON (*.json)")
        if not path:
            return
        d = {
            "mag_cal": self.mag_cal.to_dict(),
            "axis_ag": self.axis_ag.cfg,
            "axis_mag": self.axis_mag.cfg,
        }
        import json
        with open(path, "w", encoding="utf-8-sig") as f:
            json.dump(d, f, indent=2)
        self._log(f"Saved mag cal to {path}")

    def _on_mag_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load mag calibration", "", "JSON (*.json)")
        if not path:
            return
        import json
        with open(path, "r", encoding="utf-8-sig") as f:
            d = json.load(f)
        if "mag_cal" in d:
            self.mag_cal.from_dict(d["mag_cal"])
        # axis maps (optional)
        if "axis_ag" in d:
            self.axis_ag.set_cfg(tuple(tuple(x) for x in d["axis_ag"]))  # type: ignore
        if "axis_mag" in d:
            self.axis_mag.set_cfg(tuple(tuple(x) for x in d["axis_mag"]))  # type: ignore
        self._apply_axis_map_ui_from_state()
        self.lb_mag_info.setText("Mag cal loaded")
        self._log(f"Loaded mag cal from {path}")

    def _apply_axis_map_ui_from_state(self) -> None:
        def set_widgets(cfg, cbs, flips):
            for i, (src, sgn) in enumerate(cfg):
                cbs[i].setCurrentIndex(int(src))
                flips[i].setChecked(bool(sgn < 0))

        set_widgets(self.axis_ag.cfg, self.map_ag_cbs, self.map_ag_flips)
        set_widgets(self.axis_mag.cfg, self.map_m_cbs, self.map_m_flips)

    def _on_axis_map_changed(self) -> None:
        def read_widgets(cbs, flips):
            cfg = []
            for i in range(3):
                src = cbs[i].currentIndex()
                sgn = -1 if flips[i].isChecked() else +1
                cfg.append((int(src), int(sgn)))
            return tuple(cfg)  # type: ignore

        self.axis_ag.set_cfg(read_widgets(self.map_ag_cbs, self.map_ag_flips))
        self.axis_mag.set_cfg(read_widgets(self.map_m_cbs, self.map_m_flips))

    # ---------- Device command build ----------
    def _current_device_req(self, stream_start: bool) -> dict:
        # 用于“应用成功/失败”反馈
        fs = int(self.cb_fs.currentText())
        gain = int(self.cb_gain.currentText())
        test = 1 if self.ck_test.isChecked() else 0
        short = 1 if self.ck_short.isChecked() else 0
        rld = 1 if self.ck_rld.isChecked() else 0
        imu = 1 if self.ck_imu.isChecked() else 0
        acc = int(self.cb_imu_acc.currentText())
        gyr = int(self.cb_imu_gyr.currentText())
        glitch_abs = int(self.sp_glitch_abs.value())
        glitch_step = int(self.sp_glitch_step.value())

        req = {
            'fs': fs,
            'gain': gain,
            'test': test,
            'short': short,
            'rld': rld,
            'imu': imu,
            'imu_rate': 100,
            'acc_fs': acc,
            'gyro_fs': gyr,
            'glitch_abs_uV': glitch_abs,
            'glitch_step_uV': glitch_step,
            'stream': 1 if stream_start else 0,
        }

        # 双通道：通道使能（CH1/CH2）
        if self.dual_mode:
            try:
                req['ch1_en'] = 1 if (self.ck_ch1_en is None or self.ck_ch1_en.isChecked()) else 0
                req['ch2_en'] = 1 if (self.ck_ch2_en is None or self.ck_ch2_en.isChecked()) else 0
            except Exception:
                req['ch1_en'] = 1
                req['ch2_en'] = 1

        return req

    def _build_apply_cmds(self, stream_start: bool) -> List[str]:
        fs = int(self.cb_fs.currentText())
        gain = int(self.cb_gain.currentText())
        test = 1 if self.ck_test.isChecked() else 0
        short = 1 if self.ck_short.isChecked() else 0
        rld = 1 if self.ck_rld.isChecked() else 0
        imu = 1 if self.ck_imu.isChecked() else 0
        acc = int(self.cb_imu_acc.currentText())
        gyr = int(self.cb_imu_gyr.currentText())
        glitch_abs = int(self.sp_glitch_abs.value())
        glitch_step = int(self.sp_glitch_step.value())

        cmds = []
        cmds.append("STREAM 0")
        cmds.append(f"FS {fs}")
        cmds.append(f"GAIN {gain}")
        cmds.append(f"TEST {test}")
        cmds.append(f"SHORT {short}")
        cmds.append(f"RLD {rld}")
        # 双通道设备：下发 CH1/CH2 使能
        if self.dual_mode:
            ch1_en = 1 if (self.ck_ch1_en is None or self.ck_ch1_en.isChecked()) else 0
            ch2_en = 1 if (self.ck_ch2_en is None or self.ck_ch2_en.isChecked()) else 0
            cmds.append(f"CH1 {ch1_en}")
            cmds.append(f"CH2 {ch2_en}")
        cmds.append(f"GLITCHABS {glitch_abs}")
        cmds.append(f"GLITCHSTEP {glitch_step}")
        cmds.append(f"IMUACC {acc}")
        cmds.append(f"IMUGYR {gyr}")
        cmds.append(f"IMU {imu}")
        cmds.append("CFG?")
        cmds.append("IMUCFG?")
        cmds.append("GLITCH?")
        if stream_start:
            cmds.append("STREAM 1")
        return cmds

    # ---------- Filters ----------
    def _get_notch_f0(self) -> float:
        if self.cb_notch_f.currentText() == "Custom":
            return float(self.sp_notch.value())
        return float(self.cb_notch_f.currentText())

    def _rebuild_filters(self) -> None:
        fs = float(self.cur_fs if self.cur_fs > 0 else 1000)
        try:
            self.hpf = design_hpf(fs, float(self.sp_hpf.value())) if self.ck_hpf.isChecked() else None
            self.lpf = design_lpf(fs, float(self.sp_lpf.value())) if self.ck_lpf.isChecked() else None
            if self.ck_notch.isChecked():
                f0 = float(self._get_notch_f0())
                q = float(self.sp_notch_q.value())
                self.notch = design_notch(fs, f0, q)
            else:
                self.notch = None

            # 双通道：CH2 需要独立滤波器状态（避免与 CH1 共用 z1/z2）
            if self.dual_mode:
                self.hpf2 = design_hpf(fs, float(self.sp_hpf.value())) if self.ck_hpf.isChecked() else None
                self.lpf2 = design_lpf(fs, float(self.sp_lpf.value())) if self.ck_lpf.isChecked() else None
                if self.ck_notch.isChecked():
                    self.notch2 = design_notch(fs, f0, q)
                else:
                    self.notch2 = None
            else:
                self.hpf2 = None
                self.lpf2 = None
                self.notch2 = None

        except Exception as e:
            self._log(f"[ERR] filter build: {e}")
            self.hpf = None;
            self.lpf = None;
            self.notch = None
            self.hpf2 = None;
            self.lpf2 = None;
            self.notch2 = None

        # reset filter states
        for f in (self.hpf, self.lpf, self.notch, self.hpf2, self.lpf2, self.notch2):
            if f is not None:
                f.reset()

        # Envelope window depends on fs -> update N
        self._update_env_n(reset=False)

        # 窗口缓存长度与采样率相关：fs变化时自动调整缓存（避免无谓的大缓存造成卡顿）
        try:
            maxlen = int(self.win_sec * max(1, self.cur_fs))
            if getattr(self.buf_t, 'maxlen', None) != maxlen:
                t_old = list(self.buf_t)
                raw_old = list(self.buf_raw)
                fil_old = list(self.buf_filt)
                env_old = list(self.buf_env)
                raw2_old = list(self.buf_raw2)
                fil2_old = list(self.buf_filt2)
                env2_old = list(self.buf_env2)

                self.buf_t = deque(t_old[-maxlen:], maxlen=maxlen)
                self.buf_raw = deque(raw_old[-maxlen:], maxlen=maxlen)
                self.buf_filt = deque(fil_old[-maxlen:], maxlen=maxlen)
                self.buf_env = deque(env_old[-maxlen:], maxlen=maxlen)

                # CH2 buffers 同步调整长度
                self.buf_raw2 = deque(raw2_old[-maxlen:], maxlen=maxlen)
                self.buf_filt2 = deque(fil2_old[-maxlen:], maxlen=maxlen)
                self.buf_env2 = deque(env2_old[-maxlen:], maxlen=maxlen)
        except Exception:
            pass

        # 反馈：在日志和界面上显示“当前滤波参数”是否已更新
        try:
            desc = (
                f"HPF={'ON' if self.ck_hpf.isChecked() else 'OFF'} {self.sp_hpf.value():.1f}Hz; "
                f"LPF={'ON' if self.ck_lpf.isChecked() else 'OFF'} {self.sp_lpf.value():.1f}Hz; "
                f"Notch={'ON' if self.ck_notch.isChecked() else 'OFF'} f0={self._get_notch_f0():.1f}Hz Q={self.sp_notch_q.value():.1f}; "
                f"RMS={int(self.sp_env_ms.value())}ms"
            )
            if hasattr(self, 'lb_hostflt_status'):
                self.lb_hostflt_status.setText("Current filter: " + desc)
            if desc != self._last_hostflt_desc:
                self._last_hostflt_desc = desc
                self._log("Host filter updated: " + desc)
        except Exception:
            pass


    def _on_ch_en_ui_changed(self) -> None:
        """界面侧的通道ONOFF回调（真正下发到下位机需要点击“应用/Start”）。"""
        if not self.dual_mode:
            return
        try:
            if self.ck_ch1_en is not None:
                self.ch1_enabled = bool(self.ck_ch1_en.isChecked())
            if self.ck_ch2_en is not None:
                self.ch2_enabled = bool(self.ck_ch2_en.isChecked())
        except Exception:
            return
        # 仅做显示/缓存清理，避免关闭CH2后仍显示旧曲线
        self._apply_ch_enable_state_to_view(rebuild_filters=True)

    def _apply_ch_enable_state_to_view(self, rebuild_filters: bool = False) -> None:
        """根据 ch1_enabled/ch2_enabled 更新界面显示/缓存（不影响网络接收）。"""
        if not self.dual_mode:
            return

        if rebuild_filters:
            try:
                self._rebuild_filters()
            except Exception:
                pass

        # CH2 关闭：清空第二通道缓存/曲线/包络状态，避免残留
        if not getattr(self, "ch2_enabled", True):
            try:
                self.buf_raw2.clear()
                self.buf_filt2.clear()
                self.buf_env2.clear()
            except Exception:
                pass
            try:
                self._env_sq2.clear()
            except Exception:
                try:
                    self._env_sq2 = deque()
                except Exception:
                    pass
            try:
                self._env_sum2 = 0.0
            except Exception:
                pass
            try:
                self.last_good_uv_ch2 = 0.0
            except Exception:
                pass
            try:
                if self.curve_raw2 is not None:
                    self.curve_raw2.setData([], [])
                if self.curve_filt2 is not None:
                    self.curve_filt2.setData([], [])
                if self.curve_env2 is not None:
                    self.curve_env2.setData([], [])
            except Exception:
                pass

    # ---------- Main update ----------
    def _update_ui(self) -> None:
        # connection watchdog: if no data for long time, mark disconnected
        if self.connected and self.rx.connected:
            pass
        elif self.connected and not self.rx.connected:
            # rx thread ended
            self.connected = False
            self.btn_conn.setEnabled(True)
            self.btn_disc.setEnabled(False)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("Not connected")
            return

        # Drain IMU queue first (so EMG rows can attach latest IMU + quaternion)
        while self.rx.q_imu:
            imu: ImuSample = self.rx.q_imu.popleft()
            try:
                self._process_imu(imu)
            except Exception as e:
                # 防止定时器槽函数反复抛异常导致卡顿/刷屏
                if not hasattr(self, '_imu_proc_err_logged'):
                    self._imu_proc_err_logged = 0
                if self._imu_proc_err_logged < 3:
                    self._log(f"IMU processing error: {e}")
                    import traceback as _tb
                    _tb.print_exc()
                self._imu_proc_err_logged += 1

                # Drain EMG queue
        # 双通道说明：
        #   - 1-11：单通道（只处理/显示 CH1）
        #   - 12-17：双通道（同时处理/显示 CH1 + CH2）
        updated = False
        while self.rx.q_emg:
            start_idx, gain, flags, fs, ch1, ch2 = self.rx.q_emg.popleft()

            # init time base for plot (sample index -> seconds)
            if self._base_start_idx is None:
                self._base_start_idx = int(start_idx)

            # Convert counts -> uV
            uV_per_count = ADS1292R_uV_per_count(gain)
            try:
                ch1_uv = ch1.astype(np.float64) * uV_per_count
                ch2_uv = ch2.astype(np.float64) * uV_per_count
            except Exception:
                ch1_uv = np.asarray(ch1, dtype=np.float64) * uV_per_count
                ch2_uv = np.asarray(ch2, dtype=np.float64) * uV_per_count

            use_ch2 = bool(self.dual_mode) and bool(getattr(self, "ch2_enabled", True))

            # spike guard (protect plot/CSV from occasional bad samples)
            if self.ck_spike.isChecked():
                abs_th = float(self.sp_glitch_abs.value())
                step_th = float(self.sp_glitch_step.value())
            else:
                abs_th = 0.0
                step_th = 0.0
            ch1_uv, ch2_uv, valid_mask = self._spike_guard(ch1_uv, ch2_uv, abs_th, step_th, use_ch2=use_ch2)

            # filter CH1
            y1 = ch1_uv.copy()
            for f in (self.hpf, self.notch, self.lpf):
                if f is not None:
                    y1 = f.process(y1)

            # filter CH2 (if enabled)
            y2 = None
            if use_ch2:
                y2 = ch2_uv.copy()
                for f in (self.hpf2, self.notch2, self.lpf2):
                    if f is not None:
                        y2 = f.process(y2)

            # envelope (RMS)
            n = int(y1.size)
            env1_block = np.empty(n, dtype=np.float64)
            env2_block = np.empty(n, dtype=np.float64) if use_ch2 else None

            si = int(start_idx)
            fs_f = float(fs) if fs and float(fs) > 0 else 1.0
            inv_fs = 1.0 / fs_f

            # local refs for speed
            env_sq1 = self._env_sq
            env_sum1 = float(self._env_sum)
            env_n = int(self._env_n)

            if use_ch2:
                env_sq2 = self._env_sq2
                env_sum2 = float(self._env_sum2)

            for i in range(n):
                t_i = (si + i - int(self._base_start_idx)) * inv_fs
                self.buf_t.append(t_i)

                raw1_i = float(ch1_uv[i])
                fil1_i = float(y1[i])

                # --- CH1 envelope update ---
                sq = fil1_i * fil1_i
                env_sq1.append(sq)
                env_sum1 += sq
                if len(env_sq1) > env_n:
                    env_sum1 -= env_sq1.popleft()

                # numeric safety
                if env_sum1 < 0.0:
                    if env_sum1 > -1e-6:
                        env_sum1 = 0.0
                    else:
                        self._log("[WARN] env_sum negative too large; reset env state")
                        try:
                            env_sq1.clear()
                        except Exception:
                            self._env_sq = deque()
                            env_sq1 = self._env_sq
                        env_sum1 = 0.0

                denom = max(1, len(env_sq1))
                mean_sq = env_sum1 / float(denom)
                if (not math.isfinite(mean_sq)) or (mean_sq < 0.0):
                    mean_sq = 0.0
                env1_i = math.sqrt(mean_sq)
                env1_block[i] = env1_i

                self.buf_raw.append(raw1_i)
                self.buf_filt.append(fil1_i)
                self.buf_env.append(env1_i)

                # --- CH2 (dual only) ---
                if use_ch2 and (y2 is not None) and (env2_block is not None):
                    raw2_i = float(ch2_uv[i])
                    fil2_i = float(y2[i])

                    sq2 = fil2_i * fil2_i
                    env_sq2.append(sq2)
                    env_sum2 += sq2
                    if len(env_sq2) > env_n:
                        env_sum2 -= env_sq2.popleft()

                    if env_sum2 < 0.0:
                        if env_sum2 > -1e-6:
                            env_sum2 = 0.0
                        else:
                            self._log("[WARN] env_sum2 negative too large; reset env2 state")
                            try:
                                env_sq2.clear()
                            except Exception:
                                self._env_sq2 = deque()
                                env_sq2 = self._env_sq2
                            env_sum2 = 0.0

                    denom2 = max(1, len(env_sq2))
                    mean_sq2 = env_sum2 / float(denom2)
                    if (not math.isfinite(mean_sq2)) or (mean_sq2 < 0.0):
                        mean_sq2 = 0.0
                    env2_i = math.sqrt(mean_sq2)
                    env2_block[i] = env2_i

                    self.buf_raw2.append(raw2_i)
                    self.buf_filt2.append(fil2_i)
                    self.buf_env2.append(env2_i)

            # write back sums
            self._env_sum = env_sum1
            if use_ch2:
                self._env_sum2 = env_sum2

            # write CSV if enabled
            if self.csv.enabled:
                try:
                    self.csv.write_emg_block(
                        start_idx=si,
                        fs=int(fs),
                        gain=int(gain),
                        flags=int(flags),
                        ch1_raw_uv=ch1_uv,
                        ch1_filt_uv=y1,
                        ch1_env_uv=env1_block,
                        valid_mask=valid_mask,
                        blank_invalid=self.ck_blank_csv.isChecked(),
                        ch2_raw_uv=(ch2_uv if use_ch2 else None),
                        ch2_filt_uv=(y2 if use_ch2 else None),
                        ch2_env_uv=(env2_block if use_ch2 else None),
                    )
                except Exception as e:
                    self._log(f"[ERR] CSV write_emg_block: {e}")

            self._last_emg_fs = fs
            self._last_emg_gain = gain
            self._last_emg_flags = flags
            updated = True

        if updated and self.draw_enabled:
            t = np.array(self.buf_t, dtype=np.float64)
            raw = np.array(self.buf_raw, dtype=np.float64)
            fil = np.array(self.buf_filt, dtype=np.float64)
            env = np.array(self.buf_env, dtype=np.float64)

            self.curve_raw.setData(t, raw)
            self.curve_filt.setData(t, fil)

            # CH2（双通道设备且启用时显示）
            if use_ch2 and (self.curve_raw2 is not None) and (self.curve_filt2 is not None):
                raw2 = np.array(self.buf_raw2, dtype=np.float64)
                fil2 = np.array(self.buf_filt2, dtype=np.float64)
                self.curve_raw2.setData(t, raw2)
                self.curve_filt2.setData(t, fil2)
            elif self.dual_mode and (self.curve_raw2 is not None) and (self.curve_filt2 is not None):
                # CH2 未启用：清空曲线，避免残留
                self.curve_raw2.setData([], [])
                self.curve_filt2.setData([], [])

            if self.ck_env_show.isChecked():
                self.curve_env.setData(t, env)
                if use_ch2 and (self.curve_env2 is not None):
                    env2 = np.array(self.buf_env2, dtype=np.float64)
                    self.curve_env2.setData(t, env2)
                elif self.dual_mode and (self.curve_env2 is not None):
                    # CH2 未启用：清空包络
                    self.curve_env2.setData([], [])
                self.plot_env.setVisible(True)
            else:
                # hide / clear envelope
                self.curve_env.setData([], [])
                if self.dual_mode and (self.curve_env2 is not None):
                    self.curve_env2.setData([], [])
                self.plot_env.setVisible(False)
            # 自动滚动：显示最近 win_sec 秒（避免需要手动拖动追赶曲线）
            try:
                if self.ck_autoscroll.isChecked() and t.size > 0:
                    t_end = float(t[-1])
                    t0 = max(0.0, t_end - float(self.win_sec))
                    self.plot_emg.setXRange(t0, t_end, padding=0.0)
            except Exception:
                pass

            # 自动缩放 Y（EMG）
            try:
                if self.ck_autoy.isChecked() and fil.size > 10:
                    # 用高分位数估计幅度（比median更能跟随“发力”大幅度变化），再做平滑避免抖动
                    fil_abs = np.abs(fil)
                    if use_ch2 and len(self.buf_filt2) > 10:
                        try:
                            fil2 = np.array(self.buf_filt2, dtype=np.float64)
                            fil_abs = np.concatenate([fil_abs, np.abs(fil2)])
                        except Exception:
                            pass
                    try:
                        p = float(np.nanpercentile(fil_abs, 99.5))
                    except Exception:
                        p = float(np.nanmedian(fil_abs))
                    p = max(p, 50.0)
                    if self._auto_y_emg is None:
                        self._auto_y_emg = p
                    else:
                        self._auto_y_emg = 0.85 * self._auto_y_emg + 0.15 * p
                    y = 1.4 * self._auto_y_emg
                    self.plot_emg.setYRange(-y, y, padding=0)
            except Exception:
                pass

            # 自动缩放 Y（包络）
            try:
                if self.ck_autoy.isChecked() and env.size > 10 and self.ck_env_show.isChecked():
                    env_all = env
                    if use_ch2 and len(self.buf_env2) > 10:
                        try:
                            env2 = np.array(self.buf_env2, dtype=np.float64)
                            env_all = np.concatenate([env_all, env2])
                        except Exception:
                            pass
                    try:
                        p2 = float(np.nanpercentile(env_all, 99.0))
                    except Exception:
                        p2 = float(np.nanmedian(env_all))
                    p2 = max(p2, 10.0)
                    if self._auto_y_env is None:
                        self._auto_y_env = p2
                    else:
                        self._auto_y_env = 0.85 * self._auto_y_env + 0.15 * p2
                    y2 = 1.4 * self._auto_y_env
                    self.plot_env.setYRange(0, y2, padding=0)
            except Exception:
                pass

    def _spike_guard(
        self,
        ch1_uv: np.ndarray,
        ch2_uv: np.ndarray,
        abs_th: float,
        step_th: float,
        use_ch2: bool = True,
    ):
        """简单的毛刺保护：
        - abs_th: 绝对幅值阈值（uV）
        - step_th: 相邻 block 的跳变阈值（uV）
        返回：(ch1_sanitized, ch2_sanitized, valid_mask)
        """
        if ch1_uv is None or len(ch1_uv) == 0:
            return ch1_uv, ch2_uv, np.zeros(0, dtype=np.bool_)

        n = int(ch1_uv.shape[0])
        ch1 = ch1_uv.astype(np.float64, copy=True)
        ch2 = ch2_uv.astype(np.float64, copy=True)
        valid = np.ones(n, dtype=np.bool_)

        last1 = float(self.last_good_uv_ch1)
        last2 = float(self.last_good_uv_ch2)

        for i in range(n):
            v1 = float(ch1[i])
            v2 = float(ch2[i])
            bad = False

            if abs_th > 0 and abs(v1) > abs_th:
                bad = True
            if use_ch2 and abs_th > 0 and abs(v2) > abs_th:
                bad = True

            if (not bad) and step_th > 0 and abs(v1 - last1) > step_th:
                bad = True
            if use_ch2 and (not bad) and step_th > 0 and abs(v2 - last2) > step_th:
                bad = True

            if bad:
                valid[i] = False
                # replace with last good (prevents huge spikes)
                ch1[i] = last1
                if use_ch2:
                    ch2[i] = last2
            else:
                last1 = v1
                if use_ch2:
                    last2 = v2

        self.last_good_uv_ch1 = last1
        if use_ch2:
            self.last_good_uv_ch2 = last2

        return ch1, ch2, valid


    def _process_imu(self, imu: ImuSample) -> None:
        # 轴映射（解决不同安装方向/坐标系问题）
        a = np.array([imu.ax, imu.ay, imu.az], dtype=np.float64)
        g = np.array([imu.gx, imu.gy, imu.gz], dtype=np.float64)
        m = np.array([imu.mx, imu.my, imu.mz], dtype=np.float64)

        a_m = self.axis_ag.apply(a)
        g_m = self.axis_ag.apply(g)
        m_m = self.axis_mag.apply(m)

        # 磁场校准（输出/融合使用校准后的磁场）
        if self.mag_cal.valid:
            m_cal = self.mag_cal.apply(m_m)
        else:
            m_cal = m_m

        # 采集磁校准数据（使用“映射后但未校准”的磁场，用于求解offset/scale）
        if self.mag_cal_running and (imu.flags & 0x02):
            self.mag_cal_samples.append(m_m.copy())
            if len(self.mag_cal_samples) % 200 == 0:
                self.lb_mag_info.setText(f"Collecting… {len(self.mag_cal_samples)} samples")

        # 9轴融合（Madgwick）—— 用“校准后磁场”参与融合
        if getattr(self, "ck_imu", None) is None or self.ck_imu.isChecked():
            if self._imu_last_ms is None:
                dt = 0.01
            else:
                dt = (imu.ms - self._imu_last_ms) / 1000.0
                # 保护：避免异常dt导致发散（例如掉包/暂停后恢复）
                if dt <= 0.0 or dt > 0.2:
                    dt = 0.01
            self._imu_last_ms = imu.ms

            # 用于估计IMU接收率（仅显示用途）
            if self._imu_rate_last_ms is None:
                self._imu_rate_last_ms = imu.ms
            else:
                dms = int(imu.ms - self._imu_rate_last_ms)
                self._imu_rate_last_ms = imu.ms
                if 1 <= dms <= 200:
                    self._imu_dt_hist.append(dms)

            self.fusion.beta = float(self.sp_beta.value())
            # Madgwick 需要陀螺 rad/s；固件发送的是 dps（deg/s）
            g_rad = np.deg2rad(g_m)

            # 如果磁力计不可用（flags bit1==0），传入零向量 => 融合器自动退回 IMU-only
            if (imu.flags & 0x02) != 0:
                mx_use, my_use, mz_use = float(m_cal[0]), float(m_cal[1]), float(m_cal[2])
            else:
                mx_use = my_use = mz_use = 0.0

            self.fusion.update(
                gx=float(g_rad[0]), gy=float(g_rad[1]), gz=float(g_rad[2]),
                ax=float(a_m[0]), ay=float(a_m[1]), az=float(a_m[2]),
                mx=mx_use, my=my_use, mz=mz_use,
                dt=float(dt),
            )

        # 最新IMU（用于CSV/状态显示）：磁场为“校准后”
        self.latest_imu = {
            "ms": int(imu.ms),
            "ax_g": float(a_m[0]), "ay_g": float(a_m[1]), "az_g": float(a_m[2]),
            "gx_dps": float(g_m[0]), "gy_dps": float(g_m[1]), "gz_dps": float(g_m[2]),
            "mx_uT": float(m_cal[0]), "my_uT": float(m_cal[1]), "mz_uT": float(m_cal[2]),
            "temp_C": float(imu.tempC),
            "flags": int(imu.flags),
        }

        # 输出四元数：相对参考姿态（按“复位姿态”设置零点）
        q_abs = self.fusion.q.copy()
        q_out = quat_mul(self.q_ref, q_abs)
        self.latest_q = q_out
        # 若正在记录CSV：IMU单独写入（只在imu_ms变化时写一行）
        if self.csv.enabled:
            self.csv.write_imu(self.latest_imu, self.latest_q)

        # UI 节流更新（避免100Hz频繁setText/3D刷新导致卡顿）
        # 多设备版本：非当前页不做UI刷新/3D刷新，降低CPU
        if not self.draw_enabled:
            return
        now = time.time()

        if now - self._imu_last_ui_update >= 0.05:  # 20Hz 文本更新
            self._imu_last_ui_update = now

            imu_hz_txt = "--.-Hz"
            if len(self._imu_dt_hist) >= 5:
                med = float(np.median(np.array(self._imu_dt_hist, dtype=float)))
                if med > 0.0:
                    imu_hz_txt = f"{1000.0 / med:5.1f}Hz"

            self.lb_imu_vals.setText(
                "IMU≈{rate}  ms={ms}\n"
                "a(g)   {ax:+.3f} {ay:+.3f} {az:+.3f}\n"
                "g(dps) {gx:+.2f} {gy:+.2f} {gz:+.2f}\n"
                "m(uT)  {mx:+.1f} {my:+.1f} {mz:+.1f}\n"
                "T={t:.1f}C  flags=0x{fl:02X}\n"
                "q(ref) {qw:+.4f} {qx:+.4f} {qy:+.4f} {qz:+.4f}".format(
                    rate=imu_hz_txt, ms=int(imu.ms),
                    ax=float(a_m[0]), ay=float(a_m[1]), az=float(a_m[2]),
                    gx=float(g_m[0]), gy=float(g_m[1]), gz=float(g_m[2]),
                    mx=float(m_cal[0]), my=float(m_cal[1]), mz=float(m_cal[2]),
                    t=float(imu.tempC), fl=int(imu.flags),
                    qw=float(q_out[0]), qx=float(q_out[1]), qy=float(q_out[2]), qz=float(q_out[3]),
                )
            )

        if HAS_GL and self.cube is not None and (now - self._imu_last_cube_update >= 0.033):  # ~30Hz 3D更新
            self._imu_last_cube_update = now
            ang, ax, ay, az = quat_to_axis_angle(q_out)
            self.cube.resetTransform()
            self.cube.rotate(ang, ax, ay, az)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._on_stop()
            self._on_csv_off()
            self.rx.disconnect()
        except Exception:
            pass
        super().closeEvent(event)




# ======================= Multi-device (11 devices) UI =======================

def _qt_item_is_user_checkable():
    try:
        return QtCore.Qt.ItemIsUserCheckable
    except Exception:
        return QtCore.Qt.ItemFlag.ItemIsUserCheckable

def _qt_checked():
    try:
        return QtCore.Qt.Checked
    except Exception:
        return QtCore.Qt.CheckState.Checked

def _qt_unchecked():
    try:
        return QtCore.Qt.Unchecked
    except Exception:
        return QtCore.Qt.CheckState.Unchecked

def discover_devices(timeout_s: float = DISCOVERY_TIMEOUT_S) -> List[dict]:
    """
    UDP广播Scan devices（STA路由器模式）：
      - 向 255.255.255.255:DISCOVERY_PORT 广播 DISCOVERY_MAGIC
      - 固件回：EMG_HERE_V1 id=.. label=.. mac=.. ip=.. tcp=3333 rssi=.. fw=..
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(0.15)
    try:
        try:
            s.bind(("", 0))
        except Exception:
            pass

        try:
            try:
                s.sendto(DISCOVERY_MAGIC, ("255.255.255.255", int(DISCOVERY_PORT)))
            except Exception:
                s.sendto(DISCOVERY_MAGIC, ("<broadcast>", int(DISCOVERY_PORT)))

            t0 = time.time()
            found = {}
            while (time.time() - t0) < float(timeout_s):
                try:
                    data, addr = s.recvfrom(512)
                except socket.timeout:
                    continue
                except Exception:
                    break

                txt = data.decode("utf-8", errors="ignore").strip()
                if not txt.startswith(DISCOVERY_REPLY_PREFIX):
                    continue

                d = {"ip": addr[0]}
                parts = txt.split()
                for token in parts[1:]:
                    if "=" in token:
                        k, v = token.split("=", 1)
                        d[k] = v

                key = d.get("mac") or d.get("ip")
                found[key] = d

            return list(found.values())
        finally:
            try:
                s.close()
            except Exception:
                pass
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        return []


class DiscoveryService(QtCore.QObject):
    """
    after台循环扫描（不阻塞UI），用于：
      - 上电自动发现设备
      - after续上电的设备自动Connect
    """
    found_sig = QtCore.Signal(object) if QT_LIB == "PySide6" else QtCore.pyqtSignal(object)
    text_sig = QtCore.Signal(str) if QT_LIB == "PySide6" else QtCore.pyqtSignal(str)

    def __init__(self, interval_s: float = 1.0, timeout_s: float = 0.25):
        super().__init__()
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._kick = threading.Event()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._kick.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._kick.set()
        except Exception:
            pass
        th = self._thread
        self._thread = None
        if th and th.is_alive():
            try:
                th.join(timeout=1.0)
            except Exception:
                pass

    def kick(self) -> None:
        """手动触发尽快扫描一次"""
        try:
            self._kick.set()
        except Exception:
            pass

    def _loop(self) -> None:
        while self._running:
            try:
                devs = discover_devices(timeout_s=self.timeout_s)
                self.found_sig.emit(devs)
            except Exception as e:
                self.text_sig.emit(f"[DISCOVERY] {e}")
            # 等待下一次
            try:
                self._kick.wait(timeout=self.interval_s)
                self._kick.clear()
            except Exception:
                time.sleep(self.interval_s)


class OverviewWidget(QtWidgets.QWidget):
    """
    总览页：
      - All device envelopes (color-coded)
      - 在线情况：在线/离线 + 设备号 + 电量
      - CSV录制：勾选要录制的设备，Startafter只记录勾选项（EMG/IMU分ON保存，文件名自动追加设备号）
    """

    def __init__(self, devices: List[DeviceWidget], parent=None):
        super().__init__(parent)
        self.devices = devices
        self._recording = False
        self._build_ui()

        # timer: 更新总览曲线/表格
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_overview)
        self.timer.start(100)  # 10Hz足够了（避免总览页刷新过快）

    def _build_ui(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # ---- Top controls ----
        top = QtWidgets.QHBoxLayout()
        self.ck_auto = QtWidgets.QCheckBox("Auto-scan and auto-connect (map device ID=1..17 to windows; 1–11 single-channel, 12–17 dual-channel)")
        self.ck_auto.setChecked(True)
        self.btn_scan_now = QtWidgets.QPushButton("Scan now")
        self.btn_all_start = QtWidgets.QPushButton("Start all acquisitions")
        self.btn_all_stop = QtWidgets.QPushButton("Stop all acquisitions")
        top.addWidget(self.ck_auto, 1)
        top.addWidget(self.btn_scan_now, 0)
        top.addWidget(self.btn_all_start, 0)
        top.addWidget(self.btn_all_stop, 0)
        v.addLayout(top)

        hint = QtWidgets.QLabel(
            "Note: This page is for multi-device monitoring/battery/recording.\n"
            "Envelope x-axis is relative time (latest sample per device is 0 s; shows the most recent window). Devices can start at different times and still be viewed together.\n"
            "All features on the device pages (Acquisition/Filtering/Spikes/IMU/Log, etc.) are preserved; but 【Battery】 and 【CSV Logging】 are hidden on device pages and managed here."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        # ---- Envelope plot (all devices) ----
        g_plot = QtWidgets.QGroupBox("All device envelopes (color-coded)")
        l_plot = QtWidgets.QVBoxLayout(g_plot)
        self.plot_all = pg.PlotWidget()
        self.plot_all.showGrid(x=True, y=True)
        try:
            self.plot_all.setLabel('bottom', 'Relative time', units='s')
            self.plot_all.setLabel('left', 'Envelope', units='uV_RMS')
        except Exception:
            pass
        # Legend：设备数量多(17台 + 双通道CH2)，单个Legend可能显示不下；
        # 这里拆成左右两个Legend，避免“后面的设备颜色提示看不到”。
        self.legend_left = None
        self.legend_right = None
        try:
            pi = self.plot_all.getPlotItem()
            self.legend_left = pg.LegendItem(offset=(10, 10))
            self.legend_left.setParentItem(pi)
            self.legend_right = pg.LegendItem(offset=(-10, 10))  # 右上角
            self.legend_right.setParentItem(pi)
        except Exception:
            self.legend_left = None
            self.legend_right = None

        # 颜色：每个设备一组颜色；双通道设备(12-17)会额外画 CH2（不同颜色）
        # 说明：你约定 1-11 为单通道，12-17 为双通道，因此这里用“窗口号/设备号”判断。
        base_colors = [
            (255, 0, 0),      # 01
            (0, 255, 0),      # 02
            (0, 180, 255),    # 03
            (255, 200, 0),    # 04
            (255, 0, 255),    # 05
            (0, 255, 255),    # 06
            (180, 180, 180),  # 07
            (255, 120, 0),    # 08
            (120, 0, 255),    # 09
            (0, 120, 255),    # 10
            (0, 200, 120),    # 11
            (200, 60, 60),    # 12 (dual)
            (60, 200, 60),    # 13 (dual)
            (60, 60, 200),    # 14 (dual)
            (200, 60, 200),   # 15 (dual)
            (200, 140, 60),   # 16 (dual)
            (60, 200, 200),   # 17 (dual)
        ]

        def _alt_color(c):
            # CH2 用另一种颜色：简单交换通道，确保与 CH1 不同
            r, g, b = c
            return (b, r, g)

        self._pens_ch1 = [pg.mkPen(color=c, width=2) for c in base_colors]
        self._pens_ch2 = [pg.mkPen(color=_alt_color(c), width=2) for c in base_colors]

        self.curves_ch1 = []
        self.curves_ch2 = []  # 单通道设备对应 None

        for i in range(len(self.devices)):
            dev_no = i + 1

            # CH1
            name1 = f"Device {dev_no:02d} - CH1"
            c1 = self.plot_all.plot(pen=self._pens_ch1[i % len(self._pens_ch1)])
            try:
                c1.setDownsampling(auto=True, mode='peak')
                c1.setClipToView(True)
            except Exception:
                pass
            self.curves_ch1.append(c1)
            try:
                if self.legend_left is not None and self.legend_right is not None:
                    (self.legend_left if dev_no <= 9 else self.legend_right).addItem(c1, name1)
            except Exception:
                pass

            # CH2 (仅 12-17 双通道设备)
            if dev_no >= 12:
                name2 = f"Device {dev_no:02d} - CH2"
                c2 = self.plot_all.plot(pen=self._pens_ch2[i % len(self._pens_ch2)])
                try:
                    c2.setDownsampling(auto=True, mode='peak')
                    c2.setClipToView(True)
                except Exception:
                    pass
                self.curves_ch2.append(c2)
                try:
                    if self.legend_left is not None and self.legend_right is not None:
                        (self.legend_left if dev_no <= 9 else self.legend_right).addItem(c2, name2)
                except Exception:
                    pass
            else:
                self.curves_ch2.append(None)

        l_plot.addWidget(self.plot_all)
        v.addWidget(g_plot, 2)

        # ---- Status table ----
        g_tbl = QtWidgets.QGroupBox("Device online/battery status and recording selection")
        l_tbl = QtWidgets.QVBoxLayout(g_tbl)

        self.tbl = QtWidgets.QTableWidget(len(self.devices), 8)
        self.tbl.setHorizontalHeaderLabels([
            "Record", "Device ID", "LABEL", "Status", "IP", "Port", "Battery (%)", "Voltage(V)"
        ])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.horizontalHeader().setStretchLastSection(True)

        # 初始化“录制”复选框列
        for r in range(len(self.devices)):
            it = QtWidgets.QTableWidgetItem("")
            it.setFlags(it.flags() | _qt_item_is_user_checkable())
            it.setCheckState(_qt_unchecked())
            self.tbl.setItem(r, 0, it)
            # 其他列先填空
            for c in range(1, 8):
                self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem("—"))

        l_tbl.addWidget(self.tbl)
        v.addWidget(g_tbl, 1)

        # ---- CSV record ----
        g_rec = QtWidgets.QGroupBox('CSV logging (only devices checked in the "Record" column)')
        l_rec = QtWidgets.QGridLayout(g_rec)

        self.ed_base = QtWidgets.QLineEdit("emg_log.csv")
        self.ck_vt_enable = QtWidgets.QCheckBox("Enable Visual-Tactile (Camera)") # 新增：视触觉开关
        self.sp_cam_idx = QtWidgets.QSpinBox() # 新增：选择相机编号（默认0）
        self.sp_cam_idx.setPrefix("Cam Index: ")
        
        self.btn_pick = QtWidgets.QPushButton("Choose filename…")
        self.btn_rec_on = QtWidgets.QPushButton("Start recording")
        self.btn_rec_off = QtWidgets.QPushButton("Stop recording")
        self.btn_rec_off.setEnabled(False)

        self.lb_rec_hint = QtWidgets.QLabel(
            'Naming rule: "_devXX" will be appended to your filename.\n'
            "Example: enter session.csv to generate: session_dev01.csv / session_dev01_imu.csv …\n"
            "EMG and IMU are saved separately. Devices not checked will not create files or write any data."
        )
        self.lb_rec_hint.setWordWrap(True)

        #l_rec.addWidget(self.ed_base, 0, 0, 1, 3)
        #l_rec.addWidget(self.btn_pick, 0, 3)
        #l_rec.addWidget(self.btn_rec_on, 1, 0)
        #l_rec.addWidget(self.btn_rec_off, 1, 1)
        #l_rec.addWidget(self.lb_rec_hint, 2, 0, 1, 4)

        l_rec.addWidget(self.ed_base, 0, 0, 1, 2)
        l_rec.addWidget(self.ck_vt_enable, 0, 2)
        l_rec.addWidget(self.sp_cam_idx, 0, 3)
        l_rec.addWidget(self.btn_rec_on, 1, 0, 1, 2)
        l_rec.addWidget(self.btn_rec_off, 1, 2, 1, 2)

        v.addWidget(g_rec, 0)

        # ---- Signals ----
        self.btn_pick.clicked.connect(self._pick_file)
        self.btn_rec_on.clicked.connect(self._record_start)
        self.btn_rec_off.clicked.connect(self._record_stop)

        # 这两个按钮需要 main window 提供回调（稍后由 MultiMainWindow 绑定）
        self.btn_scan_now.clicked.connect(lambda: None)
        self.btn_all_start.clicked.connect(lambda: None)
        self.btn_all_stop.clicked.connect(lambda: None)

    def bind_actions(self, scan_cb, all_start_cb, all_stop_cb) -> None:
        """由 MultiMainWindow 绑定：扫描/全部Start/全部Stop"""
        try:
            self.btn_scan_now.clicked.disconnect()
        except Exception:
            pass
        try:
            self.btn_all_start.clicked.disconnect()
        except Exception:
            pass
        try:
            self.btn_all_stop.clicked.disconnect()
        except Exception:
            pass

        self.btn_scan_now.clicked.connect(scan_cb)
        self.btn_all_start.clicked.connect(all_start_cb)
        self.btn_all_stop.clicked.connect(all_stop_cb)

    def _pick_file(self) -> None:
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose CSV filename (base name)", self.ed_base.text(),
                                                           "CSV Files (*.csv);;All Files (*)")
            if path:
                self.ed_base.setText(path)
        except Exception:
            pass

    def _selected_device_indices(self) -> List[int]:
        sel = []
        for r in range(len(self.devices)):
            it = self.tbl.item(r, 0)
            if it and it.checkState() == _qt_checked():
                sel.append(r)
        return sel

    def _make_paths(self, base_path: str, dev_id: int) -> Tuple[str, str]:
        base_path = base_path.strip()
        if not base_path:
            base_path = "emg_log.csv"
        if base_path.lower().endswith(".csv"):
            root = base_path[:-4]
        else:
            root = base_path
        emg_path = f"{root}_dev{dev_id:02d}.csv"
        imu_path = f"{root}_dev{dev_id:02d}_imu.csv"
        return emg_path, imu_path

    def _record_start(self) -> None:
        base = self.ed_base.text().strip()
        idxs = self._selected_device_indices()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Notice", "Please tick the devices you want to record in the table (check the 'Record' column) first.")
            return

        # 先停掉所有（防止重复打开）
        for d in self.devices:
            try:
                d.csv.stop()
            except Exception:
                pass

        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for i in idxs:
            d = self.devices[i]
            # 优先使用 UDP discovery 的设备ID；否则用窗口号
            try:
                did = int(d.discovered_info.get("id", d.slot_index))
            except Exception:
                did = int(d.slot_index)

            emg_path, imu_path = self._make_paths(base, did)
            meta = [
                f"FW/GUI: v9.6.1-multi17-dual ({QT_LIB})",
                f"StartTime: {now_str}",
                f"Slot: {d.slot_index}",
                f"DeviceID: {did}",
                f"Label: {d.discovered_info.get('label', '')}",
                f"IP: {d.discovered_info.get('ip', d.ed_ip.text().strip())}",
                f"TCP: {d.discovered_info.get('tcp', d.ed_port.text().strip())}",
                f"DeviceFW: {getattr(d, 'dev_fw', '')}",
            ]
            try:
                d.csv.start(emg_path, imu_path, meta_lines=meta)
                d._log(f"[CSV] Start recording: {emg_path} / {imu_path}")
            except Exception as e:
                d._log(f"[CSV] Start failed: {e}")

        # 2. 启动视触觉视频录制 (新增)
        if self.ck_vt_enable.isChecked():
            # 获取基础文件名（例如 session01）
            root = base[:-4] if base.lower().endswith(".csv") else base
            video_filename = f"{root}_tactile.mp4"
        
            cam_idx = self.sp_cam_idx.value()
            self.video_thread = VideoRecorder(camera_idx=cam_idx, save_path=video_filename)
            self.video_thread.start()
        
            # 记录开始录制的系统时间，存入设备日志以便查验
            if self.devices:
                start_msg = f"[VIDEO] Recording started: {video_filename} at {time.time()}"
                self.devices[0]._log(start_msg)

        self._recording = True
        self.btn_rec_on.setEnabled(False)
        self.btn_rec_off.setEnabled(True)

    def _record_stop(self) -> None:
        # 停止所有 EMG 记录
        for d in self.devices:
            try:
                if d.csv.enabled: d.csv.stop()
            except Exception: pass

        # 停止视频录制逻辑
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            if self.video_thread.isRunning():
                self.video_thread.stop() # 调用改进后的带超时的 stop
            self.video_thread = None # 显式释放对象，防止重复调用

        self._recording = False
        self.btn_rec_on.setEnabled(True)
        self.btn_rec_off.setEnabled(False)

        
    def _update_overview(self) -> None:
        # 1) plot envelopes
        # 多设备场景下，各设备“开始采集”的时间不一定一致：
        # 如果直接用设备内部 start_idx 推导出的绝对时间作为横轴，
        # 会出现某些设备曲线落在当前显示窗口范围外（看起来像“没显示”，你反馈的“要点全部停止再开始才出现”）。
        # 这里改为：以每台设备“最新点”为 0s 的相对时间轴（最近 win 秒：[-win, 0]），
        # 这样无论哪台设备何时开始采集，都能立刻在总览看到曲线。
        try:
            win = float(self.devices[0].win_sec) if (self.devices and hasattr(self.devices[0], "win_sec")) else 5.0
        except Exception:
            win = 5.0
        if win <= 0:
            win = 5.0

        for i, d in enumerate(self.devices):
            # 统一的相对时间轴：最后一个点为 0s
            try:
                t = np.array(getattr(d, "buf_t", []), dtype=np.float64)
            except Exception:
                t = np.array([], dtype=np.float64)

            if t.size > 0:
                t_rel_all = t - float(t[-1])
            else:
                t_rel_all = t

            # ---- CH1 envelope ----
            try:
                env1 = np.array(getattr(d, "buf_env", []), dtype=np.float64)
                if t_rel_all.size > 0 and env1.size == t_rel_all.size:
                    if win > 0:
                        m = t_rel_all >= (-win)
                        self.curves_ch1[i].setData(t_rel_all[m], env1[m])
                    else:
                        self.curves_ch1[i].setData(t_rel_all, env1)
                else:
                    self.curves_ch1[i].setData([], [])
            except Exception:
                try:
                    self.curves_ch1[i].setData([], [])
                except Exception:
                    pass

            # ---- CH2 envelope (only for dual devices 12-17) ----
            c2 = None
            try:
                c2 = self.curves_ch2[i]
            except Exception:
                c2 = None

            if c2 is not None:
                try:
                    # CH2 关闭时不显示第二通道曲线
                    if not bool(getattr(d, "ch2_enabled", True)):
                        raise RuntimeError("CH2 disabled")
                    env2 = np.array(getattr(d, "buf_env2", []), dtype=np.float64)
                    if t_rel_all.size > 0 and env2.size == t_rel_all.size:
                        if win > 0:
                            m = t_rel_all >= (-win)
                            c2.setData(t_rel_all[m], env2[m])
                        else:
                            c2.setData(t_rel_all, env2)
                    else:
                        c2.setData([], [])
                except Exception:
                    try:
                        c2.setData([], [])
                    except Exception:
                        pass


        try:
            self.plot_all.setXRange(-win, 0.0, padding=0.0)
        except Exception:
            pass

        # 2) table status
        now = time.time()
        for r, d in enumerate(self.devices):
            # device id/label from discovery
            try:
                did = int(d.discovered_info.get("id", d.slot_index))
            except Exception:
                did = int(d.slot_index)
            label = str(d.discovered_info.get("label", f"DEV{did:02d}"))

            ip = str(d.discovered_info.get("ip", d.ed_ip.text().strip()))
            tcp = str(d.discovered_info.get("tcp", d.ed_port.text().strip()))

            # online logic
            seen_age = (now - float(d.last_discovery_time)) if d.last_discovery_time > 0 else 9999.0
            rx_age = (now - float(d.rx.last_rx_time)) if getattr(d.rx, "last_rx_time", 0.0) > 0 else 9999.0

            online = (d.connected and getattr(d.rx, "connected", False)) or (seen_age < 2.5)
            if not online:
                status = "Offline"
            else:
                if d.connected and getattr(d.rx, "connected", False):
                    status = "Acquiring" if getattr(d, "streaming", False) else "Connected"
                else:
                    status = "Online (not connected)"

            # battery from EMA values
            pct = "—"
            vb = "—"
            try:
                if getattr(d, "_bat_ema_pct", None) is not None:
                    pct = f"{float(d._bat_ema_pct):.0f}"
                if getattr(d, "_bat_ema_v", None) is not None:
                    vb = f"{float(d._bat_ema_v):.3f}"
            except Exception:
                pass

            # update items
            self.tbl.item(r, 1).setText(str(did))
            self.tbl.item(r, 2).setText(label)
            self.tbl.item(r, 3).setText(status)
            self.tbl.item(r, 4).setText(ip)
            self.tbl.item(r, 5).setText(tcp)
            self.tbl.item(r, 6).setText(pct)
            self.tbl.item(r, 7).setText(vb)


class MultiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"EMG Multi-device Acquisition (17 devices, dual-channel 12–17) v9.6.1-multi17-dual ({QT_LIB})")
        self.resize(1500, 950)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # 11 device pages
        self.devices: List[DeviceWidget] = []
        for i in range(17):
            w = DeviceWidget(slot_index=i + 1)
            self.devices.append(w)
            self.tabs.addTab(w, f"Device {i + 1:02d}")

        # overview page
        self.overview = OverviewWidget(self.devices, parent=self)
        self.tabs.addTab(self.overview, "Overview / Battery / Record")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(0)

        # auto connect flag
        self.auto_connect_enabled = True
        # 记录“总览页-全部开始采集”的意图：用于后续新上线设备自动开始采集
        self.want_streaming = False
        self.overview.ck_auto.toggled.connect(self._on_auto_toggled)

        # bind overview actions
        self.overview.bind_actions(self._manual_scan, self._all_start, self._all_stop)

        # discovery background service
        self.discovery = DiscoveryService(interval_s=1.0, timeout_s=0.25)
        self.discovery.found_sig.connect(self._on_discovery)
        self.discovery.text_sig.connect(self._log_discovery)
        self.discovery.start()

    def _log_discovery(self, s: str) -> None:
        # 写到总览页第1个设备日志里也行；这里不刷屏
        try:
            if self.devices:
                self.devices[0]._log(s)
        except Exception:
            pass

    def _on_auto_toggled(self, on: bool) -> None:
        self.auto_connect_enabled = bool(on)

    def _manual_scan(self) -> None:
        try:
            self.discovery.kick()
        except Exception:
            pass

    def _all_start(self) -> None:
        # 记录意图：之后新上线/新连接的设备也会自动开始
        # （解决“刚连接时总览曲线不全，需要再点一次停止/开始”的体验问题）
        self.want_streaming = True
        for d in self.devices:
            try:
                if d.connected and getattr(d.rx, "connected", False) and (not getattr(d, "streaming", False)):
                    d._on_start()
            except Exception:
                pass

    def _all_stop(self) -> None:
        self.want_streaming = False
        for d in self.devices:
            try:
                d._on_stop()
            except Exception:
                pass

    def _on_tab_changed(self, idx: int) -> None:
        # 仅当前设备页绘图（大幅降低17台同时在线时CPU）
        for i, d in enumerate(self.devices):
            d.draw_enabled = (idx == i)

    def _apply_tab_titles(self) -> None:
        # 如果 discovery 有 label，就把 tab 名更新成：设备XX(LABEL)
        for i, d in enumerate(self.devices):
            label = d.discovered_info.get("label", "")
            did = d.discovered_info.get("id", "")
            if label or did:
                try:
                    did_i = int(did) if str(did).isdigit() else (i + 1)
                except Exception:
                    did_i = i + 1
                name = f"{did_i:02d}"
                if label:
                    name += f"({label})"
                self.tabs.setTabText(i, "Device" + name)

    def _on_discovery(self, devs_obj) -> None:
        # devs_obj is list[dict]
        if not isinstance(devs_obj, list):
            return
        now = time.time()

        for d in devs_obj:
            try:
                did = int(d.get("id", 0))
            except Exception:
                did = 0
            if 1 <= did <= 17:
                slot = did - 1
            else:
                # 不在1..17范围内：先不自动映射（你也可以手动在某个设备页连接）
                continue

            w = self.devices[slot]
            w.discovered_info = d
            w.last_discovery_time = now

            # 未连接时自动把IP/端口填进去（方便你进入设备页一键点连接）
            try:
                if not w.connected:
                    ip = d.get("ip", "")
                    tcp = d.get("tcp", "")
                    if ip:
                        w.ed_ip.setText(str(ip))
                    if tcp:
                        w.ed_port.setText(str(tcp))
            except Exception:
                pass

            # 自动连接
            if self.auto_connect_enabled:
                try:
                    if (not w.connected) and (not w.connecting):
                        ip = d.get("ip", "")
                        tcp = int(d.get("tcp", 3333))
                        if ip:
                            w.connecting = True
                            # 使用设备页自身的连接逻辑（不会改动你原有通信/采集逻辑）
                            w.ed_ip.setText(str(ip))
                            w.ed_port.setText(str(tcp))
                            w._on_connect()
                            w.connecting = False
                except Exception:
                    w.connecting = False

            # 如果用户已点击“全部开始采集”，则新上线/新连接成功的设备也自动开始采集
            if self.want_streaming:
                try:
                    if w.connected and getattr(w.rx, "connected", False) and (not getattr(w, "streaming", False)):
                        w._on_start()
                except Exception:
                    pass

        self._apply_tab_titles()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.discovery.stop()
        except Exception:
            pass
        # 停止所有设备 & CSV
        for d in self.devices:
            try:
                d._on_stop()
                d.csv.stop()
                d.rx.disconnect()
            except Exception:
                pass
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MultiMainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
