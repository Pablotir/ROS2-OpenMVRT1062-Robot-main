#!/usr/bin/env python3
"""
roboclaw_driver.py
==================
Minimal RoboClaw packet-serial driver for the Jetson mecanum robot.

Implements only the commands needed for closed-loop velocity control
and encoder feedback.  Based on the Basicmicro packet serial protocol.

Protocol summary
-----------------
Write commands:
    Send:    [addr][cmd][data...][CRC16_hi][CRC16_lo]
    Receive: 0xFF (ACK)

Read commands:
    Send:    [addr][cmd]
    Receive: [data...][CRC16_hi][CRC16_lo]
    CRC covers: addr + cmd + response data
"""

import struct
import threading
import serial


class RoboclawDriver:
    """Thread-safe RoboClaw packet serial driver."""

    # ── Command IDs ───────────────────────────────────────────────────────────
    CMD_GETM1ENC     = 16   # Read M1 encoder count
    CMD_GETM2ENC     = 17   # Read M2 encoder count
    CMD_GETM1SPEED   = 18   # Read M1 encoder speed (QPPS)
    CMD_GETM2SPEED   = 19   # Read M2 encoder speed (QPPS)
    CMD_GETMBATT     = 24   # Read main battery voltage
    CMD_MIXEDSPEED   = 37   # Set M1+M2 speed (signed QPPS)
    CMD_GETCURRENTS  = 49   # Read M1+M2 motor currents
    CMD_GETTEMP      = 82   # Read board temperature

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1,
                 retries: int = 3):
        self._port_name = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._retries = retries
        self._ser = None
        self._lock = threading.Lock()
        self._crc = 0

    # ── Connection ────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open the serial port. Returns True on success."""
        try:
            self._ser = serial.Serial(
                self._port_name, self._baudrate, timeout=self._timeout)
            return True
        except serial.SerialException:
            self._ser = None
            return False

    def close(self):
        """Close the serial port."""
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ── CRC-CCITT (XModem) ────────────────────────────────────────────────────

    def _crc_update(self, byte: int):
        """Update running CRC with one byte."""
        self._crc = self._crc ^ (byte << 8)
        for _ in range(8):
            if self._crc & 0x8000:
                self._crc = (self._crc << 1) ^ 0x1021
            else:
                self._crc = self._crc << 1
        self._crc &= 0xFFFF

    # ── Low-level I/O ─────────────────────────────────────────────────────────

    def _send_command(self, address: int, cmd: int):
        """Send address + command bytes and start CRC tracking."""
        self._crc = 0
        self._crc_update(address)
        self._ser.write(bytes([address]))
        self._crc_update(cmd)
        self._ser.write(bytes([cmd]))

    def _write_byte(self, val: int):
        """Write one byte, updating CRC."""
        self._crc_update(val)
        self._ser.write(bytes([val]))

    def _write_s32(self, val: int):
        """Write signed 32-bit big-endian, updating CRC."""
        data = struct.pack('>i', int(val))
        for b in data:
            self._crc_update(b)
        self._ser.write(data)

    def _write_crc(self):
        """Write the accumulated CRC (2 bytes big-endian)."""
        self._ser.write(bytes([(self._crc >> 8) & 0xFF, self._crc & 0xFF]))

    def _read_byte(self) -> int | None:
        """Read one byte, updating CRC. Returns None on timeout."""
        data = self._ser.read(1)
        if len(data) != 1:
            return None
        self._crc_update(data[0])
        return data[0]

    def _read_u16(self) -> int | None:
        """Read unsigned 16-bit big-endian, updating CRC."""
        data = self._ser.read(2)
        if len(data) != 2:
            return None
        for b in data:
            self._crc_update(b)
        return struct.unpack('>H', data)[0]

    def _read_s32(self) -> int | None:
        """Read signed 32-bit big-endian, updating CRC."""
        data = self._ser.read(4)
        if len(data) != 4:
            return None
        for b in data:
            self._crc_update(b)
        return struct.unpack('>i', data)[0]

    def _read_u32(self) -> int | None:
        """Read unsigned 32-bit big-endian, updating CRC."""
        data = self._ser.read(4)
        if len(data) != 4:
            return None
        for b in data:
            self._crc_update(b)
        return struct.unpack('>I', data)[0]

    def _read_crc(self) -> bool:
        """Read 2-byte CRC and verify against accumulated CRC."""
        data = self._ser.read(2)
        if len(data) != 2:
            return False
        received = (data[0] << 8) | data[1]
        return self._crc == received

    def _read_ack(self) -> bool:
        """Read single-byte ACK (0xFF)."""
        data = self._ser.read(1)
        return len(data) == 1 and data[0] == 0xFF

    # ── Public API ────────────────────────────────────────────────────────────

    def speed_m1_m2(self, address: int, speed1: int, speed2: int) -> bool:
        """
        Set M1 and M2 speeds in QPPS (signed).
        Positive = forward (as configured during autotune).

        Returns True on ACK.
        """
        with self._lock:
            for _ in range(self._retries):
                try:
                    self._ser.reset_input_buffer()
                    self._send_command(address, self.CMD_MIXEDSPEED)
                    self._write_s32(speed1)
                    self._write_s32(speed2)
                    self._write_crc()
                    if self._read_ack():
                        return True
                except (serial.SerialException, OSError):
                    pass
            return False

    def read_encoder_m1(self, address: int) -> tuple[bool, int, int]:
        """
        Read M1 encoder position.

        Returns (success, count, status).
        Count is a signed 32-bit value (cumulative ticks).
        """
        with self._lock:
            for _ in range(self._retries):
                try:
                    self._ser.reset_input_buffer()
                    self._send_command(address, self.CMD_GETM1ENC)
                    count = self._read_u32()
                    status = self._read_byte()
                    if count is not None and status is not None and self._read_crc():
                        # Convert unsigned to signed
                        if count >= 0x80000000:
                            count -= 0x100000000
                        return (True, count, status)
                except (serial.SerialException, OSError):
                    pass
            return (False, 0, 0)

    def read_encoder_m2(self, address: int) -> tuple[bool, int, int]:
        """
        Read M2 encoder position.

        Returns (success, count, status).
        """
        with self._lock:
            for _ in range(self._retries):
                try:
                    self._ser.reset_input_buffer()
                    self._send_command(address, self.CMD_GETM2ENC)
                    count = self._read_u32()
                    status = self._read_byte()
                    if count is not None and status is not None and self._read_crc():
                        if count >= 0x80000000:
                            count -= 0x100000000
                        return (True, count, status)
                except (serial.SerialException, OSError):
                    pass
            return (False, 0, 0)

    def read_main_battery(self, address: int) -> tuple[bool, float]:
        """
        Read main battery voltage.

        Returns (success, voltage_volts). Voltage is in 0.1V units from
        the controller, converted to volts here.
        """
        with self._lock:
            for _ in range(self._retries):
                try:
                    self._ser.reset_input_buffer()
                    self._send_command(address, self.CMD_GETMBATT)
                    raw = self._read_u16()
                    if raw is not None and self._read_crc():
                        return (True, raw / 10.0)
                except (serial.SerialException, OSError):
                    pass
            return (False, 0.0)

    def _read_s16(self) -> int | None:
        """Read signed 16-bit big-endian, updating CRC."""
        data = self._ser.read(2)
        if len(data) != 2:
            return None
        for b in data:
            self._crc_update(b)
        return struct.unpack('>h', data)[0]   # lowercase h = signed

    def read_currents(self, address: int) -> tuple[bool, float, float]:
        """
        Read M1 and M2 motor currents.

        Returns (success, m1_amps, m2_amps).
        Raw values are signed 16-bit in 10mA units (divide by 100 → amps).
        Example: raw 284 → 2.84 A
        """
        with self._lock:
            for _ in range(self._retries):
                try:
                    self._ser.reset_input_buffer()
                    self._send_command(address, self.CMD_GETCURRENTS)
                    m1_raw = self._read_s16()   # signed — was _read_u16 (bug)
                    m2_raw = self._read_s16()
                    if m1_raw is not None and m2_raw is not None and self._read_crc():
                        return (True, abs(m1_raw) / 100.0, abs(m2_raw) / 100.0)
                except (serial.SerialException, OSError):
                    pass
            return (False, 0.0, 0.0)

    def read_temperature(self, address: int) -> tuple[bool, float]:
        """
        Read board temperature.

        Returns (success, temp_celsius). Raw value is in 0.1°C units.
        """
        with self._lock:
            for _ in range(self._retries):
                try:
                    self._ser.reset_input_buffer()
                    self._send_command(address, self.CMD_GETTEMP)
                    raw = self._read_u16()
                    if raw is not None and self._read_crc():
                        return (True, raw / 10.0)
                except (serial.SerialException, OSError):
                    pass
            return (False, 0.0)
