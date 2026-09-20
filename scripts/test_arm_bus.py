#!/usr/bin/env python3
"""
test_arm_bus.py — Fast Diagnostic for SO-ARM101 Feetech Servos
Checks:
  1. Serial port access (/dev/arm_controller)
  2. Ping for each servo ID (1 to 6) across standard baudrates (1,000,000 and 115,200)
  3. Clear error diagnosis (power supply, USB disconnect, cable loose)
"""
import os
import sys
import time

def main():
    print("═" * 60)
    print(" 🔍 SO-ARM101 Feetech Servo Bus Diagnostic")
    print("═" * 60)

    port = "/dev/arm_controller"
    if not os.path.exists(port):
        import glob
        raw = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if raw:
            port = raw[0]
            print(f"⚠️  /dev/arm_controller not found, using raw device: {port}")
        else:
            print("❌ No serial devices (/dev/arm_controller, /dev/ttyUSB*, /dev/ttyACM*) found!")
            print("   ↳ Check if the arm USB cable is plugged into the Jetson.")
            return

    print(f"🔌 Testing port: {port}")
    stat = os.stat(port)
    print(f"   ↳ Permissions: {oct(stat.st_mode)[-3:]} | Owner UID: {stat.st_uid}")

    try:
        import scservo_sdk as scs
    except ImportError:
        print("❌ scservo_sdk (feetech-servo-sdk) not installed in Python environment.")
        return

    baudrates = [1000000, 115200]
    found_any = False

    for baud in baudrates:
        print(f"\n📡 Scanning motor IDs 1–6 @ {baud:,} baud...")
        portHandler = scs.PortHandler(port)
        packetHandler = scs.sms_sts(portHandler)

        if not portHandler.openPort():
            print(f"❌ Failed to open port {port} (already open by another process or locked).")
            return

        if not portHandler.setBaudRate(baud):
            print(f"❌ Failed to set baudrate {baud}.")
            portHandler.closePort()
            continue

        found_motors = {}
        for mid in range(1, 7):
            model, result, err = packetHandler.ping(mid)
            if result == scs.COMM_SUCCESS:
                print(f"   ✅ Motor ID {mid}: Responded! (Model number: {model})")
                found_motors[mid] = model
                found_any = True
            else:
                err_str = packetHandler.getTxRxResult(result)
                # Just brief summary
                pass

        portHandler.closePort()

        if found_motors:
            print(f"🎉 Successfully communicated with {len(found_motors)} motors at {baud:,} baud: {found_motors}")
            break
        else:
            print(f"   ❌ No motors responded at {baud:,} baud.")

    print("\n" + "═" * 60)
    if found_any:
        print("✅ HARDWARE IS HEALTHY: Motors are powered and communicating.")
    else:
        print("❌ DIAGNOSIS: ZERO MOTORS RESPONDED")
        print("═" * 60)
        print("Most common causes:")
        print("  1. 🔴 EXTERNAL 12V / 7.4V POWER IS OFF OR DISCONNECTED:")
        print("     • The USB cable only powers the USB-UART chip, NOT the motors.")
        print("     • Check if the arm 12V power supply is plugged in and turned ON.")
        print("     • If the motors spun into a hard limit earlier, the power supply's")
        print("       over-current protection (OCP) may have tripped:")
        print("       ↳ Unplug the 12V DC power jack from the wall, wait 5 seconds,")
        print("         and plug it back in to reset the power supply.")
        print("  2. 🔌 USB BUS / SERIAL LOCKUP:")
        print("     • Unplug the USB cable from the Jetson, wait 3 seconds, and plug it back in.")
        print("  3. ⚠️  LOOSE MOTOR BUS CABLE:")
        print("     • Check the 3-pin servo cable connecting the controller board to Motor 1.")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    main()
