import nidaqmx
import time

def control_led_analog(voltage=5.0):
    """
    Control LED brightness using analog output
    voltage: 0.0 = OFF, 5.0 = full brightness (max for USB-6212)
    """
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan("Dev2/ao0")  # Use ao0 or ao1
        
        # Write voltage value
        task.write(voltage, auto_start=True)
        print(f"LED voltage: {voltage}V")

# Blink the LED using analog output
print("Starting analog LED blink (1 second intervals)...")
print("Press Ctrl+C to stop")

try:
    while True:
        control_led_analog(5.0)  # LED ON (5V)
        print("LED ON")
        time.sleep(1)
        
        control_led_analog(0.0)  # LED OFF (0V)
        print("LED OFF")
        time.sleep(1)
        
except KeyboardInterrupt:
    control_led_analog(0.0)  # Ensure LED is off when stopping
    print("\nLED blinking stopped")