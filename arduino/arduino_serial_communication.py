import serial
import time

class ArduinoSerial:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    
    def write_read(self, serial_input):
        # Send input data
        self.write(serial_input)
        # Pause
        time.sleep(0.05)
        # Read the data from the Arduino
        serial_output = self.read()
        return serial_output

    def write(self, serial_input):
        if type(serial_input) == int or type(serial_input) == float:
            # Convert to string if int or float
            serial_input = str(serial_input)
        elif type(serial_input) != str:
            # If not int or float, check if string else raise error
            raise TypeError("Input to serial write to the Arduino must be a string, int or float")
        else:
            # Dummy statement
            pass
        # Send the input value as bytes
        self.arduino.write(bytes(serial_input, 'utf-8'))

    def read(self):
        # Read the data from the Arduino. Will output b'' for no data from the Arduino
        return self.arduino.readline()

    def flush_serial(self):
        # Flush the serial port
        self.read()
        self.write(1)
        self.read()
        self.write(0)
        self.read()
        time.sleep(3)

    def feed_barker_code(self, sample_period=1):
        # Coded for Barker 13 code [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1]
        barker_code = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        for sample in barker_code:
            self.write(sample)
            time.sleep(sample_period/100)
        self.write(0)
        
if __name__ == '__main__':
    # Start the Serial communication
    port = 'COM5'
    uno = ArduinoSerial(port)

    # Feed in the Barker Code
    print("Barker Code Start")
    uno.flush_serial()
    while True:
        uno.feed_barker_code()
    print("Barker Code End")

    # Input values to test the Serial communication for custom cases
    while True:
        # Taking input from user
        num = input("Enter a number: ") 
        # Perform the transmission and reception operation
        value = uno.write_read(num)
        # Printing the value
        print(value)