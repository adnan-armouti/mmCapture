// Declare a variable for reading the Serial Input
int x=0;

void setup() {
  // Setup the Serial communication
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // Wait till an input is available
  while (!Serial.available());
  // Read the input and convert to int
  x = Serial.readString().toInt();
  // Send the input incremented by 1
  // Serial.print(x + 1);
  // LED ON if input > 0. Else LED OFF
  if (x > 0) {
  digitalWrite(LED_BUILTIN, HIGH);
  }
  else {
  digitalWrite(LED_BUILTIN, LOW);
  }
}
