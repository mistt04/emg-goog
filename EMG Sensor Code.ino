const int EMG_PIN = A0;        // MyoWare SIG pin → Arduino A0
const int SAMPLE_RATE_HZ = 500;
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;  // 2ms

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  analogReference(DEFAULT);  // 5V reference on Mega
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;
    int rawValue = analogRead(EMG_PIN);
    // Format: timestamp,value
    Serial.print(now);
    Serial.print(",");
    Serial.println(rawValue);
  }
}
