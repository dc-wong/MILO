const int bit0 = 2;
const int bit1 = 3;
const int bit2 = 4;
const int bit3 = 5;

#include <Servo.h>

Servo myservo;  // create servo object to control a servo
int pos = 0;    // variable to store the servo position
int bin_num = 0;

void setup() {
  Serial.begin(9600); // Initialize serial communication
  myservo.attach(11);
  pinMode(bit0, INPUT);
  pinMode(bit1, INPUT);
  pinMode(bit2, INPUT);
  pinMode(bit3, INPUT);
}

int convertToBinary(int a, int b, int c, int d) {
  int a1 = digitalRead(a) == HIGH ? 1 : 0;
  int b1 = digitalRead(b) == HIGH ? 1 : 0;
  int c1 = digitalRead(c) == HIGH ? 1 : 0;
  int d1 = digitalRead(d) == HIGH ? 1 : 0;

  int binary[] = {d1, c1, b1, a1};
  int binaryNumber = 0;

  for (int i = 0; i < 4; i++) {
    binaryNumber = (binaryNumber << 1) | binary[i];
  }
  return binaryNumber;
}

void loop() {
  int binN = convertToBinary(bit0, bit1, bit2, bit3);

  // Only execute if the binary number has changed
  if (binN != bin_num) {
    Serial.print("Binary number: ");
    Serial.println(binN);

    // Execute the corresponding movement based on the binary number
    if (binN == 2) {
      for (int x = 0; x <= 180; x += 1) {
        pos = x;
        myservo.write(pos);
        delay(15);
      }
    } 
    else if (binN == 3) {
      for (int x = 180; x >= 0; x -= 1) {
        pos = x;
        myservo.write(pos);
        delay(15);
      }
    } 
    else if (binN == 4) {
      for (int x = pos; x >= 0; x -= 1) {
        pos = x;
        myservo.write(pos);
        delay(15);
      }
    } 
    else if (binN == 15) {
      myservo.write(pos);
    }

    // Update the stored binary number
    bin_num = binN;
  }
}
