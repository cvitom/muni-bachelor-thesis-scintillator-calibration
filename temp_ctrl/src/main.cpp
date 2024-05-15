#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <driver/ledc.h>
#include "peltier.hpp"
#include <PID_v1.h>
#include <SoftwareSerial.h>

#define PELTIER_PIN 4
#define RGB_LED_PIN 8
#define ONE_WIRE_PIN 5
#define RELAY_PIN 6
#define AMBIENT_TEMP 18
#define PELTIER_MIN_SWITCH_BACK_TIME 5000

Adafruit_NeoPixel rgb_led = Adafruit_NeoPixel(1, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature dallas(&oneWire);
Peltier peltier(PELTIER_PIN);
HardwareSerial& cmdPort = Serial;

// PID
double kp = 1700;
double ki = 70;
double kd = 1200;
double temp(24), target(24), output;
PID pid(&temp, &output, &target, kp, ki, kd, DIRECT);

bool peltierHeating = false;
uint64_t lastPeltierSwitchTime = 0;

String readCmdPart() {
    String result = "";
    while (cmdPort.available()) {
        char c = cmdPort.read();
        if (c == ' ' || c == '\n') {
            break;
        }
        result += c;
    }

    return result;
}

void switchPeltierPolarity(bool heating) {
    peltier.setState(false);
    delay(1000);
    digitalWrite(RELAY_PIN, heating);
    delay(300);
    peltier.setState(true);
    peltierHeating = heating;
    lastPeltierSwitchTime = millis();
    // Serial.println("Peltier polarity switched");
}

void set_rgb(uint8_t r, uint8_t g, uint8_t b) {
    rgb_led.setPixelColor(0, rgb_led.Color(r, g, b));
    rgb_led.show();
}

void setup() {
    Serial.begin(115200);
    // cmdPort.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);

    set_rgb(0, 50, 0);
    
    peltier.init();
    peltier.setState(true);

    dallas.begin();
    pid.SetOutputLimits(-peltier.getResolution() / 1.02, peltier.getResolution() / 4);
    pid.SetMode(AUTOMATIC);
}


void loop() {
    // put your main code here, to run repeatedly:
    // if (Serial.available()) {
    //     target = Serial.parseFloat();
    //     Serial.readStringUntil('\n');
    // }

    if (cmdPort.available()) {
        String cmd = readCmdPart();
        if (cmd == "update_temp") {
            temp = readCmdPart().toFloat();
            cmdPort.println("OK");
            pid.Compute();
        }
        else if (cmd == "set_target") {
            target = readCmdPart().toFloat();
            cmdPort.println("OK");
        }
        else if (cmd == "get_power") {
            cmdPort.println(output);
        }
        else if (cmd == "get_target") {
            cmdPort.println(target);
        }
        else if (cmd == "turn_off") {
            peltier.setState(false);
            cmdPort.println("OK");
        }
        else if (cmd == "turn_on") {
            peltier.setState(true);
            cmdPort.println("OK");
        }
        else if (cmd == "reset_pid") {
            pid.SetMode(MANUAL);
            pid.SetMode(AUTOMATIC);
            cmdPort.println("OK");
        }
        else {
            cmdPort.print("Unknown command \"");
            cmdPort.print(cmd);
            cmdPort.println("\"");
        }
    }

    if (output < 0) {
        if (peltierHeating) {
            if (
                // (target < AMBIENT_TEMP + 2 && millis() > lastPeltierSwitchTime + PELTIER_MIN_SWITCH_BACK_TIME) || 
                millis() > lastPeltierSwitchTime + PELTIER_MIN_SWITCH_BACK_TIME || 
                target < AMBIENT_TEMP - 2 ||
                temp - target > 5
            ) {
                switchPeltierPolarity(false);
                set_rgb(0, 0, 50);
                peltier.setPower(-output);
            }
            else {
                peltier.setPower(0);
            }
        }
        else
            peltier.setPower(-output);
    }
    else {
        if (!peltierHeating) {
            if (
                // (target > AMBIENT_TEMP - 2 && millis() > lastPeltierSwitchTime + PELTIER_MIN_SWITCH_BACK_TIME) || 
                millis() > lastPeltierSwitchTime + PELTIER_MIN_SWITCH_BACK_TIME || 
                target > AMBIENT_TEMP + 2
            ) {
                switchPeltierPolarity(true);
                peltier.setPower(output);
                set_rgb(50, 0, 0);
            }
            else
                peltier.setPower(0);
        }
        else
            peltier.setPower(output);

    }
    delay(10);
}
