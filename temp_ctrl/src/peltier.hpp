#pragma once
#include <stdint.h>
#define PELTIER_RESOLUTION 16384
class Peltier {
public:
    Peltier(uint8_t pin);
    void init();
    void setPower(uint16_t power);
    uint16_t getPower();
    void setPowerPercents(float percents);
    void setState(bool on);
    uint16_t getResolution();
private:
    uint8_t pin;
    uint16_t power;
    bool on;
};