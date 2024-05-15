#include "peltier.hpp"
#include <Arduino.h>
#include <driver/ledc.h>

#define PELTIER_LEDC_CHANNEL LEDC_CHANNEL_0

Peltier::Peltier(uint8_t peltierPin) {
    this->pin = peltierPin;
}

void Peltier::init() {
    ledc_timer_config_t peltierTimerConfig = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_14_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = 2,
        .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&peltierTimerConfig);
    ledc_channel_config_t peltierChannelConfig = {
        .gpio_num = this->pin,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = PELTIER_LEDC_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = LEDC_TIMER_0,
        .duty = 10000
    };
    ledc_channel_config(&peltierChannelConfig);
}

void Peltier::setPower(uint16_t power) {
    this->power = power;
    if (this->on) {
        ledc_set_duty(LEDC_LOW_SPEED_MODE, PELTIER_LEDC_CHANNEL, PELTIER_RESOLUTION - power);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, PELTIER_LEDC_CHANNEL);
    }
    else {
        ledc_set_duty(LEDC_LOW_SPEED_MODE, PELTIER_LEDC_CHANNEL, PELTIER_RESOLUTION);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, PELTIER_LEDC_CHANNEL);
    }
}

uint16_t Peltier::getResolution() {
    return PELTIER_RESOLUTION;
}

uint16_t Peltier::getPower() {
    return this->power;
}

void Peltier::setPowerPercents(float percents) {
    this->setPower(PELTIER_RESOLUTION * percents / 100);
}

void Peltier::setState(bool on) {
    this->on = on;
    this->setPower(this->power);
}