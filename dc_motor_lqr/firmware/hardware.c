#include "hardware.h"

#include "pico/stdlib.h"
#include "hardware/pwm.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/adc.h"

#include <math.h>


void init_pwm(const unsigned int pin) {
    gpio_set_function(pin, GPIO_FUNC_PWM);

    uint slice = pwm_gpio_to_slice_num(pin);
    uint channel = pwm_gpio_to_channel(pin);

    pwm_set_wrap(slice, 2490);
    pwm_set_chan_level(slice, channel, 0);
    pwm_set_enabled(slice, true);
}


void write_pwm_percent(const unsigned int pin, float percent) {
    float level = 2490 * percent;
    pwm_set_gpio_level(pin, (uint16_t) level);
}


void init_adc() {
    adc_init();
    adc_gpio_init(26);
    adc_gpio_init(27);
}

double get_sine() {
    adc_select_input(1);
    double sine_value = (((double) adc_read()) / ((double) (1 << 12)));
    sine_value = (sine_value - 0.47401387187613986) / 0.12704995558296325;
    return sine_value;
}

double get_cosine() {
    adc_select_input(0);
    double cosine_value = (((double) adc_read()) / ((double) (1 << 12)));
    cosine_value = (cosine_value - 0.47636771811882317) / 0.10126833514930254;
    return cosine_value;
}

double get_current() {
    adc_select_input(2);
    double current_value = (double) adc_read();
    return current_value;
}