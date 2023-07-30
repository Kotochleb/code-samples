#ifndef HARDWARE
#define HARDWARE

void init_pwm(const unsigned int pin);
void write_pwm_percent(const unsigned int pin, float percent);

void init_adc();
// double get_encoder_angle();
double get_sine();
double get_cosine();
double get_current();

#endif //HARDWARE
