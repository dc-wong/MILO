#include <stdio.h>
#include "pico/stdlib.h"


int main()
{
    const uint LED_PIN = 0;
    // initialize gpio
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    // intialize serial port
    stdio_init_all();

    // loop to keep blinking
    while (true) {
        printf("ON!\n");
        gpio_put(LED_PIN, 1);
        sleep_ms(1000);

        printf("OFF!\n");
        gpio_put(LED_PIN, 0);
        sleep_ms(1000);

    }

    return 0;
}
