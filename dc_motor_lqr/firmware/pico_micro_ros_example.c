#include <stdio.h>
#include <math.h>

// #include <FreeRTOS.h>
// #include <queue.h>
// #include <task.h>

#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
// #include <std_msgs/msg/int32.h>
#include <std_msgs/msg/float64.h>
#include <rmw_microros/rmw_microros.h>

#include "pico/stdlib.h"
#include "pico_uart_transports.h"

#include "pico/util/queue.h"
#include "pico/multicore.h"

#include "hardware.h"

#include "hardware/gpio.h"
#include "hardware/adc.h"

const uint MOTOR_PIN = 16;


rcl_publisher_t sine_publisher;
rcl_publisher_t cosine_publisher;
rcl_publisher_t current_publisher;
rcl_subscription_t motor_subscriber;

std_msgs__msg__Float64 sine_msg;
std_msgs__msg__Float64 cosine_msg;
std_msgs__msg__Float64 current_msg;
std_msgs__msg__Float64 motor_msg;

rclc_executor_t executor;

typedef struct {
    double sine;
    double cosine;
    double current;
} queue_entry_t;

queue_t queue;
queue_entry_t queue_val;
queue_entry_t queue_buff;


void control_core() {

    double sine = 0.0f;
    double cosine = 0.0f;
    double current = 0.0f;
    double delta_angle = 0.0f;
    double angle = 0.0f;
    double current_angle = 0.0f;
    double last_angle = 0.0f;

    static double last_angle_aa = 0.0f;
    size_t cnt = 0;

    static absolute_time_t loop_start_time;

    while (1) {
        loop_start_time = get_absolute_time();

        sine = (double) get_sine();
        cosine = (double) get_cosine();
        current = (double) get_current();

        current_angle = (double) atan2(sine, cosine);
        delta_angle = current_angle - last_angle;

        // if (delta_angle < M_PI) {
        //     // delta_angle = (M_PI - last_angle) + (-M_PI - current_angle);
        //     delta_angle = 20.0;
        // }
        angle += delta_angle;

        if (cnt > 100) {
            cnt = 0;
            queue_val.sine = sine;
            queue_val.cosine = cosine;
            // queue_val.current = current;
            queue_val.current = delta_angle;
            // queue_val.current = angle - last_angle_aa;
            // last_angle_aa = angle;
            queue_add_blocking(&queue, &queue_val);
        }
        last_angle = current_angle;
        cnt++;

        sleep_until(loop_start_time + 200);
    }
}

void motor_pwm_callback(const void * msgin) {
    const std_msgs__msg__Float64 * msg = (const std_msgs__msg__Float64 *)msgin;
    write_pwm_percent(MOTOR_PIN, msg->data);
}

// double sine_buff;
// double cosine_buff;
// double current_buff;


void timer_callback(rcl_timer_t * timer, int64_t last_call_time) {  
  RCLC_UNUSED(last_call_time);
  if (timer != NULL) {
    if(queue_try_remove(&queue, &queue_buff)) {
        sine_msg.data = (double) queue_buff.sine;
        rcl_publish(&sine_publisher, &sine_msg, NULL);

        cosine_msg.data = (double) queue_buff.cosine;
        rcl_publish(&cosine_publisher, &cosine_msg, NULL);

        current_msg.data = (double) queue_buff.current;
        rcl_publish(&current_publisher, &current_msg, NULL);
    }
  }
}

int main()
{
    stdio_init_all();

    rmw_uros_set_custom_transport(
		true,
		NULL,
		pico_serial_transport_open,
		pico_serial_transport_close,
		pico_serial_transport_write,
		pico_serial_transport_read
	);


    init_pwm(MOTOR_PIN);
    write_pwm_percent(MOTOR_PIN, 0.0);

    init_adc();

    rcl_timer_t timer;
    rcl_node_t node;
    rcl_allocator_t allocator;
    rclc_support_t support;


    allocator = rcl_get_default_allocator();

    // Wait for agent successful ping for 2 minutes. 
    const int timeout_ms = 1000; 
    const uint8_t attempts = 120;

    rcl_ret_t ret = rmw_uros_ping_agent(timeout_ms, attempts);
    

    if (ret != RCL_RET_OK)
    {
        // Unreachable agent, exiting program.
        return ret;
    }

    rclc_support_init(&support, 0, NULL, &allocator);

    rclc_node_init_default(&node, "scara_motor", "", &support);

    rclc_publisher_init_default(
        &sine_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64),
        "sine"
    );

    rclc_publisher_init_default(
        &cosine_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64),
        "cosine"
    );

    rclc_publisher_init_default(
        &current_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64),
        "current"
    );

    rclc_subscription_init_default(
        &motor_subscriber,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64),
        "motor_pwm"
    );

    const unsigned int timer_timeout = 10;
    rclc_timer_init_default(
        &timer,
        &support,
        RCL_MS_TO_NS(timer_timeout),
        timer_callback
    );

    rclc_executor_init(&executor, &support.context, 4, &allocator);
    rclc_executor_add_subscription(&executor, &motor_subscriber, &motor_msg, &motor_pwm_callback, ON_NEW_DATA);
    rclc_executor_add_timer(&executor, &timer);

    queue_init(&queue, sizeof(queue_entry_t), 5);

    multicore_launch_core1(control_core);

    while (true)
    {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));
    }
    return 0;
}
