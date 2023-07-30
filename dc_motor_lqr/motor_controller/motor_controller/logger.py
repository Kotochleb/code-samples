import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Float64
from rclpy.parameter import Parameter

import numpy as np
import pandas as pd

import math
import time


class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        self._sine = None
        self._cosine = None
        self._current = None

        # create sine and cosine callback
        self._sine_subscriber = self.create_subscription(Float64, 'sine', self._sine_callback, ReliabilityPolicy.RELIABLE)
        self._cosine_subscriber = self.create_subscription(Float64, 'cosine', self._cosine_callback, ReliabilityPolicy.RELIABLE)
        self._current_subscriber = self.create_subscription(Float64, 'current', self._current_callback, ReliabilityPolicy.RELIABLE)


        # await data on sine and cosine
        # while self._sine is None or self._cosine is None:
        #     self.get_logger().info('Waiting for /sine and /cosine topics')
        #     time.sleep(1)
        

        # create /motor_pwm topic publisher
        self._pwm_publisher = self.create_publisher(Float64, 'motor_pwm', 10)

        # create timer to publish data at 100 Hz
        self._cnt = 0
        self._max_points = 2000
        self._points = np.zeros((3, self._max_points))
        timer_period = 1./100. # 100 Hz
        self._timer = self.create_timer(timer_period, self._timer_callback)


    def _sine_callback(self, msg):
        # extract data from sine message
        self._sine = msg.data


    def _cosine_callback(self, msg):
        # extract data from cosine message
        self._cosine = msg.data

    def _current_callback(self, msg):
        # extract data from cosine message
        self._current = msg.data


    def _timer_callback(self):
        if self._sine is not None and self._cosine is not None and self._current is not None:
            if self._cnt < self._max_points:
                self._points[0, self._cnt] = self._sine
                self._points[1, self._cnt] = self._cosine
                self._points[2, self._cnt] = self._current
                self._cnt += 1
            else:
                print(self._points)
                df = pd.DataFrame(self._points.T)
                df.to_csv('/ros2_ws/src/motor_controller/logs/measurement.csv')
                exit(0)




def main(args=None):
    rclpy.init(args=args)

    motor_controller = MotorController()

    rclpy.spin(motor_controller)

    motor_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()