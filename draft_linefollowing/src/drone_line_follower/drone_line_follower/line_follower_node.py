""" The primary task of this node is to see the line and figure out how to keep the drone on it
Sends continuous commands to the drone to follow the line.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist # Standard message for velocity commands
from cv_bridge import CvBridge # For converting ROS Image messages to OpenCV format
import cv2 # OpenCV for image processing
import numpy as np # NumPy for numerical operations

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        #Parameters for line detection
        self.declare_parameter('lower_hsv_h', 30) # Hue lower bound
        self.declare_parameter('lower_hsv_s', 100) # Saturation lower bound
        self.declare_parameter('lower_hsv_v', 100) # Value lower bound

        self.declare_parameter('upper_hsv_h', 90) # Hue upper bound
        self.declare_parameter('upper_hsv_s', 255) # Saturation upper bound
        self.declare_parameter('upper_hsv_v', 255) # Value upper bound

        self.declare_parameter('kp_angular', 0.01)
        self.declare_parameter('linear_velocity', 0.5) #constant forward speed

        # Let's the code know what the parameters are
        self.update_parameters()

        self.image_subscriber = self.create_subscription(
            # The Image recieved from the camera is sent to the image_callback method.
            Image,
            '/camera/image_raw', # MAKE SURE THIS TOPIC MATCHES YOUR CAMERA
            self.image_callback, #most important method, it will be called every time a new image is received!!
            10
            )
        
        self.image_subscriber # Keeps code running when no other subscribers are present
        
        self.velocity_publisher = self.create_publisher(Twist, '/fmu/in/setpoint_velocity/cmd_vel', 10) #publishes linear velocity and angilar velocity
        self.cmd_vel_msg = Twist()

        self.mask_publisher = self.create_publisher(Image, '/detectedLine', 10) # publishes the formated image with the detected line

        self.bridge = CvBridge() # makes a new translator between ROS Image messages and OpenCV images
        self.get_logger().info("Line Follower Node Initialized. Waiting for image data...") # telemetry in the terminal

        self.timer = self.create_timer(1.0, self.update_parameters) #update parameters will be triggered every second

    def update_parameters(self): #to tune drone behavior while it is running
       #update the parameters from the terminal using (ros2 param set /line_follower_node kp_angular 0.02)
       self.lower_hsv = np.array([
           self.get_parameter('lower_hsv_h').value,
           self.get_parameter('lower_hsv_s').value,
           self.get_parameter('lower_hsv_v').value
       ])
       self.upper_hsv = np.array([
           self.get_parameter('upper_hsv_h').value,
           self.get_parameter('upper_hsv_s').value,
           self.get_parameter('upper_hsv_v').value
       ])
       self.kp_angular = self.get_parameter('kp_angular').value
       self.linear_velocity = self.get_parameter('linear_velocity').value

    