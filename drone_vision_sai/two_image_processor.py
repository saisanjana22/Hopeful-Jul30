from __future__ import print_function
import cv2
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import glob
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# --- Image loading is correctly outside the class ---
image_folder = "."
image_paths = glob.glob(f"{image_folder}/*.jpg")
image_paths.sort()
images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
        print(f"Loaded image from {path}")
    else:
        print(f"Warning: Could not load image from {path}")


class StandaloneImageProcessor:
    def __init__(self, lower_hsv, upper_hsv, kp_angular, linear_velocity, look_ahead_y_ratio=0.9):
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.kp_angular = kp_angular
        self.linear_velocity = linear_velocity
        self.look_ahead_y_ratio = look_ahead_y_ratio # Ratio down the image to look for the line

        print("Image Processor Initialized with parameters:")
        print(f"  Lower HSV: {self.lower_hsv}")
        print(f"  Upper HSV: {self.upper_hsv}")
        print(f"  Kp Angular: {self.kp_angular}")
        print(f"  Linear Velocity: {self.linear_velocity}")
        print(f"  Look Ahead Y Ratio: {self.look_ahead_y_ratio}")

    def _my_regression(self, xs, ys):
        xs = xs.astype(float)
        ys = ys.astype(float)
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        xy_mean = np.mean(xs * ys)
        x_squared_mean = np.mean(xs ** 2)
        denominator = (x_mean ** 2 - x_squared_mean)
        if abs(denominator) < 1e-9:
            return np.nan, np.nan
        m = (x_mean * y_mean - xy_mean) / denominator
        b = y_mean - m * x_mean
        return m, b

    def _opencv_regression(self, points):
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if vx == 0:
            m_cv = np.inf if vy > 0 else -np.inf
        else:
            m_cv = vy / vx
        b_cv = y0 - m_cv * x0
        return m_cv, b_cv

    def _scipy_regression(self, xs, ys):
        res = linregress(xs, ys)
        return res.slope, res.intercept

    def process_image(self, image):
        """
        Processes a single OpenCV image to detect a line and calculate control outputs.
        Returns:
            - mask (grayscale image showing detected line)
            - error (deviation from center, x-coordinate in pixels)
            - linear_velocity (constant)
            - angular_velocity (calculated based on error)
            - (optional) tuple of all three regression lines' params for plotting
              (my_m, my_b), (ocv_m, ocv_b), (sp_m, sp_b)
        """

        height, width, _ = image.shape # Assuming color image for consistency
        blurred_image = cv2.GaussianBlur(image, (9, 9), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_hsv, self.upper_hsv)

        kernelerosion = np.ones((9, 9), np.uint8)
        kerneldilation = np.ones((39, 39), np.uint8)
        mask = cv2.erode(mask, kernelerosion, iterations=1)
        mask = cv2.dilate(mask, kerneldilation, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize regression line parameters for optional plotting
        my_reg_params = (np.nan, np.nan)
        ocv_reg_params = (np.nan, np.nan)
        sp_reg_params = (np.nan, np.nan)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            min_contour_area = 100

            if cv2.contourArea(largest) >= min_contour_area:
                points = largest.squeeze()
                if points.ndim == 2 and points.shape[0] >= 2:
                    xs, ys = points[:, 0], points[:, 1]

                    # Calculate all three regressions for visualization
                    my_reg_params = self._my_regression(xs, ys)
                    ocv_reg_params = self._opencv_regression(points)
                    sp_reg_params = self._scipy_regression(xs, ys)

        return mask, (my_reg_params, ocv_reg_params, sp_reg_params)


# --- STANDALONE FUNCTION FOR DIRECTIONALITY / CONTROL LOGIC ---
def calculate_drone_directionality(ocv_m, ocv_b, image_width, image_height,
                                   kp_angular, linear_velocity, look_ahead_y_ratio=0.7):
    """
    Converts the OpenCV regression line (slope and intercept) into
    lateral error and angular velocity control signals for a drone.

    Args:
        ocv_m (float): Slope of the OpenCV regression line.
        ocv_b (float): Intercept of the OpenCV regression line.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        kp_angular (float): Proportional gain for angular velocity.
        linear_velocity (float): Desired constant linear velocity.
        look_ahead_y_ratio (float, optional): Ratio down the image to look for the line. Defaults to 0.9.

    Returns:
        tuple: (error_pixels, linear_velocity_x, angular_velocity_z)
               Returns (0.0, linear_velocity, 0.0) if the line parameters are invalid (NaN/Inf).
    """
    error_pixels = 0.0
    angular_velocity_z = 0.0
    linear_velocity_x = linear_velocity

    if np.isnan(ocv_m) or np.isnan(ocv_b):
        # print("  Warning: OpenCV regression line resulted in NaN. Defaulting to 0 error.")
        return error_pixels, linear_velocity_x, angular_velocity_z
    
    # Calculate the y-coordinate for "look-ahead" in pixels
    look_ahead_y = int(image_height * look_ahead_y_ratio)

    if np.isinf(ocv_m):
        # If slope is infinite (vertical line), the x-coordinate is constant.
        # We need an explicit x-value if the slope is infinite.
        # The 'b_cv' in this case doesn't mean y-intercept; it's the intercept for x = b.
        # But this implies a specific (x0, y0) from fitLine.
        # For a truly vertical line, we'd ideally use the x0 from fitLine, or the mean x of points.
        # Since we only have m and b here, we'll make a pragmatic choice.
        # If the line is 'vertical' and its calculated 'b' (intercept) is finite,
        # we can assume 'b' represents the x-coordinate of this vertical line.
        # This is a heuristic that works if b_cv is correctly returned as the x-intercept when m_cv is inf.
        if np.isinf(ocv_b): # If both are inf, it's highly ambiguous, default to center.
            regression_line_x_at_look_ahead = image_width // 2
        else: # m is inf, b is finite, likely x = b (a vertical line at x=b)
            regression_line_x_at_look_ahead = int(ocv_b)
        
        # print(f"  Warning: OpenCV regression line is vertical (infinite slope). Predicted x_at_lookahead: {regression_line_x_at_look_ahead}")
    else:
        regression_line_x_at_look_ahead = int(ocv_m * look_ahead_y + ocv_b)

    # Calculate Error (deviation from the center of the image)
    error_pixels = regression_line_x_at_look_ahead - (image_width / 2)

    # Calculate angular velocity (turning speed)
    angular_velocity_z = -float(error_pixels) * kp_angular

    return error_pixels, linear_velocity_x, angular_velocity_z


# --- Main execution block ---
if not images:
    print("No images found or loaded. Exiting.")
else:
    processor = StandaloneImageProcessor(
        lower_hsv=[20, 100, 100], # EXAMPLE: Tune these for your line color (e.g., yellow)
        upper_hsv=[30, 255, 255], # EXAMPLE: Tune these for your line color (e.g., yellow)
        kp_angular=0.01,          # Proportional gain for angular velocity - needs tuning
        linear_velocity=0.5,      # Constant forward velocity
        look_ahead_y_ratio=0.9    # Drone looks 90% down the image for the line
    )

    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")
        
        # process_image now returns mask and all regression params
        mask, reg_params = processor.process_image(img)

        # Extract OpenCV regression parameters to pass to the external function
        my_reg, ocv_reg, sp_reg = reg_params
        ocv_m, ocv_b = ocv_reg # This is the line we want the drone to follow!

        height, width, _ = img.shape

        # --- Call the new standalone function to get control outputs ---
        error, linear_vel, angular_vel = calculate_drone_directionality(
            ocv_m, ocv_b, width, height,
            processor.kp_angular, processor.linear_velocity, processor.look_ahead_y_ratio
        )

        print(f"  Calculated Error: {error:.2f} pixels")
        print(f"  Linear Velocity: {linear_vel:.2f} m/s")
        print(f"  Angular Velocity: {angular_vel:.2f} rad/s (turns { 'right' if angular_vel < 0 else 'left' if angular_vel > 0 else 'straight'})")


        # --- Plotting section for visualization ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Image {i+1}\nError: {error:.2f} px, Angular Vel: {angular_vel:.2f} rad/s")

        if reg_params and reg_params[0] is not None:
            x_vals = np.array([0, width - 1])
            look_ahead_y = int(height * processor.look_ahead_y_ratio)

            # Plot My Regression
            m, b = my_reg
            if not np.isnan(m) and not np.isinf(m):
                y_vals = m * x_vals + b
                ax.plot(x_vals, y_vals, color='lime', linewidth=2, label='My Regression')
            else:
                print(f"  My Regression for image {i+1} resulted in NaN/Inf. Not plotted.")

            # Plot OpenCV Regression (the one used for control)
            m_cv, b_cv = ocv_reg
            if np.isinf(m_cv):
                # If you need to plot a specific vertical line, you'd need the x0 from fitLine.
                # For now, if m_cv is inf, and b_cv is finite (meaning x=b_cv), plot at b_cv.
                # Otherwise, default to center or handle with a better strategy.
                if not np.isinf(b_cv) and not np.isnan(b_cv):
                    ax.axvline(x=b_cv, color='red', linewidth=2, linestyle='--', label='OpenCV Reg (Vertical)')
                else: # Fallback if b_cv is also problematic
                    ax.axvline(x=width/2, color='red', linewidth=2, linestyle='--', label='OpenCV Reg (Vertical - Approx)')
                print(f"  OpenCV Regression for image {i+1} is vertical. Plotting vertical line.")
            elif not np.isnan(m_cv):
                y_cv = m_cv * x_vals + b_cv
                ax.plot(x_vals, y_cv, color='red', linewidth=2, linestyle='--', label='OpenCV Regression (Used for Control)')
            else:
                print(f"  OpenCV Regression for image {i+1} resulted in NaN. Not plotted.")

            # Plot Scipy Regression
            m_sp, b_sp = sp_reg
            if not np.isnan(m_sp) and not np.isinf(m_sp):
                y_sp = m_sp * x_vals + b_sp
                ax.plot(x_vals, y_sp, color='blue', linewidth=2, linestyle=':', label='Scipy Regression')
            else:
                print(f"  Scipy Regression for image {i+1} resulted in NaN/Inf. Not plotted.")

            # --- Plot the look-ahead point and error vector based on the calculated values ---
            # These values (error, linear_vel, angular_vel) come from the calculate_drone_directionality function
            
            # Re-calculate look_ahead_x for plotting consistency
            if not np.isnan(ocv_m) and not np.isnan(ocv_b): # Only plot lookahead if OpenCV line is valid
                look_ahead_y = int(height * processor.look_ahead_y_ratio) # Recalculate if not passed
                if np.isinf(ocv_m):
                    if not np.isinf(ocv_b) and not np.isnan(ocv_b):
                        x_lookahead_pt = int(ocv_b)
                    else:
                        x_lookahead_pt = width // 2
                else:
                    x_lookahead_pt = int(ocv_m * look_ahead_y + ocv_b)
                
                ax.plot(x_lookahead_pt, look_ahead_y, 'o', color='cyan', markersize=12, markeredgecolor='black', label='Lookahead Point (OpenCV)')
                ax.axvline(x=width/2, color='gray', linestyle=':', linewidth=1, label='Image Center')
                if abs(error) > 0.5:
                    ax.plot([width/2, x_lookahead_pt], [look_ahead_y, look_ahead_y], 'k--', linewidth=1, label='Error Vector')
                    ax.text( (width/2 + x_lookahead_pt)/2, look_ahead_y - 20, f'Error: {error:.0f}',
                             color='black', ha='center', va='bottom',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

            ax.legend(loc='upper right')
        else:
            ax.set_title(f"Image {i+1}: No line detected or not enough points for regression.")

        plt.axis('off')
        plt.tight_layout()
        plt.show()