import numpy as np
x_offset = 50
y_offset = 60
sobelx_kernel_size = 3
sobely_kernel_size = 3
mag_kernel_size = 3
dir_kernel_size = 3
sobelx_threshold = (20, 150)
sobely_threshold = (30, 100)
mag_threshold = (30, 100)
direction_threshold = (0.85, 1.3)
src_perspective = np.float32([[(202, 720),(600, 446), (680, 446), (1112,720)]])
dst_perspective = np.float32([[(340, 720), (340,0), (940, 0) , (940, 720)]])
# Define conversions in x and y from pixels space to meters
lane_width = dst_perspective[0][3][0] - dst_perspective[0][0][0]
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/lane_width # meters per pixel in x dimension
frame_counter = 0
n_threshold = 3
debug_on = False

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

leftLine = Line()
rightLine = Line()
