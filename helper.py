import numpy as np
import cv2
import glob
from p4constant import *
import matplotlib.pyplot as plt

def calibrateCamera():
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    images = glob.glob('./camera_cal/calibration*.jpg')
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if idx == 1:
            img_size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    return ret, mtx, dist, rvecs, tvecs

def cal_undistort(img, mtx, dist):
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    if debug_on == True:
        cv2.imwrite("output_images/undistor.jpg", undist_img)
    return undist_img

def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug_on == True:
        cv2.imwrite("output_images/grayscale.jpg", gray_img)
        #cv2.imshow('gray_img', gray_img)
        #cv2.waitKey(0)
    return gray_img

def hls(img, thresh=(0,255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls_img[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s)
    s_binary[(s >= thresh[0]) & (s <= thresh[1])] = 1    
    if debug_on == True:
        cv2.imwrite("output_images/s_channel.jpg", s_binary*255)
        #cv2.imshow('saturation', s_binary*255)
        #cv2.waitKey(0)
    return s_binary

def abs_sobel_thresh(img, orient='x', sobel_kernel= 3, abs_thresh=(0,255)):
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1
    if debug_on == True:
        file_name = orient + "sobel.jpg"
        cv2.imwrite("output_images/" + file_name, binary_output*255)
        #cv2.imshow(orient, binary_output*255)
        #cv2.waitKey(0)
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0,255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1
    if debug_on == True:
        binary_output = binary_output * 255
        cv2.imshow("mag", binary_output)
        cv2.waitKey(0)
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_gredient = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_gredient)
    binary_output[(dir_gredient >= thresh[0]) & (dir_gredient <= thresh[1])] = 1
    if debug_on == True:
        binary_output = binary_output * 255
        cv2.imshow("direction", binary_output)
        cv2.waitKey(0)

    # 6) Return this mask as your binary_output image
    
    return binary_output

def combined(img):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=sobelx_kernel_size, abs_thresh=sobelx_threshold)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=sobely_kernel_size, abs_thresh=sobely_threshold)
    mag_binary = mag_thresh(img, sobel_kernel= mag_kernel_size, mag_thresh=mag_threshold)
    dir_binary = dir_threshold(img, sobel_kernel=dir_kernel_size, thresh=direction_threshold)
    combined = np.zeros_like(gradx)
    combined[((gradx == 1)&(grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    if debug_on == True:
        combined[((gradx == 255)&(grady == 255)) | ((mag_binary == 255) & (dir_binary == 255))] = 255
        cv2.imshow('combined', combined)
        cv2.waitKey(0)
    return combined

def combined_binary(s_channel, edge_image):
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(s_channel == 1) |( edge_image == 1)] = 1
    if debug_on == True:
        cv2.imwrite("output_images/combined_binary.jpg", combined_binary*255)
    return combined_binary

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    if debug_on == True:
        cv2.imwrite("output_images/masked_image.jpg", masked_image*255)

    return masked_image

def warpPerspective(img, M, size, saved_name):
    # calculate warped image
    binary_warped_img = cv2.warpPerspective(img, M, size)
    if debug_on == True:
        cv2.imwrite("output_images/"+ saved_name, binary_warped_img*255)
    return binary_warped_img



def calculate_radius(left_fitx, right_fitx, ploty):
    left_fitx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
    right_fitx = right_fitx[::-1]  # Reverse to match top-to-bottom in y
    
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    center_l_r = (left_fitx[0] + right_fitx[0])/2
    
    return left_curverad, right_curverad, center_l_r

import p4constant

def sanity_check(left_fitx, right_fitx, left_curverad, right_curverad):
    look_good = True
    diff = np.subtract(right_fitx, left_fitx)
    abs_diff = np.absolute(diff)
    indx = np.argmin(diff)
    min_distance = diff[indx]
    mean_distance = np.mean(diff)
    
    curved_distance = len(abs_diff[(abs_diff < (mean_distance - 100)) | (abs_diff > (mean_distance + 100)) ])
    if curved_distance > 5:
        look_good = False
    return look_good

def p4reset():
    p4constant.leftLine = Line()
    p4constant.rightLine = Line()
    p4constant.frame_counter = 0
    
def sliding_search(img, margin):
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    #margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &\
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &\
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    p4constant.leftLine.allx = nonzerox[left_lane_inds]
    p4constant.leftLine.ally = nonzeroy[left_lane_inds] 
    p4constant.rightLine.allx = nonzerox[right_lane_inds]
    p4constant.rightLine.ally = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each (get back coefficients
    left_current_fit = np.polyfit(p4constant.leftLine.ally, p4constant.leftLine.allx, 2)
    right_current_fit = np.polyfit(p4constant.rightLine.ally, p4constant.rightLine.allx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_current_fit[0]*ploty**2 + left_current_fit[1]*ploty + left_current_fit[2]
    right_fitx = right_current_fit[0]*ploty**2 + right_current_fit[1]*ploty + right_current_fit[2]

    left_curvature, right_curvature, center_l_r = calculate_radius(left_fitx, right_fitx, ploty)

    look_good = sanity_check(left_fitx, right_fitx, left_curvature, right_curvature)

    if look_good == True:
        if p4constant.frame_counter < p4constant.n_threshold:         
            p4constant.leftLine.current_fit.append(left_current_fit)
            p4constant.rightLine.current_fit.append(right_current_fit)

            p4constant.leftLine.recent_xfitted.append(left_fitx)
            p4constant.rightLine.recent_xfitted.append(right_fitx)
        else:
            indx = p4constant.frame_counter % p4constant.n_threshold
            p4constant.leftLine.current_fit[indx] = left_current_fit
            p4constant.rightLine.current_fit[indx] = right_current_fit

            p4constant.leftLine.recent_xfitted[indx] = left_fitx
            p4constant.rightLine.recent_xfitted[indx] = right_fitx
            
        p4constant.leftLine.bestx = np.divide(np.sum(p4constant.leftLine.recent_xfitted, axis=0), len(p4constant.leftLine.recent_xfitted))
        p4constant.rightLine.bestx = np.divide(np.sum(p4constant.rightLine.recent_xfitted, axis=0), len(p4constant.rightLine.recent_xfitted))
        
        p4constant.leftLine.best_fit = np.divide(np.sum(p4constant.leftLine.current_fit, axis=0), len(p4constant.leftLine.current_fit))
        p4constant.rightLine.best_fit = np.divide(np.sum(p4constant.rightLine.current_fit, axis=0),len(p4constant.rightLine.current_fit))

        p4constant.leftLine.radius_of_curvature = left_curvature
        p4constant.rightLine.radius_of_curvature = right_curvature
        
        p4constant.frame_counter += 1
        p4constant.leftLine.detected = True
        p4constant.rightLine.detected = True
    else:
        if len(p4constant.leftLine.recent_xfitted) == 0:
            left_fitx = []
            right_fitx = []
        else:
            left_fitx = p4constant.leftLine.bestx
            right_fitx = p4constant.rightLine.bestx
        p4constant.leftLine.detected = False
        p4constant.rightLine.detected = False    

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, left_curvature, right_curvature, center_l_r

def fillPoly(img, left_fitx, right_fitx, margin, ploty):
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp



def search_previous_data(img, margin, ploty):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (p4constant.leftLine.best_fit[0]*(nonzeroy**2) + p4constant.leftLine.best_fit[1]*nonzeroy + p4constant.leftLine.best_fit[2] - margin))\
                      & (nonzerox < (p4constant.leftLine.best_fit[0]*(nonzeroy**2) + p4constant.leftLine.best_fit[1]*nonzeroy + p4constant.leftLine.best_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (p4constant.rightLine.best_fit[0]*(nonzeroy**2) + p4constant.rightLine.best_fit[1]*nonzeroy + p4constant.rightLine.best_fit[2] - margin))\
                       & (nonzerox < (p4constant.rightLine.best_fit[0]*(nonzeroy**2) + p4constant.rightLine.best_fit[1]*nonzeroy + p4constant.rightLine.best_fit[2] + margin)))  


    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_current_fit = np.polyfit(lefty, leftx, 2)
    right_current_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    left_fitx = left_current_fit[0]*ploty**2 + left_current_fit[1]*ploty + left_current_fit[2]
    right_fitx = right_current_fit[0]*ploty**2 + right_current_fit[1]*ploty + right_current_fit[2]

    left_curvature, right_curvature, center_l_r = calculate_radius(left_fitx, right_fitx, ploty)

    look_good = sanity_check(left_fitx, right_fitx, left_curvature, right_curvature)

    if look_good == True:
        if p4constant.frame_counter < p4constant.n_threshold:         
            p4constant.leftLine.current_fit.append(left_current_fit)
            p4constant.rightLine.current_fit.append(right_current_fit)

            p4constant.leftLine.recent_xfitted.append(left_fitx)
            p4constant.rightLine.recent_xfitted.append(right_fitx)
        else:
            indx = p4constant.frame_counter % p4constant.n_threshold
            p4constant.leftLine.current_fit[indx] = left_current_fit
            p4constant.rightLine.current_fit[indx] = right_current_fit

            p4constant.leftLine.recent_xfitted[indx] = left_fitx
            p4constant.rightLine.recent_xfitted[indx] = right_fitx
            
        p4constant.leftLine.bestx = np.divide(np.sum(p4constant.leftLine.recent_xfitted, axis=0), len(p4constant.leftLine.recent_xfitted))
        p4constant.rightLine.bestx = np.divide(np.sum(p4constant.rightLine.recent_xfitted, axis=0), len(p4constant.rightLine.recent_xfitted))
        
        p4constant.leftLine.best_fit = np.divide(np.sum(p4constant.leftLine.current_fit, axis=0), len(p4constant.leftLine.current_fit))
        p4constant.rightLine.best_fit = np.divide(np.sum(p4constant.rightLine.current_fit, axis=0),len(p4constant.rightLine.current_fit))

        p4constant.leftLine.radius_of_curvature = left_curvature
        p4constant.rightLine.radius_of_curvature = right_curvature
        
        p4constant.frame_counter += 1
        p4constant.leftLine.detected = True
        p4constant.rightLine.detected = True

    else:
        if len(p4constant.leftLine.recent_xfitted) == 0:
            left_fitx = []
            right_fitx = []
        else:
            left_fitx = p4constant.leftLine.bestx
            right_fitx = p4constant.rightLine.bestx

        p4constant.leftLine.detected = False
        p4constant.rightLine.detected = False

    return left_fitx, right_fitx,left_curvature, right_curvature, center_l_r

def search_from_scratch(img, margin, ploty):
    window_img, left_fitx, right_fitx, left_curvature, right_curvature, center_l_r = sliding_search(img, margin)
    color_warp = fillPoly(img, left_fitx, right_fitx, margin, ploty)
        
    return color_warp, left_curvature, right_curvature, center_l_r


def find_lines(img):
    margin = 20
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    if  p4constant.leftLine.detected == False and  p4constant.rightLine.detected == False:
        color_warp, left_curverad, right_curverad, center_l_r = search_from_scratch(img, margin, ploty)
    else:
        left_fitx, right_fitx, left_curverad, right_curverad, center_l_r = search_previous_data(img, margin, ploty)      
        color_warp = fillPoly(img, left_fitx, right_fitx, margin, ploty)
            
    return color_warp, left_curverad, right_curverad, center_l_r

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    result = cv2.addWeighted(initial_img, α, img, β, λ)
    return result

def putText(img, left_curverad, right_curverad, center_l_r):
    cv2.putText(img, "Radius of Curvature = " + "{0:.2f}".format(left_curverad) + "(m)", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    vehicle_center = img.shape[1]/2
    vehicle_pos = (center_l_r - vehicle_center) * xm_per_pix
    abs_pos = np.absolute(vehicle_pos)
    if vehicle_pos < 0:
        cv2.putText(img, "Vehicle is " + "{0:.2f}".format(abs_pos) +"m left of the center", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    elif vehicle_pos > 0:
        cv2.putText(img, "Vehicle is " + "{0:.2f}".format(abs_pos) +"m right of the center", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    else:
        cv2.putText(img, "Vehicle is in the center of the lane", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    if debug_on == True:
        cv2.imwrite("output_images/result.jpg", img)
    return img
