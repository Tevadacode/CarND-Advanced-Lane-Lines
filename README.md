# Advanced Lane Finding
# Camera Calibration
1. The camera matrix and distortion coeffients been computed correctly and check on one of the calibration images as a test. 
    The code for this step is in calibrateCamera() function defined in helper.py (line 7 to 30). 
    I started by creating an objp [54:3] (x,y,z) array of zeros.
    Then I assigned grid index to the first two columns (x,y).

    I used the images provided in camera_cal folder to find all corners of each image using cv2.findChessoardCorners and appended to imgpoints array. 
    I also appended objp to objpoints array for all images. Then I used cv2.calibrateCamera to get mtx and dist to undistort distorted images.

2. Here is an example of undistorted image:
![Undistorted Chessboard](/output_images/undistorted_chessboard.PNG)

# Pipeline (single image)
1. I have applied distortion to a single image correctly.
   Here is an example:
   ![Undistorted Image](/output_images/undistorted_image.PNG) 

2. A binary image has been created using gray scale, gradient, and 
    ![Undistorted Image](/output_images/grayscale.jpg)
    Gray scale image

    ![Undistorted Image](/output_images/xsobel.jpg)
    Gradient result along x-axis

    ![Undistorted Image](/output_images/s_channel.jpg)
    Binary S channel from HLS image

    ![Undistorted Image](/output_images/combined_binary.jpg)
    Combined gradient and s channel binary image
3. Applying a perspective transorm to rectify an image
    ![Undistorted Image](/output_images/masked_image.jpg)
    Masked image from the binary result
    
    ![Undistorted Image](/output_images/binary_warped.jpg)
    Warped image from the masked image
 
 4.  
# Pipeline (Video)
Click on the image to see the video result
[![Undistorted Image](/output_images/result.jpg)](https://youtu.be/YCKgXicwOiU)