from helper import *
import matplotlib.pyplot as plt
# Import everything needed to edit/save/watch video clips

def process_image(image, ret, mtx, dist, rvecs, tvecs, roi, M, Minv):
    # undistort image
    undistort_img = cal_undistort(image, mtx, dist)
    # convert to different color space
    gray_img = grayscale(undistort_img)
    # get s_channel form undistort_img
    s = hls(undistort_img, (120, 150))
    # sobelx gradients
    edge_img = abs_sobel_thresh(gray_img, 'x', sobel_kernel=sobelx_kernel_size,\
                                abs_thresh=sobelx_threshold)#combined(s1)

    combined_binary_img = combined_binary(s, edge_img)
    # filter regoin of interest
    binary_roi_img = region_of_interest(combined_binary_img, roi)
    # calculate warped image
    binary_warped_img = warpPerspective(binary_roi_img, M, gray_img.shape[::-1], "binary_warped.jpg")
    # find lines
    color_warp, left_curverad, right_curverad, center_l_r = find_lines(binary_warped_img)
    # remap to the orignal image
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warpPerspective(color_warp, Minv, gray_img.shape[::-1], "newwarp.jpg") 
    # Combine the result with the original image
    result = weighted_img(newwarp, undistort_img, 1, 0.3, 0)
    result = putText(result, left_curverad, right_curverad, center_l_r)
    return result
    
def main():
    ret, mtx, dist, rvecs, tvecs = calibrateCamera()
    image = cv2.imread('test_images/test2.jpg')
    imshape = image.shape[0:2]
    roi = np.array([[(100, imshape[0]),\
                          (imshape[1]/2 - x_offset, imshape[0]/2 + y_offset),\
                          (imshape[1]/2 + x_offset + 25, imshape[0]/2 + y_offset),\
                          (imshape[1]- x_offset,imshape[0])]], dtype=np.int32)

    M = cv2.getPerspectiveTransform(src_perspective, dst_perspective)
    Minv = cv2.getPerspectiveTransform(dst_perspective, src_perspective)
    '''
    result = process_image(image, ret, mtx, dist, rvecs, tvecs, roi, M, Minv)
    cv2.imshow('test', result)
    cv2.waitKey(0)
    '''
    video_name = 'project_video.mp4'

    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('project_output.avi',fourcc, 25.0, (imshape[1],imshape[0]))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Our operations on the frame come here
        result = process_image(frame, ret, mtx, dist, rvecs, tvecs, roi, M, Minv) 
        # Display the resulting frame
        out.write(result)
        cv2.imshow('frame', result)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
main()

def test():
    ret, mtx, dist, rvecs, tvecs = calibrateCamera()
    image = cv2.imread('test_images/straight_lines1.jpg')
    M = cv2.getPerspectiveTransform(src_perspective, dst_perspective)
    Minv = cv2.getPerspectiveTransform(dst_perspective, src_perspective)
    imshape = image.shape[0:2]
    roi = np.array([[(100, imshape[0]),\
                          (imshape[1]/2 - x_offset, imshape[0]/2 + y_offset),\
                          (imshape[1]/2 + x_offset + 25, imshape[0]/2 + y_offset),\
                          (imshape[1]- x_offset,imshape[0])]], dtype=np.int32)
    result = process_image(image, ret, mtx, dist, rvecs, tvecs, roi, M, Minv)
    cv2.imshow('test', result)
    cv2.waitKey(0)
    
#test()   
