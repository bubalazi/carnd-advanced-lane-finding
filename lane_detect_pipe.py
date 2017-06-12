#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Computer Vision based lane detection

In this project, the radius of curvature of the road ahead and the
vehicle offset with respect to lane-centre position are infered, using a
histogram based sliding window approach, applied to infer probability of lane
marking pixels, followed by a second degree polynomial fit of the found left and
 right lane markings in a perspectively corrected top-down (bird's eye) view.
"""
__author__ = "Lyuboslav Petrov"

import os
import sys
import glob
import cv2
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from Queue import Queue

# =================================Helpers=====================================#
def catch_user_interrupt(wait_time=1):
    """
    OpenCV specific high-gui window interruption handler
    """
    ch = 0xFF & cv2.waitKey(wait_time)
    return True if ch == 27 else False

def draw_str(img, (x, y), s, scale=0.3):

    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255),
                lineType=cv2.LINE_AA)
    cv2.putText(img, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0),
                thickness=1, lineType=cv2.LINE_AA)
# =============================================================================#

# ===============================Business Logic================================#
def get_calib_data(path, nx=9, ny=5):
    """
    Computes camera calibration data given the path to the images folder

    Parameters
    ----------
    path : string
        Path to calibration images folder
    nx : int
        Number of inner edges along x dimension
    ny : int
        Number of inner edges along y dimension

    Returns
    -------
    calibration_data: dict
        Dictionary of all infered data

    """

    # Make a list of calibration images
    imgs = glob.glob(os.path.join(path, '*.jpg'))

    objpoints = []
    imgpoints = []
    imgnames = []

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for impath in imgs:
        imgnames.append(os.path.split(impath)[-1])
        img = cv2.imread(impath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
#             cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
    #         plt.imshow(img)

    # Compute distortion corrections
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1],None,None)

    cdata = {
        "imgnames": imgnames,
        "objpoints": objpoints,
        "imgpoints": imgpoints,
        "results":{
            "ret": ret,
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs
        }
    }

    # Save the data at the same location
    with open(os.path.join(path, 'cdata.p'),  'w') as fid:
        pickle.dump(cdata, fid)

    return cdata

def get_perspective_matrix(src=[], dst=[]):
    """
    Finds the perspective distortion matrix and its inverse

    Parameters
    ----------
    src : ndarray
        Array carying the four source points
    dst : ndarray
        Array for the destination (unwarped) points

    Returns
    -------
    M : ndarray
        The perspective distortion matrix
    M_INV : ndarray
        The inverse perspective distortion matrix
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_INV = cv2.getPerspectiveTransform(dst, src)
    return M, M_INV

def warp(img, M):
    """
    Warp an image given a perspective distortion matrix

    Parameters
    ----------
    img : ndarray
        Input image to warp
    M : ndarray
        Distortion coefficient matrix

    Returns
    -------
    warped : ndarray
        The warped `img`
    """
    (height, width) = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped

def binarize(img, s_thresh=(170, 255), edge_thresh=(20, 100)):
    """
    Combined color and edge based masking

    Parameters
    ----------
    img : ndarray
        Input image
    s_thresh : tuple
        The lower and upper thresholds for the S channel
    edge_thresh : tuple
        Lower and upper thresholds for the sobel filter on the L channel

    Returns
    -------
    binary : ndarray
        The binirized combined mask
    """

    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    H,L,S = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float))
    # Sobel x
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= edge_thresh[0]) & (scaled_sobel <= edge_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1

    sxbinary = sxbinary.astype(np.float) - sxbinary.astype(np.float).min()
    sxbinary /= sxbinary.max()
    s_binary = s_binary.astype(np.float) - s_binary.astype(np.float).min()
    s_binary /= s_binary.max()

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    binary = color_binary.sum(axis=2).astype(np.uint8) * 255
    return binary

def initial_line_fit(binary_warped, nwindows=9, margin=100, minpix=50, polyorder=2):
    """
    Initial sliding window based lane finding

    Parameters
    ----------
    binary_warped : ndarray
        Input binary image
    nwindows : int
        Number of sliding windows
    margin : int
        Width of sliding windows +/- margin
    minpix : int
        Minimum number of pixels found to recenter window
    polyorder : int
        Order of polynomial for line fitting

    Returns
    -------


    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2.0)
    leftx_base = np.argmax(histogram[margin:midpoint]) + margin
    rightx_base = np.argmax(histogram[midpoint:-margin]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/float(nwindows))
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(
            out_img,
            (win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),
            (0,255,0), 2)
        cv2.rectangle(
            out_img,
            (win_xright_low,win_y_low),
            (win_xright_high,win_y_high),
            (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low) & \
            (nonzeroy < win_y_high) & \
            (nonzerox >= win_xleft_low) & \
            (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & \
            (nonzeroy < win_y_high) & \
            (nonzerox >= win_xright_low) & \
            (nonzerox < win_xright_high)
        ).nonzero()[0]
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
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, polyorder)
    right_fit = np.polyfit(righty, rightx, polyorder)

    return {
    'status': True,
    'left_fit': left_fit,
    'right_fit':right_fit,
    'nonzerox': nonzerox,
    'nonzeroy': nonzeroy,
    'out_img': out_img,
    'left_lane_inds': left_lane_inds,
    'right_lane_inds': right_lane_inds
    }

def fast_fit(binary_warped, left_fit, right_fit, margin=100, min_pts=10):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    height, width = binary_warped.shape[:2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_line = np.polyval(left_fit, nonzeroy)
    left_lane_inds = (
        (nonzerox > left_line - margin) & \
        (nonzerox < left_line + margin)
    )
    right_line = np.polyval(right_fit, nonzeroy)
    right_lane_inds = (
        (nonzerox > right_line - margin) & \
        (nonzerox < right_line + margin)
    )

    # Again, extract left and right line pixel positions
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # If we don't find enough relevant points, return all None (this means error)
    if lefty.shape[0] + righty.shape[0] < (2 * min_pts):
        return {'status': False}

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, height-1, height)
    left_fitx, right_fitx = np.polyval(left_fit, ploty), np.polyval(right_fit, ploty)

    return {
    'status': True,
    'left_fit': left_fit,
    'right_fit':right_fit,
    'nonzerox': nonzerox,
    'nonzeroy': nonzeroy,
    'out_img': None,
    'left_lane_inds': left_lane_inds,
    'right_lane_inds': right_lane_inds
    }

def get_vehicle_offset(img, left_fit, right_fit, xmppx=3.7/700):
    (height, width) = img.shape[:2]
    left_line = np.polyval(left_fit, height-1)
    right_line = np.polyval(right_fit, height-1)
    vehicle_offset = width / 2.0 - (left_line + right_line)/2
    return vehicle_offset * xmppx

def compute_curvature(line_fit_dict, xm_per_pix=3.7/700, ym_per_pix=30.0/720):
    """
    Compute radius of curvature

    Parameters
    ----------
    xm_per_pix : float
        Meters per pixel in x dimension
    ym_per_pix : float
        Meters per pixel in y dimension

    Returns
    -------
    """
    y_eval = 719 * ym_per_pix

    # Extract left and right line pixel positions
    leftx = line_fit_dict["nonzerox"][line_fit_dict["left_lane_inds"]]
    lefty = line_fit_dict["nonzeroy"][line_fit_dict["left_lane_inds"]]
    rightx = line_fit_dict["nonzerox"][line_fit_dict["right_lane_inds"]]
    righty = line_fit_dict["nonzeroy"][line_fit_dict["right_lane_inds"]]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    A2_l, A2_r = 2 * left_fit_cr[0], 2 * right_fit_cr[0]
    B_l, B_r = left_fit_cr[1], right_fit_cr[1]

    left_curverad = ((1 + (A2_l * y_eval + B_l)**2)**1.5) / np.absolute(A2_l)
    right_curverad = ((1 + (A2_r * y_eval + B_r)**2)**1.5) / np.absolute(A2_r)
    # Now our radius of curvature is in meters

    return (left_curverad, right_curverad)

def visualize_results(undist, line_fit_dict, M_INV, curvatures, vehicle_offset):
    """
    Final lane line prediction visualized and overlayed on top of original image
    """
    height, width = undist.shape[:2]
    # Generate x and y values for plotting
    ploty = np.linspace(0, height-1, height)
    left_fitx = np.polyval(line_fit_dict['left_fit'], ploty)
    right_fitx = np.polyval(line_fit_dict['right_fit'], ploty)

    color_warp = np.zeros_like(undist)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = cv2.warpPerspective(color_warp, M_INV, (width, height))
    # Combine the result with the original image
    output = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)

    # Annotate lane curvature values and vehicle offset from center
    avg_curve = (curvatures[0] + curvatures[1]) / 2.0
    _str = 'Mean Curvature: %.1f m' % avg_curve
    draw_str(output, (20,30), _str, scale=1.0)
    _str = 'Vehicle offset: %.1f m' % vehicle_offset
    draw_str(output, (20,60), _str, scale=1.0)

    return output

def pipe(img, cdata, M, M_INV, line_fit_dict={'status': False}):
    """
    Combine all procedures for lane detection

    Parameters
    ----------
    img : ndarray
        Input image (presumably of the road)
    cdata : dict
        A dictionary of camera calibration coefficients
    M : ndarray
        Perspective distortion coefficients
    M_INV : ndarray
        The inverse perspective distortion coefficients
    line_fit_dict : dict
        Previos output dictionary. Used to determine if a complete step or must
        be made or only a fast fit, using previos measurements

    Returns
    -------
    line_fit_dict : dict
        Dictionary outputs
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Lens Distortion Correction
    undist = cv2.undistort(img, cdata['results']['mtx'],
        cdata['results']['dist'], None, cdata['results']['mtx'])

    # Step 2: Combined Binary Masking
    binarized = binarize(undist)

    # Step 3: Perspective Distortion Correction
    warped = warp(binarized, M)

    # Step 4: Histogram based sliding window line fit
    if line_fit_dict['status']:
        # If previos step was successful
        line_fit_dict = fast_fit(warped,
                                line_fit_dict['left_fit'],
                                line_fit_dict['right_fit'])
    else:
        # Else, perform initial computation
        line_fit_dict = initial_line_fit(warped)

    return line_fit_dict, undist

def parser():
    parser = argparse.ArgumentParser(description="""
    Pure computer vision implementation for road curvature estimation
    """)

    parser.add_argument(
        '-i',
        type=str,
        default=None,
        dest='image',
        help="""Input image
        """
    )

    parser.add_argument(
        '-v',
        type=str,
        default=None,
        dest='input',
        help="""Input video
        """
    )

    parser.add_argument(
        '-o',
        type=str,
        default=None,
        dest='output',
        help="""Output video
        """
    )

    parser.add_argument(
        '-c',
        type=str,
        default='camera_cal/cdata.p',
        dest='cdata',
        help="""Camera Calibration data
        """
    )

    parser.add_argument(
        '-f',
        type=int,
        default=0,
        dest='frame',
        help="""Start Video from frame number `-f`
        """
    )

    args = parser.parse_args()
    return args

def movav(Q, data, stds=6, axis_thresh=1):
    """
    Moving average of a queue with outlier rejection

    Outliers  lying `stds` standard  deviations away from the mean of the queue
    are discarded.

    Parameters
    ----------
    Q : Queue.Queue object
        A Queue object
    data : ndarray
        A queue item candidate
    stds : int
        Number of standard deviations for rejection threshold
    axis_thresh : float
        Threshold for number of correct axes of `data` (within the `stds` standard
        deviations) in order to at an element to the queue `Q`

    Returns
    -------
    movav : ndarray
        The average of the filtered queue
    """
    if Q.full():
        Q.get()

    if len(Q.queue) < (Q.maxsize):
        Q.put(data)
    else:
        current_mean = np.mean(Q.queue, axis=0)
        current_std = np.std(Q.queue, axis=0)
        cond = np.abs(data - current_mean) < (stds * current_std)
        if cond.sum() >= axis_thresh:
            print("{}: Added new frame".format(time.time()))
            Q.put(data)
        else:
            print("{}:Discarded outlier frame".format(time.time()))

    return np.mean(Q.queue, axis=0)

def main():

    # Queues for holding lane estimate values
    MAX_Q_SIZE = 12
    Q_left = Queue(maxsize=MAX_Q_SIZE)
    Q_right = Queue(maxsize=MAX_Q_SIZE)

    args = parser()

    if args.cdata:
        try:
            with open(args.cdata, 'r') as fid:
                cdata = pickle.load(fid)
        except:
            print("Calibration data file not found or invalid.")
            return 1
    else:
        print("No precomputed calibration data given.")

        ans  = raw_input('Do you want to compute calibration data now (N/y): ')
        if not 'y' in ans.lower():
            return 1

        cdata = get_calib_data('camera_cal')

    src = np.float32([
        [595, 450],
        [685, 450],
        [1100, 720],
        [200, 720]])
    dst = np.float32([
        [300, 0],
        [980, 0],
        [980, 720],
        [300, 720]])

    M, M_INV = get_perspective_matrix(src=src, dst=dst)

    line_fit_dict = {'status': False}

    if args.image is not None:
        try:
            img = cv2.imread(args.image)
        except:
            print("Image not found or invalid")
            return 1

        lane_fit_dict, undist = pipe(img, cdata, M, M_INV,
            line_fit_dict=line_fit_dict)
        if line_fit_dict['status']:
            vehicle_offset = get_vehicle_offset(undist,
                line_fit_dict['left_fit'], line_fit_dict['right_fit'])
            curvatures = compute_curvature(line_fit_dict)
            img = visualize_results(undist, line_fit_dict, M_INV, curvatures,
                vehicle_offset)
            plt.imshow(drawn)
            plt.show()
        return 0

    cap = cv2.VideoCapture(args.input)
    if args.frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if args.output is not None:
        if os.path.exists(args.output):
            ans  = raw_input('Video exists. Do you want to override it (N/y): ')
            if not 'y' in ans.lower():
                return 1
        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        )
    else:
        writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        line_fit_dict, undist = pipe(frame, cdata, M, M_INV,
            line_fit_dict=line_fit_dict)

        if line_fit_dict['status']:
            # Average out the line estimations over `MAX_Q_SIZE` measurements
            left, right = line_fit_dict['left_fit'], line_fit_dict['right_fit']
            left, right = movav(Q_left, left), movav(Q_right, right)
            line_fit_dict['left_fit'], line_fit_dict['right_fit'] = left, right
            # Now compute offset and curvatures based on more stable readings
            vehicle_offset = get_vehicle_offset(undist,
                line_fit_dict['left_fit'], line_fit_dict['right_fit'])
            curvatures = compute_curvature(line_fit_dict)
            analyzed = visualize_results(undist, line_fit_dict, M_INV, curvatures,
                vehicle_offset)
        else:
            analyzed  = frame
        _str = "Frame: {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES))
        draw_str(analyzed, (20,90), _str, scale=1.0)
        analyzed = cv2.cvtColor(analyzed, cv2.COLOR_RGB2BGR)
        if writer:
            writer.write(analyzed)
        cv2.imshow('video', analyzed)
        if catch_user_interrupt():
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    return 0
# ============================================================================#

if __name__ == "__main__":
    sys.exit( main() )
