import numpy as np
import cv2

from multipoint.utils.homographies import warp_keypoints

def warp_keypoints_updated(kp, M):
    """
    Transforms given keypoints by homography M.

    Parameters: 
    - kp: (cv2.Keypoint object) N keypoints to transform 
    - M: (3x3 matrix) homography specifying keypoint transformation
    
    Returns: 
    - kp_warped: [N,2] array of transformed keypoint locations
    """

    # Copy kp points
    kp_warped = np.float32([ kp[i].pt for i in range(len(kp)) ])

    # Swap columns so points are [x,y,1]
    kp_warped = np.vstack((kp_warped[:,1],kp_warped[:,0])).T

    # Apply warping homography
    kp_warped = warp_keypoints(kp_warped, M)

    # Swap back to 
    kp_warped = np.vstack((kp_warped[:,1],kp_warped[:,0])).T

    return kp_warped


def compute_repeatability(kp0, kp1, H, threshold=4, mask=False):
    """
    Computes the ratio of interest points detected in both images to total number of interest points detected

    Parameters: 
    - kp0: keypoints from image0 (cv2.Keypoint object)
    - kp1: keypoints from image1 (cv2.Keypoint object) 
    - H: *groundtruth* homography that transforms points from image0 into image1
    - thresholdhold: distance (in pixels) below which interest points considered the same (i.e. redetected)
    - mask: (Boolean) whether or not to return mask (numpy array) of repeated vars (1 is repeated, 0 is not)
    """

    #------- WARP KEYPOINTS -------#
    # WARP KP0 -> 1
    kp0_warped = warp_keypoints_updated(kp0, H)
    # CONVERT KP0 to [N,2] array
    kp0_copy   = warp_keypoints_updated(kp0, np.eye(3))

    # WARP KP1 -> 0
    kp1_warped = warp_keypoints_updated(kp1, np.linalg.inv(H))
    # CONVERT KP1 to [N,2] array
    kp1_copy   = warp_keypoints_updated(kp1, np.eye(3))


    #------- CHECK REPEATED POINTS -------#
    repeated = 0
    kp0_mask = np.ones((kp0_copy.shape[0],1))
    kp1_mask = np.ones((kp1_copy.shape[0],1))

    # For each (warped) keypoint, check if any keypoint on the other image is < threshold distance away
    for idx,pt in enumerate(kp0_warped): 
        # Subtract pt from list of other points & take the norm to get distance to each point
        dist = np.linalg.norm(kp1_copy - pt, axis=1)
        # If distance is below (or equal to) threshold, count it as re-detected
        if min(dist) <= threshold: repeated += 1; kp0_mask[idx] = 1
        # If not, only update the "mask"
        else: kp0_mask[idx] = 0
    
    # Repeat for "other direction"
    for idx,pt in enumerate(kp1_warped): 
        dist = np.linalg.norm(kp0_copy - pt, axis=1)
        if min(dist) <= threshold: repeated += 1; kp1_mask[idx] = 1
        else: kp1_mask[idx] = 0

    # Total Number of keypoints 
    total_points = (len(kp0) + len(kp1))

    # Return 
    if not mask:
        return repeated,total_points
    else:
        return repeated,total_points, kp0_mask, kp1_mask

def compute_m_score(kp0, kp1, matches, H, threshold=4, mask=False): 

    # Copy kp0 and kp1 (change to [N,2] array)
    kp0_copy   = warp_keypoints_updated(kp0, np.eye(3))
    kp1_copy   = warp_keypoints_updated(kp1, np.eye(3))
   
    # Warp kp1 back to img0 & vice versa
    kp0_warped = warp_keypoints_updated(kp0, H)
    kp1_warped = warp_keypoints_updated(kp1, np.linalg.inv(H))

    # Initialize ordered points
    kp0_copy_ordered   = np.empty([len(matches),2])
    kp1_copy_ordered   = np.empty([len(matches),2])
    kp0_warped_ordered = np.empty([len(matches),2])
    kp1_warped_ordered = np.empty([len(matches),2])

    # Get all the points in order
    for idx, m in enumerate(matches):
        kp0_copy_ordered[idx,:]   = kp0_copy[m.queryIdx,:]
        kp1_copy_ordered[idx,:]   = kp1_copy[m.trainIdx,:]
        kp0_warped_ordered[idx,:] = kp0_warped[m.queryIdx,:]
        kp1_warped_ordered[idx,:] = kp1_warped[m.trainIdx,:]

    # Get the euclidean distance
    img0_match_dist = np.linalg.norm(kp0_copy_ordered - kp1_warped_ordered, axis=1)
    img1_match_dist = np.linalg.norm(kp1_copy_ordered - kp0_warped_ordered, axis=1)

    # Compare all match distances to the threshold
    img0_match_mask = (img0_match_dist < threshold).astype(int)
    img1_match_mask = (img1_match_dist < threshold).astype(int)

    # Get correct matches
    matches_groundtruth_mask = (img0_match_mask | img1_match_mask).astype(int)

    # Take the avg
    good_matches = sum(matches_groundtruth_mask) 

    # Also return the total number of matches
    total_matches = len(matches)

    if not mask: 
        return good_matches, total_matches
    else:
        return good_matches, total_matches, matches_groundtruth_mask

def homography_estimation_success(H_est, H_groundtruth, img, threshold):
    # Get the dimensions of the original image
    H = img.shape[0]
    W = img.shape[1]

    # Create "keypoints" at image edges
    kp = np.array([ [0,0], [0, W-1], [H-1, 0], [H-1,W-1]])

    # Compute the estimated INVERSE mapping
    H_est_inv = np.linalg.inv(H_est)

    # Apply the warping homography (H_groundtruth), then the estimated inverse mapping (H_est_inv)
    # If H_est is 100% accurace, the keypoints will be the same before and after transformation
    kp_est = warp_keypoints(kp, np.matmul(H_est_inv,H_groundtruth))

    # Get distance between keypoints 
    kp_dist = np.linalg.norm(kp - kp_est, axis = 1)

    # Check which distances are below the threshold
    kp_pass = (kp_dist < threshold).astype(int)

    # If ALL 4 corners are within the threshold, it passes
    return sum(kp_pass) == 4
