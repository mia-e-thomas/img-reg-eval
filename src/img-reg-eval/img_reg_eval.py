# General
import argparse
import yaml
import cv2
import numpy as np

# Multipoint
from multipoint.utils.homographies import sample_homography

# Custom
import ImageLoader
import ClassicalFeatures
import Matcher
import Evaluation

def main():

    #------------#
    # Parameters #
    #------------#
    # Define script arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--yaml-config',    default='config/config.yaml', help='YAML config file')
    # parser.add_argument('-r', '--results-folder', default='results/',           help='Relative path to results folder')
    parser.add_argument('-d',  dest='display',    action='store_true',          help='If set, the images/plots are displayed (numerical results always displayed)')
    parser.add_argument('-s',  dest='save',       action='store_true',          help='If set, the images/plots are saved (numerical results always saved)')

    # Get arguments
    args = parser.parse_args()

    # Get YAML file
    with open(args.yaml_config, 'r') as f_yaml:
        config_yaml = yaml.load(f_yaml, Loader=yaml.FullLoader)

    # If scientific_notation parameter = False, suppress scientific notation
    np.set_printoptions(suppress= ~config_yaml['dataset']['scientific_notation'])

    #--------------#
    # Load Dataset #
    #--------------#
    #------- GET DATASET -------#
    test_dataset = ImageLoader.H5ImageLoader(config_yaml['dataset']['filepath'])

    #------- GET IMAGE PAIR -------#
    # Get Image Pair specified in 'index' parameter
    # Also, use 'subindex' parameter in case > 2 images are in a set
    img_pair = test_dataset.getImgPair(index = config_yaml['dataset']['index'], subindex = config_yaml['dataset']['subindex'])
    # Convert image pair to RGB (& 0-255)
    img_pair = [ImageLoader.cvtToRGB(img_pair[0]), ImageLoader.cvtToRGB(img_pair[1])]
    # Get the image types
    img_types = [test_dataset.img_types[config_yaml['dataset']['subindex'][0]],test_dataset.img_types[config_yaml['dataset']['subindex'][1]]]
    # Get the timestamp
    img_ts = test_dataset.timestamps[config_yaml['dataset']['index']]

    #------- FLIP -------#
    # RANDOMLY choose to flip images 
    # (so we're not always warping one type of image)
    if config_yaml['dataset']['flip']:
        flip = np.random.randint(low=0,high=2,size=1)
        if flip: img_pair = [img_pair[1], img_pair[0]]; img_types = [img_types[1], img_types[0]]

    #------- DISPLAY -------#
    display_str = "Aligned " + img_types[0] + " and " + img_types[1] + " (" + img_ts + ")"
    # Show image pair (if parameter set)
    if args.display:
        cv2.imshow(display_str, np.hstack((img_pair[0],img_pair[1])))
        cv2.waitKey()
        cv2.destroyWindow(display_str)
    
    #------- SAVE -------#
    save_str = "Aligned-" + img_types[0] + "-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE


    #------- BLENDED THERMAL & OPTICAL IMAGE -------#
    # Make original green
    img_blend_0 = np.copy(img_pair[0]); img_blend_0[:,:,0] = 0; img_blend_0[:,:,2] = 0

    # Make registered purple
    img_blend_1 = np.copy(img_pair[1]); img_blend_1[:,:,1] = 0; 

    # Blend
    img_blend_tir_vis = cv2.addWeighted(img_blend_0, 0.5, img_blend_1, 0.5, 0)

    if args.display:
        display_str = "Blended "+img_types[0]+" (green) and "+img_types[1]+" (purple) images (" + img_ts + ")"
        cv2.imshow(display_str, img_blend_tir_vis)
        cv2.waitKey()
        cv2.destroyWindow(display_str)
 
    # Save
    save_str = "blended-"+img_types[0]+"-"+img_types[1]+"-"+img_ts 
    # TODO: ADD SAVE

    #----------------#
    # Transformation # 
    #----------------#


    
    #---------#
    # Warping # 
    #---------#

    #------- SAMPLE HOMOGRAPHY -------#
    # Sample homography
    img_shape = (img_pair[0].shape[0],img_pair[0].shape[1])
    M_warp = sample_homography(img_shape)

    #------- DISPLAY -------#
    # Print result
    print("Warp Homography:\n"+ str(M_warp))

    #------- WARPING -------#
    # Warp Image
    img_warped = cv2.warpPerspective(img_pair[1], M_warp, img_pair[1].T.shape[1:3])

    #------- DISPLAY -------#
    display_str = "Normal " + img_types[0] + " and warped " + img_types[1] + " (" + img_ts + ")"

    if args.display:
        cv2.imshow(display_str, np.hstack((img_pair[0],img_warped)))
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    #------- SAVE -------#
    save_str = "Normal-" + img_types[0] + "-warped-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE

    #----------#
    # Matching # 
    #----------#

    #------- DETECT & COMPUTE -------#
    # Feature object
    feature = ClassicalFeatures.Feature(config_yaml['features'])
    
    # Detect and describe (& change output to float32)
    kp0, des0 = feature.method.detectAndCompute(img_pair[0],None); 
    kp1, des1 = feature.method.detectAndCompute(img_warped,None); 
    
    #------- MATCH -------#
    # Matcher object
    matcher = Matcher.Matcher(config_yaml['matching'])

    # Compute matches
    matches = matcher.match(des0,des1)

    # Sort matches in order of distance
    #matches = sorted(matches, key = lambda x:x.distance)

    #------- DISPLAY MATCH -------#
    # Draw matches
    matches_draw = cv2.drawMatches(
                        img_pair[0], kp0,
                        img_warped,  kp1,
                        matches,None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )

    # Show images
    display_str = "Matches: normal " + img_types[0] + " and warped " + img_types[1] + " (" + img_ts + ")"

    if args.display:
        cv2.imshow(display_str, matches_draw)
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    #------- SAVE -------#
    save_str = "matches-normal-" + img_types[0] + "-warped-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE


    #------------#
    # HOMOGRAPHY # 
    #------------#

    #------- CHECK # MATCHES -------#
    # Check number of matches
    if len(matches) < config_yaml['features']['min_matches']: raise RuntimeError('Insufficient matches')

    #------- ESTIMATE HOMOGRAPHY -------#
    # Convert Point representation
    src_pts = np.float32([ kp0[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    # Estimate the homography (potentialy w/ ransac)
    # W/ ransac
    if config_yaml['matching']['ransac']:
        M_warp_est, matches_ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC) 
    # W/o ransac
    else:
        M_warp_est, matches_ransac_mask = cv2.findHomography(src_pts, dst_pts)

    #------- DISPLAY RESULT -------#
    # Print result
    print("Estimated Homography: \n" + str(M_warp_est))
    

    #------- REDRAW INLIERS -------#

    # Inliers = Green, Outliers = Red
    # 1. Draw all matches in red
    matches_draw_inliers = cv2.drawMatches(img_pair[0], kp0, img_warped, kp1,
                                matches, None,
                                matchColor = (0,0,255),   # draw matches in red color
                                )
    # 2. Draw inliers in green 
    w = int(matches_draw.shape[1]/2)
    matches_draw_inliers = cv2.drawMatches(matches_draw_inliers[:,:w,:],kp0,matches_draw_inliers[:,w:,:],kp1,
                                matches, None,
                                matchColor = (0,255,0),   # draw matches in green color
                                matchesMask = matches_ransac_mask[:,0],  # draw only inliers
                                )

    # Show images
    display_str = "RANSAC Inlier Matches: normal " + img_types[0] + " and warped " + img_types[1] + " (" + img_ts + ")"
    if args.display:
        cv2.imshow(display_str, matches_draw_inliers)
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    #------- SAVE -------#
    save_str = "ransac-matches-normal-" + img_types[0] + "-warped-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE


    #-----------#
    # Unwarping # 
    #-----------#

    #------- COMPUTE INVERSE WARPING -------#
    M_unwarp_est = np.linalg.inv(M_warp_est)


    #------- UNWARP IMAGE -------#
    # New method applies the forward and inverse homographies to get "unwarped"
    img_unwarped = cv2.warpPerspective(img_pair[1], np.matmul(M_unwarp_est,M_warp), dsize = img_warped.T.shape[1:3])
    # Old method unwarps "warped" image, but that includes the mask. 
    #img_unwarped = cv2.warpPerspective(img_warped, M_unwarp_est, dsize = img_warped.T.shape[1:3])

    # Show
    display_str = "Normal " + img_types[0] + " and registered " + img_types[1] + " (" + img_ts + ")"
    if args.display:
        cv2.imshow(display_str, np.hstack((img_pair[0],img_unwarped)))
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    #------- SAVE -------#
    save_str = "Normal-" + img_types[0] + "-registered-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE

    #--------------------#
    # Error Calculations # 
    #--------------------#

    #------- NUMBER OF KEYPOINTS -------#
    N_kp = np.rint(0.5*(len(kp0) + len(kp1))).astype(int)

    # Print result
    print("N_kp: " + str(N_kp))

    #------- REPEATABILITY -------#
    repeat_threshold = config_yaml['evaluation']['repeatability']['threshold']
    repeated_points, total_points, kp0_mask, kp1_mask = Evaluation.compute_repeatability(
                                                            kp0, kp1, M_warp, 
                                                            threshold = repeat_threshold, 
                                                            mask=True)

    # Print result
    print("Repeatability: " + str(repeated_points/total_points))

    #------- M SCORE -------#
    match_threshold = config_yaml['evaluation']['m_score']['threshold']
    correct_matches, total_matches, matches_groundtruth_mask =  Evaluation.compute_m_score(
                                                            kp0, kp1, matches, 
                                                            M_warp, 
                                                            threshold=match_threshold, 
                                                            mask=True)
    
    # Print result
    print("M Score: " + str(correct_matches/total_matches))

    
    #------- HOMOGRAPHY ERROR -------#
    # Get the list of error thresholds
    homography_thresholds = config_yaml['evaluation']['homography']['thresholds']

    # Set up empty array to store results
    homography_success = np.empty( ( len(homography_thresholds) , ) )

    # Compute if homography successful for each threshold
    for idx in range(len(homography_thresholds)):
        homography_success[idx] = Evaluation.homography_estimation_success(
                                    M_warp_est, M_warp, img_pair[1], 
                                    threshold = homography_thresholds[idx]
                                    ).astype(int)

    # Print results
    display_str = "Homography Success for thresholds " + str(homography_thresholds) + ": "
    print(display_str + str(homography_success.astype(bool)))

    # TODO: Add additional stats for homography error
    #   - Go beyond what multipoint does (will need new function)
    #   - Give the avg & highest corner error
    #   - Error on all points? RMSE? Median?

    #------------------#
    # Visualize Errors # 
    #------------------#

    #------- WARP KEYPOINTS -------#
    # WARP KP0 -> 1
    kp0_warped = Evaluation.warp_keypoints_updated(kp0, M_warp)
    # CONVERT KP0 to [N,2] array
    kp0_copy   = Evaluation.warp_keypoints_updated(kp0, np.eye(3))

    # WARP KP1 -> 0
    kp1_warped = Evaluation.warp_keypoints_updated(kp1, np.linalg.inv(M_warp))
    # CONVERT KP1 to [N,2] array
    kp1_copy   = Evaluation.warp_keypoints_updated(kp1, np.eye(3))

    # Repeated points
    kp0_copy_repeated = kp0_copy*kp0_mask
    kp1_copy_repeated = kp1_copy*kp1_mask
    kp0_warped_repeated = kp0_warped*kp0_mask
    kp1_warped_repeated = kp1_warped*kp1_mask

    #------- REPEATED KEYPOINTS -------#

    #------- IMG 0 -------#
    # Make a copy of the image
    img0_keypoints = np.copy(img_pair[0])

    # Display kp0 (blue) & kp1_warped (red)
    for pt in kp0_copy:            cv2.circle(img0_keypoints, pt.astype(int), 1, (255,0,0), 1)
    for pt in kp1_warped:          cv2.circle(img0_keypoints, pt.astype(int), 1, (0,0,255), 1)
    for pt in kp0_copy_repeated:   cv2.circle(img0_keypoints, pt.astype(int), 1, (0,255,0), 1)
    for pt in kp1_warped_repeated: cv2.circle(img0_keypoints, pt.astype(int), 1, (0,255,0), 1)

    # Legend
    cv2.putText(img=img0_keypoints, text='Non-redetected original keypoints',        org=(5, 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255, 0, 0),thickness=1)
    cv2.putText(img=img0_keypoints, text='Non-redetected warped keypoints',          org=(5, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(0, 0, 255),thickness=1)
    cv2.putText(img=img0_keypoints, text='Redetected keypoints (original & warped)', org=(5, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(0, 255, 0),thickness=1)

    # Display img
    if args.display:
        display_str = "Keypoints: " + img_types[0] + " (" + img_ts + ")"
        cv2.imshow(display_str, img0_keypoints)
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    # Save
    save_str = img_types[0] + "-keypoints-" + img_ts
    # TODO: ADD SAVE

    #------- IMG 1 -------#
    # Make a copy of the image
    img1_keypoints = np.copy(img_warped)

    # Display kp1 (blue) & kp0_warped (red)
    for pt in kp1_copy:            cv2.circle(img1_keypoints, pt.astype(int), 1, (255,0,0), 1)
    for pt in kp0_warped:          cv2.circle(img1_keypoints, pt.astype(int), 1, (0,0,255), 1) 
    for pt in kp1_copy_repeated:   cv2.circle(img1_keypoints, pt.astype(int), 1, (0,255,0), 1)
    for pt in kp0_warped_repeated: cv2.circle(img1_keypoints, pt.astype(int), 1, (0,255,0), 1)
    
    # Legend
    cv2.putText(img=img1_keypoints, text='Non-redetected original keypoints',        org=(5, 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255, 0, 0),thickness=1)
    cv2.putText(img=img1_keypoints, text='Non-redetected warped keypoints',          org=(5, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(0, 0, 255),thickness=1)
    cv2.putText(img=img1_keypoints, text='Redetected keypoints (original & warped)', org=(5, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(0, 255, 0),thickness=1)

    # Display img
    if args.display:
        display_str = "Keypoints: " + img_types[1] + " (" + img_ts + ")"
        cv2.imshow(display_str, img1_keypoints)
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    # Save
    save_str = img_types[1] + "-keypoints-" + img_ts 
    # TODO: ADD SAVE

    

    #------- DRAW TRUE MATCHES -------#
    # Inliers = Green, Outliers = Red
    # 1. Draw all matches in red
    matches_draw_groundtruth = cv2.drawMatches(img_pair[0], kp0, img_warped, kp1,
                                matches, None,
                                matchColor = (0,0,255),   # draw matches in red color
                                )
    # 2. Draw inliers in green 
    w = int(matches_draw.shape[1]/2)
    matches_draw_groundtruth = cv2.drawMatches(matches_draw_groundtruth[:,:w,:],kp0,matches_draw_groundtruth[:,w:,:],kp1,
                                matches, None,
                                matchColor = (0,255,0),   # draw matches in green color
                                matchesMask = matches_groundtruth_mask,  # draw only inliers
                                )

    # Display img
    if args.display:
        display_str = "Groundtruth Inlier Matches: normal " + img_types[0] + " and warped " + img_types[1] + " (" + img_ts + ")"
        cv2.imshow(display_str, matches_draw_groundtruth)
        cv2.waitKey()
        cv2.destroyWindow(display_str)

    # Save
    save_str = "groundtruth-matches-normal-" + img_types[0] + "-warped-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE

    #------- BLENDED ORIGINAL & REGISTERED IMAGE -------#
    # Make original green
    img_blend_orig = np.copy(img_pair[1]); img_blend_orig[:,:,0] = 0; img_blend_orig[:,:,2] = 0

    # Make registered purple
    img_blend_registered = np.copy(img_unwarped); img_blend_registered[:,:,1] = 0; 

    # Blend
    img_blend_warped_unwarped = cv2.addWeighted(img_blend_orig, 0.5, img_blend_registered, 0.5, 0)

    if args.display:
        display_str = "Blended original (green) and registered (purple) " + img_types[1] + " images (" + img_ts + ")"
        cv2.imshow(display_str, img_blend_warped_unwarped)
        cv2.waitKey()
        cv2.destroyWindow(display_str)
 
    # Save
    save_str = "blended-original-registered-" + img_types[1] + "-" + img_ts 
    # TODO: ADD SAVE
   
   


if __name__ == "__main__":
    main()



