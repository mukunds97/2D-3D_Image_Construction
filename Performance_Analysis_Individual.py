import cv2
import numpy as np
import pydicom
import time
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

filenamea = []
sift_kp1 = []
sift_kp2 = []
sift_match = []
sift_match_ran = []
sift_comp = []
sift_mat = []
surf_kp1 = []
surf_kp2 = []
surf_match = []
surf_match_ran = []
surf_comp = []
surf_mat = []
akaze_kp1 = []
akaze_kp2 = []
akaze_match = []
akaze_match_ran = []
akaze_comp = []
akaze_mat = []
orb_kp1 = []
orb_kp2 = []
orb_match = []
orb_match_ran = []
orb_comp = []
orb_mat = []

    # Loop through all DICOM images
for i in range(1, 10):
    # Load DICOM image
    filename = f'11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-{i:03d}.dcm'
    formatted_string = filename.split("/")[-1]  
    i_format = formatted_string.split("-")[-1].split(".")[0]
    ds = pydicom.dcmread(filename)


    sift = cv2.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()
    akaze = cv2.AKAZE_create()

    pixel_array = ds.pixel_array
    cv_image = np.array(pixel_array, dtype=np.uint8)
    img1 = cv2.convertScaleAbs(cv_image, alpha=(255.0/255.0))
    start_time = time.time()

    kp1sift, des1sift = sift.detectAndCompute(img1, None)
    kp1surf, des1surf = surf.detectAndCompute(img1, None)
    kp1orb, des1orb = orb.detectAndCompute(img1, None)
    kp1akaze, des1akaze = akaze.detectAndCompute(img1, None)

    for j in range(0,1):
        j = i+5
        # Load DICOM image
        filename2 = f'11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-{j:03d}.dcm'
        formatted_stringj = filename2.split("/")[-1]  
        j_format = formatted_stringj.split("-")[-1].split(".")[0]
        ds2 = pydicom.dcmread(filename2)

        # Extract the pixel array
        pixel_array2 = ds2.pixel_array

        # Convert the pixel array to an OpenCV image
        cv_image2 = np.array(pixel_array2, dtype=np.uint8)
        img2 = cv2.convertScaleAbs(cv_image2, alpha=(255.0/255.0))

        kp2sift, des2sift = sift.detectAndCompute(img2, None)
        kp2surf, des2surf = surf.detectAndCompute(img2, None)
        kp2orb, des2orb = orb.detectAndCompute(img2, None)
        kp2akaze, des2akaze = akaze.detectAndCompute(img2, None)

        end_time = time.time()
        comp_time = end_time - start_time

        match_start = time.time()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1sift,des2sift,k=2)
        match_end = time.time()
        match_time = match_end - match_start
        total_matches = len(matches)


        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        img3 = cv2.drawMatches(img1,kp1sift,img2,kp2sift,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # Apply RANSAC to find the best set of matches
        src_pts = np.float32([ kp1sift[m.queryIdx].pt for m in good ]).reshape(-1, 2)
        dst_pts = np.float32([ kp2sift[m.trainIdx].pt for m in good ]).reshape(-1, 2)

        #print(len(src_pts), len(dst_pts))
        best_src_pts = None
        best_dst_pts = None
        best_inliers = 0

        # Ransac
        model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform, min_samples=4,
                residual_threshold=8, max_trials=10000
            )

        n_inliers = np.sum(inliers)
        inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        # image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
        # plt.imshow(image3), plt.title('SIFT'), plt.show()

            # cv.imshow('Matches', image3)
            # cv.waitKey(0)

        src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
        dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

        conc = f"{i_format} - {j_format}"
        filenamea.append(conc)
        sift_kp1.append(len(kp1sift))
        sift_kp2.append(len(kp2sift))
        sift_match.append(len(good))    
        sift_match_ran.append(n_inliers)
        sift_comp.append(comp_time*1000)
        sift_mat.append(match_time*1000)
        #print(match_time)
        


        match_start = time.time()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1surf,des2surf,k=2)
        match_end = time.time()
        match_time = match_end - match_start
        total_matches = len(matches)


        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        img3 = cv2.drawMatches(img1,kp1surf,img2,kp2surf,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # Apply RANSAC to find the best set of matches
        src_pts = np.float32([ kp1surf[m.queryIdx].pt for m in good ]).reshape(-1, 2)
        dst_pts = np.float32([ kp2surf[m.trainIdx].pt for m in good ]).reshape(-1, 2)

        #print(len(src_pts), len(dst_pts))
        best_src_pts = None
        best_dst_pts = None
        best_inliers = 0

        # Ransac
        model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform, min_samples=4,
                residual_threshold=8, max_trials=10000
            )

        n_inliers = np.sum(inliers)
        inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
        plt.imshow(image3), plt.title('surf'), plt.show()

        # cv2.imshow('Matches - SURF', image3)
        # cv2.waitKey(0)

        src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
        dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

        conc = f"{i_format} - {j_format}"
        filenamea.append(conc)
        surf_kp1.append(len(kp1surf))
        surf_kp2.append(len(kp2surf))
        surf_match.append(len(good))    
        surf_match_ran.append(n_inliers)
        surf_comp.append(comp_time*1000)
        surf_mat.append(match_time*1000)
        #print(match_time)




        
        

        # Find the keypoints and descriptors for the images
        start = time.time()
        kp1orb, des1orb = orb.detectAndCompute(img1, None)
        kp2orb, des2orb = orb.detectAndCompute(img2, None)
        end = time.time()
        feat_time = end - start

        # Create Brute-Force Matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the descriptors
        start = time.time()
        matches = bf.match(des1orb, des2orb)
        # print('-'*10)
        # print(len(matches))
        end = time.time()
        comp_time = end - start

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # # Print the features in each image
        # print("Features in Image 1: ", len(kp1))
        # print("Features in Image 2: ", len(kp2))

        # # Print the number of matched features
        # print("No. matched features: ", len(matches))

        # Apply RANSAC to filter out outliers

        src_pts = np.float32([kp1orb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        dst_pts = np.float32([kp2orb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Select only the inliers

        matches_mask = mask.ravel().tolist()

        good_matches = [m for i, m in enumerate(matches) if matches_mask[i]]

        good = len(good_matches)
        # Print the number of matched features after removing outliers and finding the best match
        # print("No. of matched features after removing outliers and finding the best match: ", len(good_matches))

        # Draw the matches
        img3 = cv2.drawMatches(img1, kp1orb, img2, kp2orb, good_matches, None, flags=2)
        plt.title('ORB'), plt.imshow(img3), plt.show()

        #Display the matches
        cv2.imshow('Matches - ORB', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print feature description computation time
        # print("Feature description computation time (2 Images): ", feat_time)

        # # Print feature matching computation time
        # print("Feature matching computation time: ", match_time)
        filenamea.append(conc)
        orb_kp1.append(len(kp1orb))
        orb_kp2.append(len(kp2orb))
        orb_match.append(len(matches))    
        orb_match_ran.append(good)
        orb_comp.append(comp_time*1000)
        orb_mat.append(match_time*1000)


        match_start = time.time()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1akaze,des2akaze,k=2)
        match_end = time.time()
        match_time = match_end - match_start
        total_matches = len(matches)


        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        img3 = cv2.drawMatches(img1,kp1akaze,img2,kp2akaze,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # Apply RANSAC to find the best set of matches
        src_pts = np.float32([ kp1akaze[m.queryIdx].pt for m in good ]).reshape(-1, 2)
        dst_pts = np.float32([ kp2akaze[m.trainIdx].pt for m in good ]).reshape(-1, 2)

        ##print(len(src_pts), len(dst_pts))
        best_src_pts = None
        best_dst_pts = None
        best_inliers = 0

        # Ransac
        model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform, min_samples=4,
                residual_threshold=8, max_trials=10000
            )

        n_inliers = np.sum(inliers)
        inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        # image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
        # plt.imshow(image3), plt.title('akaze'), plt.show()

            # cv.imshow('Matches', image3)
            # cv.waitKey(0)

        src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
        dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

        conc = f"{i_format} - {j_format}"
        filenamea.append(conc)
        akaze_kp1.append(len(kp1akaze))
        akaze_kp2.append(len(kp2akaze))
        akaze_match.append(len(good))    
        akaze_match_ran.append(n_inliers)
        akaze_comp.append(comp_time*1000)
        akaze_mat.append(match_time*1000)
        #print(match_time)


#print('images :',filenamea)  
print("SIFT KP1", round(np.array(sift_kp1).mean()))
print("SIFT KP2", round(np.array(sift_kp2).mean()))
print("sift_match", round(np.array(sift_match).mean()))
print("sift_match_ater_ransac", round(np.array(sift_match_ran).mean()))
print("sift_comp_time", round(np.array(sift_comp).mean(),2))
print("sift_match_time", round(np.array(sift_mat).mean(),2))


#print('images :',filenamea)  
print("SURF KP1", round(np.array(surf_kp1).mean()))
print("SURF KP2", round(np.array(surf_kp2).mean()))
print("surf_match", round(np.array(surf_match).mean()))
print("surf_match_ater_ransac", round(np.array(surf_match_ran).mean()))
print("surf_comp_time", round(np.array(surf_comp).mean(),2))
print("surf_match_time", round(np.array(surf_mat).mean(),2))



#print('images :',filenamea)  
print("ORB KP1", round(np.array(orb_kp1).mean()))
print("ORB KP2", round(np.array(orb_kp2).mean()))
print("orb_match", round(np.array(orb_match).mean()))
print("orb_match_ater_ransac", round(np.array(orb_match_ran).mean()))
print("orb_comp_time", round(np.array(orb_comp).mean(),2))
print("orb_match_time", round(np.array(orb_mat).mean(),2))


#print('images :',filenamea)  
print("AKZE KP1", round(np.array(akaze_kp1).mean()))
print("AKZEKP2", round(np.array(akaze_kp2).mean()))
print("akaze_match", round(np.array(akaze_match).mean()))
print("akaze_match_ater_ransac", round(np.array(akaze_match_ran).mean()))
print("akaze_comp_time", round(np.array(akaze_comp).mean(),2))
print("akaze_match_time", round(np.array(akaze_mat).mean(),2))

