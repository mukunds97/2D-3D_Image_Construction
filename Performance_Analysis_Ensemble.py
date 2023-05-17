import cv2
import numpy as np
import pydicom
import time
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

algorithms = ['ORB_create', 'xfeatures2d.SURF_create','SIFT_create', 'ORB_create', 'AKAZE_create', 'xfeatures2d.SURF_create']
algo_res = []
# results = []

for a, b in itertools.combinations(algorithms, 2):
    if a == 'xfeatures2d.SURF_create':
        sift = cv2.xfeatures2d.SURF_create()
    else:
        sift = cv2.__getattribute__(a)()

    if b == 'xfeatures2d.SURF_create':
        orb = cv2.xfeatures2d.SURF_create()
    else: 
        orb = cv2.__getattribute__(b)()
    print(f"Running {a} and {b}")
    # print(f"Running {a} and {b}")
    # print(a)
    # print(b)
    algo_res.append(a.split('_')[0] + " + " + b.split('_')[0])
    


# sift = cv2.SIFT_create()
# orb = cv2.AKAZE_create()
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    filenamea = []
    kp1ca = []
    kp2ca = []
    matchesa1 = []
    matchesa2 = []
    good_matchesa = []
    match_timea = []
    comp_timea = []
    columns = ['KP1', 'KP2', 'Matches', 'Good Matches', 'Comp Time', 'Match Time']
    df = pd.DataFrame(columns=columns)

    # Loop through all DICOM images
    for i in range(1, 10):
        # Load DICOM image
        filename = f'11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-{i:03d}.dcm'
        formatted_string = filename.split("/")[-1]  
        i_format = formatted_string.split("-")[-1].split(".")[0]
        ds = pydicom.dcmread(filename)

        # Extract the pixel array
        pixel_array = ds.pixel_array

        # Convert the pixel array to an OpenCV image
        cv_image = np.array(pixel_array, dtype=np.uint8)
        img1 = cv2.convertScaleAbs(cv_image, alpha=(255.0/255.0))

        # Extract SIFT features from image 1
        start_time = time.time()
        kp1, des1 = sift.detectAndCompute(img1, None)
        # end_time = time.time()
        # sift_time1 = end_time - start_time

        # Extract ORB features from image 1
        # start_time = time.time()
        kp2, des2 = orb.detectAndCompute(img1, None)
        # end_time = time.time()
        # orb_time1 = end_time - start_time

        # Loop through all other DICOM images to find matches
        for j in range(0,1):
            # Load DICOM image
            j=i+5
            filename2 = f'11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/1-{j:03d}.dcm'
            formatted_stringj = filename2.split("/")[-1]  
            j_format = formatted_stringj.split("-")[-1].split(".")[0]
            ds2 = pydicom.dcmread(filename2)

            # Extract the pixel array
            pixel_array2 = ds2.pixel_array

            # Convert the pixel array to an OpenCV image
            cv_image2 = np.array(pixel_array2, dtype=np.uint8)
            img2 = cv2.convertScaleAbs(cv_image2, alpha=(255.0/255.0))

            # Extract SIFT features from image 2
            # start_time = time.time()
            kp3, des3 = sift.detectAndCompute(img2, None)
            # end_time = time.time()
            # sift_time2 = end_time - start_time

            # Extract ORB features from image 2
            # start_time = time.time()
            kp4, des4 = orb.detectAndCompute(img2, None)
            end_time = time.time()
            comp_time = end_time - start_time

            if (a != 'ORB_create'):
                # print('ff')
                start_time = time.time()
                bf = cv2.BFMatcher()
                matches_sift = bf.knnMatch(des1,des3,k=2)
                
                # Lowe's Ratio test
                good1 = []
                for m, n in matches_sift:
                    if m.distance < 0.7*n.distance:
                        good1.append(m)

                # Apply RANSAC to find the best set of matches
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1, 2)
                dst_pts = np.float32([ kp3[m.trainIdx].pt for m in good1 ]).reshape(-1, 2)

                # print(len(src_pts), len(dst_pts))
                best_src_pts = None
                best_dst_pts = None
                best_inliers = 0

                # Ransac
                model, inliers = ransac((src_pts, dst_pts),AffineTransform, min_samples=4,residual_threshold=8, max_trials=10000)

                n_inliers = np.sum(inliers)
                inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
                inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
                placeholder_matches1 = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
                end_time = time.time()
                m1 = end_time - start_time
                # print(placeholder_matches1)

            else:
                pass
            
            if (b != 'ORB_create'):
                start_time = time.time()
                bf = cv2.BFMatcher()
                matches_orb = bf.knnMatch(des2,des4,k=2)
                
                # Lowe's Ratio test
                good2 = []
                for m, n in matches_orb:
                    if m.distance < 0.7*n.distance:
                        good2.append(m)

                # Apply RANSAC to find the best set of matches
                src_pts = np.float32([ kp2[m.queryIdx].pt for m in good2 ]).reshape(-1, 2)
                dst_pts = np.float32([ kp4[m.trainIdx].pt for m in good2 ]).reshape(-1, 2)

                # print(len(src_pts), len(dst_pts))
                best_src_pts = None
                best_dst_pts = None
                best_inliers = 0

                # Ransac
                model, inliers = ransac((src_pts, dst_pts),AffineTransform, min_samples=4,residual_threshold=8, max_trials=10000)

                n_inliers = np.sum(inliers)
                inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
                inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
                placeholder_matches2 = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
                end_time = time.time()
                m2 = end_time - start_time
            
            else:
                pass

            if (a == 'ORB_create'):
                start_time = time.time()
                bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # print('check1')
                # if (a == 'ORB_create'):
                matches_sift = bf1.match(des1, des3)
                matches_sift1 = sorted(matches_sift, key=lambda x: x.distance)
                src_pts = np.float32([kp2[m.queryIdx].pt for m in matches_sift1]).reshape(-1, 1, 2)

                dst_pts = np.float32([kp4[m.trainIdx].pt for m in matches_sift1]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    # Select only the inliers

                matches_mask = mask.ravel().tolist()
                # print('check2')

                placeholder_matches3 = [m for i, m in enumerate(matches_sift1) if matches_mask[i]]
                end_time = time.time()
                m3 = end_time - start_time
            
            else:
                pass

            if (b == 'ORB_create'):
                start_time = time.time()
                bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches_orb = bf1.match(des2, des4)
                matches_orb1 = sorted(matches_orb, key=lambda x: x.distance)
                src_pts = np.float32([kp2[m.queryIdx].pt for m in matches_orb1]).reshape(-1, 1, 2)

                dst_pts = np.float32([kp4[m.trainIdx].pt for m in matches_orb1]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    # Select only the inliers

                matches_mask = mask.ravel().tolist()
                # print('check3')

                placeholder_matches4 = [m for i, m in enumerate(matches_orb1) if matches_mask[i]]
                end_time = time.time()
                m4 = end_time - start_time
            
            else:
                pass
                    
   
            # if ((a != 'ORB_create') and (b != 'ORB_create')):
            #     gd_matches = placeholder_matches1 + placeholder_matches2
            # elif ((a == 'ORB_create') or (b == 'ORB_create')) and (b != 'ORB_create'):
            #     gd_matches = placeholder_matches2 + placeholder_matches3
            # elif ((a == 'ORB_create') or (b == 'ORB_create')) and (a != 'ORB_create'):
            #     gd_matches = placeholder_matches1 + placeholder_matches3

            
            
            if ((a != 'ORB_create') and (b != 'ORB_create')):
                gd_matches = placeholder_matches1 + placeholder_matches2
                match_time = m1 + m2
                a1  = good1
                a2 = good2
            elif (a != 'ORB_create') and (b == 'ORB_create'):
                gd_matches = placeholder_matches1 + placeholder_matches4
                match_time = m1 + m4
                a1 = good1
                a2 = matches_orb1
            elif (b != 'ORB_create') and (a == 'ORB_create'):
                gd_matches = placeholder_matches2 + placeholder_matches3
                match_time = m2 + m3
                a1 = matches_sift1
                a2 = good2






            kp1C = np.concatenate((kp1, kp3))
            kp2C = np.concatenate((kp2, kp4))

            #Display image
            img3 = cv2.drawMatches(img1, kp1C, img2, kp2C, gd_matches, None, flags=2)
            plt.title('Final'), plt.imshow(img3), plt.show()
            conc = f"{i_format} - {j_format}"
            filenamea.append(conc)
            kp1ca.append(len(kp1C))
            kp2ca.append(len(kp2C))
            matchesa1.append(len(a1))
            matchesa2.append(len(a2))
            good_matchesa.append(len(gd_matches))
            comp_timea.append(comp_time*1000)
            match_timea.append(match_time*1000)
            
    #
    # print('images :',filenamea)        
    # print("Number of features in image 1 :", kp1ca)
    # print("Number of features in image 2 :", kp2ca)
    # print("Number of features matched by " +  a + " Algorithm:", matchesa1)
    # print("Number of features matched by " +  b + " Algorithm:", matchesa2)
    # print("Number of features matched after RANSAC(combined):", good_matchesa)
    # print('Feature Extraction Computation Time - ',comp_timea) 
    # print('Feature Matching Computation Time - ',match_timea)


    print("Number of Average features in image 1 :", round(np.array(kp1ca).mean()))
    print("Number of Average features in image 2 :", round(np.array(kp2ca).mean()))
    print("Number of features matched by " +  a + " Algorithm:", round(np.array(matchesa1).mean()))
    print("Number of features matched by " +  b + " Algorithm:", round(np.array(matchesa2).mean()))
    print("Number of Average features matched after RANSAC:", round(np.array(good_matchesa).mean()))
    print('Feature Extraction Computation Average Time - ',round(np.array(comp_timea).mean(),2)) 
    print('Feature Matching Computation Average Time  - ',round(np.array(match_timea).mean(),2))








































































            # # Match descriptors using Brute-Force Matcher
            # start_time = time.time()
            
           
            # end_time = time.time()
            # match_time = end_time - start_time

            # # Lowe's Ratio test
            # good = []
            # for m, n in matches_sift:
            #     if m.distance < 0.7*n.distance:
            #         good.append(m)

            # #img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


            # # Apply RANSAC to find the best set of matches
            # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
            # dst_pts = np.float32([ kp3[m.trainIdx].pt for m in good ]).reshape(-1, 2)

            # print(len(src_pts), len(dst_pts))
            # best_src_pts = None
            # best_dst_pts = None
            # best_inliers = 0

            # # Ransac
            # model, inliers = ransac((src_pts, dst_pts),AffineTransform, min_samples=4,residual_threshold=8, max_trials=10000)

            # n_inliers = np.sum(inliers)
            # inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            # inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            # placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

            # if ((sift == 'ORB_create') or (orb == 'ORB_create')):

            #     matches_orb = sorted(matches_orb, key=lambda x: x.distance)

            #     # Print the features in each image
            #     print("Features in Image 1: ", len(kp1))
            #     print("Features in Image 2: ", len(kp2))

            #     # Print the number of matched features
            #     print("No. matched features: ", len(matches_orb))

            #     # Apply RANSAC to filter out outliers

            #     src_pts = np.float32([kp2[m.queryIdx].pt for m in matches_orb]).reshape(-1, 1, 2)

            #     dst_pts = np.float32([kp4[m.trainIdx].pt for m in matches_orb]).reshape(-1, 1, 2)

            #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            #     # Select only the inliers

            #     matches_mask = mask.ravel().tolist()

            #     good_matches = [m for i, m in enumerate(matches_orb) if matches_mask[i]]

            #     good = len(good_matches)
            #     # Print the number of matched features after removing outliers and finding the best match
            #     print("No. of matched features after removing outliers and finding the best match: ", len(good_matches))
            
            # else:

            #     good = []
            #     for m, n in matches_sift:
            #         if m.distance < 0.7*n.distance:
            #             good.append(m)

            # #img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


            # # Apply RANSAC to find the best set of matches
            #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
            #     dst_pts = np.float32([ kp3[m.trainIdx].pt for m in good ]).reshape(-1, 2)

            #     print(len(src_pts), len(dst_pts))
            #     best_src_pts = None
            #     best_dst_pts = None
            #     best_inliers = 0

            #     # Ransac
            #     model, inliers = ransac((src_pts, dst_pts),AffineTransform, min_samples=4,residual_threshold=8, max_trials=10000)

            #     n_inliers = np.sum(inliers)
            #     inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            #     inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            #     placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]


















            
    # results = np.array([round(np.array(kp1ca).mean()), round(np.array(kp2ca).mean()), round(np.array(matchesa).mean()),
    #                         round(np.array(good_matchesa).mean()),round(np.array(comp_timea).mean()),
    #                         round(np.array(match_timea).mean())])
    # print(len(df))


    # df.loc[i] = results

# print(df) 










    


#     results = (np.array([round(np.array(kp1ca).mean()), round(np.array(kp2ca).mean()), round(np.array(matchesa).mean()),
#                     round(np.array(good_matchesa).mean()),round(np.array(comp_timea).mean()),
#                     round(np.array(match_timea).mean())]))
    
# print(results)
# plt.plot(matchesa)
# plt.show()

# df = pd.DataFrame(results, index=algo_res, columns=['Avg_features_img1', 'Avg_features_img2', 'Avg_features_matched', 'Avg_features_matched_RANSAC', 'Avg_Feature_Extract_time','Avg_Feature_Match_time'])

# print(df)


