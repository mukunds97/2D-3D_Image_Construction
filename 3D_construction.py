import cv2
import math
import numpy as np
import os
import pydicom
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import matplotlib.pyplot as plt

algo = 'AKAZE'

# creates a point cloud file (.ply) from numpy array
def createPointCloud(filename, arr):
    # open file and write boilerplate header
    file = open(filename, 'w');
    file.write("ply\n");
    file.write("format ascii 1.0\n");

    # count number of vertices
    num_verts = arr.shape[0];
    file.write("element vertex " + str(num_verts) + "\n");
    file.write("property float32 x\n");
    file.write("property float32 y\n");
    file.write("property float32 z\n");
    file.write("end_header\n");

    # write points
    point_count = 0;
    for point in arr:
        # progress check
        point_count += 1;
        if point_count % 1000 == 0:
            print("Point: " + str(point_count) + " of " + str(len(arr)));

        # create file string
        out_str = "";
        for axis in point:
            out_str += str(axis) + " ";
        out_str = out_str[:-1]; # dump the extra space
        out_str += "\n";
        file.write(out_str);
    file.close();


# extracts points from mask and adds to list
def addPoints(pts, points_list, depth, algo):
    # mask_points = np.where(mask == 255);
    for ind in range(len(pts)):
        # get point
        print('All point',pts)
        # print(pts[0][1])
        # print('Point',pts[0][1][1])
        if algo == 'ORB':
            print(ind)
            x = pts[ind][0][0];
            y = pts[ind][0][1];
        else:
            x = pts[ind][0]
            y = pts[ind][1]
        point = [x,y,depth];
        points_list.append(point);

def main():
    # tweakables
    slice_thickness = 1.5; # distance between slices
    xy_scale = 1; # rescale of xy distance

    # load images
    folder = "/Users/mukund/Documents/ECE_613_PROJECT/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/";
    files = os.listdir(folder);
    # print(files)
    images = [];
    for file in files:
        if file[-4:] == ".dcm":
            ds = pydicom.dcmread(folder + file)
            pixel_array = ds.pixel_array
            cv_image = np.array(pixel_array, dtype=np.uint8)
            img = cv2.convertScaleAbs(cv_image, alpha=(255.0/255.0))
            images.append(img);


    print('Images', images)
    # keep a blank mask
    # blank_mask = np.zeros_like(images[0], np.uint8);

    # print(images)
    # # create masks
    # masks = [];
    # masks.append(blank_mask);
    # for image in images:
    #     # mask
    #     mask = cv2.inRange(image, 0, 100);

    #     # show
    #     # cv2.imshow("Mask", mask);
    #     # cv2.waitKey(0);
    #     masks.append(mask);
    # masks.append(blank_mask);
    # cv2.destroyAllWindows();

    # go through and get points
    depth = 0;
    points = [];

    # Declare Detector
    if algo == 'SIFT':
        detector = cv2.SIFT_create()
    elif algo == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif algo == 'ORB':
        detector = cv2.ORB_create()
    elif algo == 'AKAZE':
        detector = cv2.AKAZE_create()


    for index in range(1,len(images)-1):
        kp1, des1 = detector.detectAndCompute(images[index],None)
        kp2, des2 = detector.detectAndCompute(images[index+1],None)


        if algo == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:    
            bf = cv2.BFMatcher()

        if algo == 'ORB':
            matches = bf.match(des1,des2)
            matches = sorted(matches, key=lambda x: x.distance)
        else:
            matches = bf.knnMatch(des1,des2,k=2)

        if algo != 'ORB':
            good = []
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)
        
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


        # print(src_pts)

        addPoints(src_pts, points, depth, algo)
        addPoints(dst_pts, points, depth+slice_thickness, algo)

        # best_src_pts = None
        # best_dst_pts = None
        # best_inliers = 0

        #Removing Ransac

        # Ransac
        # try:
        #     model, inliers = ransac(
        #     (src_pts, dst_pts),
        #     AffineTransform, min_samples=4,
        #     residual_threshold=8, max_trials=10000
        #     )

        #     n_inliers = np.sum(inliers)
        #     inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        #     inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        #     placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

        #     src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
        #     dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

        #     print('Source Points',src_pts)
        #     print('Dest Points', dst_pts)
        #     addPoints(src_pts, points, depth)
        #     addPoints(dst_pts, points, depth+slice_thickness)
        
        # except:
        #     pass

        # progress check
        # print("Index: " + str(index) + " of " + str(len(masks)));

        # get three masks
        # prev = masks[index - 1];
        # curr = masks[index];
        # after = masks[index + 1];

        # print(prev)
        # print(curr)
        # print(after)

        # do a slice on previous
        # prev_mask = np.zeros_like(curr);
        # cv2.imshow("Prev Mask Before", prev_mask);
        # # cv2.imshow(0)
        # print(prev_mask)
        # prev_mask[prev == 0] = curr[prev == 0];
        # cv2.imshow("Prev Mask After", prev_mask);
        # cv2.waitKey(0)
        # print(prev_mask)
        # addPoints(prev_mask, points, depth);

        # # # do a slice on after
        # next_mask = np.zeros_like(curr);
        # print(next_mask)
        # next_mask[after == 0] = curr[after == 0];
        # print(next_mask)
        # addPoints(next_mask, points, depth);

        # # get contour points (_, contours) in OpenCV 2.* or 4.*
        # c, contours = cv2.findContours(curr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);
        # cv2.drawContours(curr, c, -1, (0, 255, 0),3)
        # plt.imshow(curr)
        # plt.show()
        # #BP
        # for con in contours:
        #     for point in con:
        #         print(point)
        #         p = point; # contours have an extra layer of brackets
        #         points.append([p[0], p[1], depth]);

        # increment depth
        depth += slice_thickness;

    # rescale x,y points
    for ind in range(len(points)):
        # unpack
        x,y,z = points[ind];

        # scale
        x *= xy_scale;
        y *= xy_scale;
        points[ind] = [x,y,z];

    # convert points to numpy and dump duplicates
    points = np.array(points).astype(np.float32);
    # print('Points',points)
    points = np.unique(points.reshape(-1, points.shape[-1]), axis=0);
    # print(points.shape);

    # save to point cloud file
    createPointCloud("test_" + algo + ".ply", points);

if __name__ == "__main__":
    main();