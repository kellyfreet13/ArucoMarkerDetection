import numpy as np
import cv2
import glob
import time
import json


def camera_calibration():
    #criteria termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    board_h = 9
    board_w = 6

    objp = np.zeros((board_w*board_h, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_h,0:board_w].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./calib_images/*.jpg')
    images = sorted(images)

    for fname in images:
        print('reading img: ', fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        start = time.time()

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (board_h, board_w), None)
        end = time.time()

        # If found, add object points, image points (after refining them)
        if ret == True:
            print('found corners!')
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            window_name = 'img'
            img = cv2.drawChessboardCorners(img, (board_h, board_w), corners2,ret)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 1200)
            cv2.imshow(window_name,img)
            cv2.waitKey(0)

    # see what the calibration comparision looks like
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print('ret: ', ret)
    print('mtx: ', mtx)
    print('dist: ', dist)
    print('rvecs: ', rvecs)
    print('tvecs: ', tvecs)

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    print('new mtx: ', newcameramtx)

    list_mtx = mtx.tolist()
    list_dist = dist.tolist()
    calib = {"camera_matrix": list_mtx, "dist_coeff": list_dist}
    calib_fname = 'camera_calib.json'
    with open(calib_fname, "w") as f:
        json.dump(calib, f)

    cv2.destroyAllWindows()


# call the fcn
camera_calibration()
