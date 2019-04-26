import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv2.aruco as aruco
import glob
import time
import json
import math
from transforms3d.euler import mat2euler, euler2mat

from picamera.array import PiRGBArray
from picamera import PiCamera

# size in meters of aruco marker
ARUCO_SQUARE_WIDTH = 0.141  # formerly 0.152
CALIB_FILENAME = 'camera_calib.json'


def video_debugging():
    # init the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=(640, 480))

    # allow camera to warmup
    time.sleep(0.1)

    # create aruco marker dictionary and get camera calibration
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    camera_mtx, dist_coeffs = load_camera_calibration(CALIB_FILENAME)

    print('using pi camera video')

    raw_capture.truncate(0)  # clean

    # test continuous video first, easier debugging
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

        image = frame.array
        u_frame = cv2.UMat(image)

        # detectMarkers with frame, marker dict, marker corners, and marker ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(u_frame, marker_dict)
        if ids.get() is None:
            print('no aruco marker detected')
            raw_capture.truncate(0)  # clean up
            time.sleep(1)  # don't overload the lil pi
            continue

        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))
        rvecs, tvecs, obj_pts = \
            aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

        if ids is not None:
            for i in range(ids.get().shape[0]):
                rvec = rvecs.get()[i][0].reshape((3,1))
                tvec = tvecs.get()[i][0].reshape((3,1))
                R, _ = cv2.Rodrigues(rvec)
                R = np.mat(R).T
                cam_pose = -R * np.mat(tvec)
                z = cam_pose[2][0]
                z = np.squeeze(np.asarray(z))
                print('x offset', tvec)
                print('R.type', type(R))
                print('tvecs.get()', tvecs.get())
                print('tvecs.get()[i].type)', type(tvecs.get()[i]))
                print('CP:', cam_pose)
                print('CP.shape', cam_pose.shape)
                print('z:', z)

                aruco.drawDetectedMarkers(u_frame, corners, ids)
                posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)
                cv2.putText(posed_img, "z: %.1f meters" % z, (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10,10,10))

                cv2.imshow("aruco detection", posed_img)

        # with a single marker for now
        rvec = rvecs.get()[0][0]
        tvec = tvecs.get()[0][0]

        # getting point coords in camera's world
        #rot_mat = cv2.Rodrigues(rvec)
        #print('rot_mat: ', rot_mat)

        # S.O. answer via alexk. no upvotes, this looks promising
        #R, _ = cv2.Rodrigues(rvecs.get())
        #camera_pose = -np.mat(R).T*np.mat(tvecs.get().reshape((3,1)))
        #x = camera_pose[0][0][0]
        #y = camera_pose[0][0][1]
        #z = camera_pose[0][0][2]
        #print('R:', R)
        #print('CP:', camera_pose)
        #print('z:', z)
        #print('tvecs.get() dtype:', type(tvecs.get()))

        # draw corner and id
        #aruco.drawDetectedMarkers(u_frame, corners, ids)

        # change in the future when multiple markers (0.1 is length of vevs drawn)
        #num_ids = ids.get().shape[0]
        #for i in range(num_ids):
            #posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvecs, tvecs, 0.1)

        # via hungarian guy hoverno. TODO: check units
        #cv2.putText(posed_img, "z: %.1f units" % z, (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,120,244))
        #cv2.putText(posed_img, "y: %.1f units" % y, (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,244,70))
        #cv2.putText(posed_img, "x: %.1f units" % x, (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244,60,140))

        #cv2.imshow("aruco_detection", posed_img)
        print('rvecs: ', rvecs.get())
        print('tvecs: ', tvecs.get())
        print('\n')

        # clean up and exit condition
        key = cv2.waitKey(1) & 0XFF
        raw_capture.truncate(0)
        if key == ord("q"):
            break


# TODO: Pass in frames per second to capture?
def single_frame_continuous_capture():
    print('using pi camera single frame capture')

    # init the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=(640, 480))

    # allow camera to warm up
    time.sleep(0.1)

    # create aruco marker dictionary and get camera calibration
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    camera_mtx, dist_coeffs = load_camera_calibration(CALIB_FILENAME)

    while True:
        # grab an image from the camera and get numpy arr
        camera.capture(raw_capture, format="bgr")
        image = raw_capture.array
        u_frame = cv2.UMat(image)

        # detect the aruco marker
        corners, ids, rejectedImgPoints = aruco.detectMarkers(u_frame, marker_dict)
        if ids.get() is None:
            print('no aruco marker detected')
            raw_capture.truncate(0)  # clean
            time.sleep(0.2)
            continue

        # do conversion to UMat for opencv
        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))

        # get rotation, translation for each single aruco marker
        rvecs, tvecs, obj_pts = \
             aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

        if ids is not None:
            for i in range(ids.get().shape[0]):
                rvec = rvecs.get()[i][0].reshape((3,1))
                tvec = tvecs.get()[i][0].reshape((3,1))

                R, _ = cv2.Rodrigues(rvec)
                R = np.mat(R).T

                # get the camera pose
                cam_pose = -R * np.mat(tvec)
                cam_pose = np.squeeze(np.asarray(cam_pose))

                # extract x offset and z camera to marker distance
                z = cam_pose[-1]
                x = cam_pose[-3]
                z_fmt = 'z: {0} meters'.format(z)
                x_fmt = 'x: {0} meters'.format(x)

                print('x offset', tvec)
                print('R.type', type(R))
                print('tvecs.get()', tvecs.get())
                print('tvecs.get()[i].type)', type(tvecs.get()[i]))
                print('CP:', cam_pose)
                print('CP.shape', cam_pose.shape)
                print('z:', z)

                aruco.drawDetectedMarkers(u_frame, corners, ids)
                posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)
                cv2.putText(posed_img, z_fmt, (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10,10,10))
                cv2.putText(posed_img, x_fmt, (0, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10,10,10))
                cv2.imshow("aruco detection", posed_img)

                # clean up and exit condition
                key = cv2.waitKey(0) & 0XFF
                raw_capture.truncate(0)
                if key == ord("q"):
                    break

        # execute this every second
        print(time.time())
        time.sleep(2)
        print(time.time())


def test_aruco_image_folder(dir, expression):

    # create an aruco marker dictionary load camera calibration
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    camera_mtx, dist_coeffs = load_camera_calibration(CALIB_FILENAME)

    images = glob.glob(dir+expression)
    images = sorted(images)
    for fname in images:
        print(fname)
        image = cv2.imread(fname, 0)
        u_frame = cv2.UMat(image)

        # detectMarkers with frame, marker dict, marker corners, and marker ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(u_frame, marker_dict)

        if ids.get() is None:
            print('no aruco marker detected')
            time.sleep(1)  # don't overload the lil pi
            continue

        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))
        rvecs, tvecs, obj_pts = \
            aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

        for i in range(ids.get().shape[0]):
            rvec = rvecs.get()[i][0].reshape((3,1))
            tvec = tvecs.get()[i][0].reshape((3,1))

            R, _ = cv2.Rodrigues(rvec)
            R = np.mat(R).T
            cam_pose = -R * np.mat(tvec)
            cam_pose = np.squeeze(np.asarray(cam_pose))

            x = cam_pose[-3]
            y = cam_pose[-2]
            z = cam_pose[-1]
            z_str = 'z: {0} meters'.format(z)

            aruco.drawDetectedMarkers(u_frame, corners, ids)
            posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)

            cv2.namedWindow('aruco', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('aruco', 600,600)
            cv2.putText(posed_img, z_str, (300,300), cv2.FONT_HERSHEY_DUPLEX, 5.0, (50,50,50))

            cv2.imshow('aruco', posed_img)
            print('x distance:', x)
            print('y distance:', y)
            print('z distance:', z)
            print('x offset: ', tvec[0][0])
            print('\n')

            key = cv2.waitKey(10000) & 0XFF
            if key == ord("q"):
                break


def load_camera_calibration(filename):
    with open(filename) as f:
        calib_json = json.load(f)
    camera_mtx = calib_json["camera_matrix"]
    dist_coeffs = calib_json["dist_coeff"]

    return camera_mtx, dist_coeffs


if __name__ == "__main__":
    test_aruco_image_folder('./aruco_imgs/', 'x_offset*.jpg')
    #test_rotations()
    #init_camera()
    #camera_calibration()
