import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import time
import json

from picamera.array import PiRGBArray
from picamera import PiCamera

# size in meters of aruco marker
ARUCO_SQUARE_WIDTH = 0.141  # formerly 0.152
CALIB_FILENAME = 'camera_calib.json'


def video_debugging():
    # init the camera and grab a reference to the raw camera capture
    # BROKEN --> CAMERA RESOLUTION MISMATCH BETWEEN CALIBRATION AND BELOW
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 16
    raw_capture = PiRGBArray(camera, size=(640, 480))

    # allow camera to warmup -> tested safe value
    time.sleep(0.2)

    # create aruco marker dictionary and get camera calibration
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    camera_mtx, dist_coeffs = load_camera_calibration(CALIB_FILENAME)

    print('using pi camera video')

    #raw_capture.truncate(0)  # clean
    # test continuous video first, easier debugging
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # convert -> array -> umat
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
                # reshape r and t vectors for matrix multiplication
                rvec = rvecs.get()[i][0].reshape((3,1))
                tvec = tvecs.get()[i][0].reshape((3,1))

                # get camera pose
                R, _ = cv2.Rodrigues(rvec)
                R = np.mat(R).T
                cam_pose = -R * np.mat(tvec)
                cam_pose = np.squeeze(np.asarray(cam_pose))

                # extract z value from pose and get x offset
                z = cam_pose[-1]
                x = tvec[0][0]
                z = round(z, 3)
                x = round(x, 3)
                z_str = 'z: {0} meters'.format(z)
                x_str = 'x: {0} meters'.format(x)

                # draw detected marker borders and x, y, and z axes
                aruco.drawDetectedMarkers(u_frame, corners, ids)
                posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)

                # write x and z values to frame and who
                cv2.putText(posed_img, z_str, (0, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
                cv2.putText(posed_img, x_str, (0, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
                cv2.imshow("aruco detection", posed_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        # clean for next frame
        raw_capture.truncate(0)


# THIS WORKS BECAUSE --> CAMERA RESOLUTION FROM CALIBRATION IS SAME
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
        print('image shape: ', image.shape)
        # detectMarkers with frame, marker dict, marker corners, and marker ids
        corners, ids, _ = aruco.detectMarkers(u_frame, marker_dict)

        if ids.get() is None:
            print('no aruco marker detected')
            time.sleep(1)  # don't overload the lil pi
            continue

        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))
        rvecs, tvecs, obj_pts = \
            aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

        for i in range(ids.get().shape[0]):
            # reshape r and t vectors for matrix multiplication
            rvec = rvecs.get()[i][0].reshape((3,1))
            tvec = tvecs.get()[i][0].reshape((3,1))

            # get camera pose
            R, _ = cv2.Rodrigues(rvec)
            R = np.mat(R).T
            cam_pose = -R * np.mat(tvec)
            cam_pose = np.squeeze(np.asarray(cam_pose))

            # extract z value from pose and get x offset
            z = cam_pose[-1]
            x = tvec[0][0]
            z = round(z, 3)
            x = round(x, 3)
            z_str = 'z: {0} meters'.format(z)
            x_str = 'x: {0} meters'.format(x)

            # draw detected markers on frame
            aruco.drawDetectedMarkers(u_frame, corners, ids)
            posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)

            # show to the user (remove once on teensy car)
            cv2.namedWindow('aruco', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('aruco', 600,600)
            cv2.putText(posed_img, z_str, (260,290), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0,0,0))
            cv2.putText(posed_img, x_str, (260,420), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0,0,0))

            # to confirm, console output
            cv2.imshow('aruco', posed_img)
            print('z distance:', z)
            print('x offset: ', x)
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
    # testing pre-captured images -> not live
    x_offset_reg = 'x_offset*.jpg'
    z_dist_reg = 'meter*.jpg'
    xz_test_dir = './aruco_imgs/'
    test_dir_apr = './test_images_Apr_25/'
    all_reg = '*.jpg'
    #test_aruco_image_folder(test_dir_apr, all_reg)
