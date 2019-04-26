import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import time
import json
import io
import subprocess
import threading

from picamera.array import PiRGBArray
from picamera import PiCamera

# size in meters of aruco marker
ARUCO_SQUARE_WIDTH = 0.141  # formerly 0.152
CALIB_FILENAME = 'camera_calib.json'

# share between my and miguel's thread
offset = 0
distance = 0
marker_id_to_find = None
c = threading.Condition()


def video_debugging():
    # init the camera and grab a reference to the raw camera capture
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


class ArucoCalculator(threading.Thread):
    def __init__(self, secs):
        threading.Thread.__init__(self)
        self.continual_capture_seconds = secs

    def run(self):
        print('using pi camera single frame capture')

        # declare as global to share between my and miguel's thread
        global offset
        global distance
        global marker_id_to_find

        # init the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        camera.resolution = (2592, 1952)
        #camera.framerate = 32
        raw_capture = PiRGBArray(camera, size=(2592, 1952))

        # allow camera to warm up -> tested safe value
        time.sleep(0.3)

        # create aruco marker dictionary and get camera calibration
        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        camera_mtx, dist_coeffs = load_camera_calibration(CALIB_FILENAME)

        while True:
            c.acquire()

            # check if we've been given a marker to find
            if marker_id_to_find is None:
                # time.sleep(self.continual_capture_seconds)
                print('[AC, f] no marker set to find. waiting')
                c.wait()

            else:
                print('[AC, t] finding marker with id {0}'.format(marker_id_to_find))
                camera.capture(raw_capture, format="bgr")
                img = raw_capture.array

                fname = 'cv2image.jpg'
                cv2.imwrite(fname, img)

                # some sketch shit
                subprocess.run(['convert', fname, '-resize', '3280x2464', fname])
                image = cv2.imread(fname, 0)

                u_frame = cv2.UMat(image)

                # detect the aruco marker
                corners, ids, _ = aruco.detectMarkers(u_frame, marker_dict)
                if ids.get() is None:
                    print('[AC, t] no aruco marker detected')
                    raw_capture.truncate(0)  # clean
                    time.sleep(self.continual_capture_seconds)

                else:
                    # do conversion to UMat for opencv
                    u_camera_mtx = cv2.UMat(np.array(camera_mtx))
                    u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))

                    # get rotation, translation for each single aruco marker
                    rvecs, tvecs, obj_pts = \
                         aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

                    # just for testing what the ids look like
                    print('ids: ', ids.get())

                    for i in range(ids.get().shape[0]):
                        # reshape t and r vectors for matrix multiplication
                        rvec = rvecs.get()[i][0].reshape((3,1))
                        tvec = tvecs.get()[i][0].reshape((3,1))

                        # Rodrigues baby, look it up, and convert to matrix
                        R, _ = cv2.Rodrigues(rvec)
                        R = np.mat(R).T

                        # get the camera pose
                        cam_pose = -R * np.mat(tvec)
                        cam_pose = np.squeeze(np.asarray(cam_pose))

                        # extract x offset and z camera to marker distance
                        z = cam_pose[-1]
                        x = tvec[0][0]

                        # testing an idea here
                        z *= .79024
                        x *= .78896
                        z = round(z, 3)
                        x = round(x, 3)
                        z_fmt = 'z: {0} meters'.format(z)
                        x_fmt = 'x: {0} meters'.format(x)

                        # set to global
                        offset = x
                        distance = z

                        # draw detected markers on frame
                        aruco.drawDetectedMarkers(u_frame, corners, ids)
                        posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)

                        # draw x and z distance calculation on frame
                        cv2.namedWindow('aruco', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('aruco', 600, 600)
                        cv2.putText(posed_img, z_fmt, (260, 290), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (10,10,10))
                        cv2.putText(posed_img, x_fmt, (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (10,10,10))

                        save_fname = './live_images/' + str(offset) + str(distance) + '.jpg'
                        cv2.imwrite(save_fname, posed_img)
                        #cv2.resizeWindow('aruco', 600, 600)
                        #cv2.waitKey(0)

                        # confirm with console
                        #print(z_fmt)
                        #print(x_fmt)

                        # clean up and exit condition
                        key = cv2.waitKey(1) & 0XFF
                        raw_capture.truncate(0)
                        if key == ord("q"):
                            break

                cv2.destroyAllWindows()
                # execute this every second

                time.sleep(self.continual_capture_seconds)
                #print('[aruco_calc.py] global x: {0}, z: {1}'.format(offset, distance))

                # let everyone know we're done
                c.notify_all()

            # free the lock
            c.release()


class MiguelsThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # declare as global to share between my and miguel's thread
        global offset
        global distance
        global marker_id_to_find

        while True:
            c.acquire()

            if marker_id_to_find is not None:
                print('[MT, t] offset: {0}, distance {1}'.format(offset, distance))
                c.wait()
            else:
                marker_id_to_find = 1
                print('[MT, f] setting marker to find as {0}'.format(marker_id_to_find))
                c.notify_all()
            c.release()


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

    ac = ArucoCalculator(2)
    mt = MiguelsThread()

    mt.start()
    ac.start()

    #video_debugging()
