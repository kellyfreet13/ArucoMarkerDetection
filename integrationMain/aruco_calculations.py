import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import time
import json
import io
import subprocess
import threading

# import shared variables
import config

from picamera.array import PiRGBArray
from picamera import PiCamera

# size in meters of aruco marker
ARUCO_SQUARE_WIDTH = 0.141  # formerly 0.152
CALIB_FILENAME = 'camera_calib.json'


class ArucoCalculator(threading.Thread):
    def __init__(self, secs, c):
        threading.Thread.__init__(self)
        self.continual_capture_seconds = secs

        condition = c


    def run(self):

        print('[AC] using pi camera single frame capture')

        # determining whether to save to files and draw axis
        # anything besides distance and offset calculation
        debugging = False

        # init the camera and grab a reference to the raw camera capture
        # (680, 480) is almost twice as fast for conversion, less accurate though?
        # if low accuracy, change to (2592, 1952) or (3296, 2464): was a bit overestimating
        camera = PiCamera()
        camera.resolution = (640, 480)
        raw_capture = PiRGBArray(camera, size=(640, 480))

        # allow camera to warm up -> tested safe value
        time.sleep(0.3)

        # create aruco marker dictionary and get camera calibration
        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        camera_mtx, dist_coeffs = load_camera_calibration(CALIB_FILENAME)

        while True:
            # check if we've been given a marker to find.
            # if not, wait until the other thread notifies us for an id
            if config.QR_code is None:
                print('[AC, f] no marker set to find. sleeping')

                # sleep instead using wait()
                time.sleep(0.5)

            else:
                print('[AC, t] trying to find marker with id {0}'.format(config.QR_code))
                camera.capture(raw_capture, format="bgr") # shape W x H x 3
                image = raw_capture.array  # shape H x W x 3

                # input:  W x H (desired)
                # output: H x W x 3
                image = cv2.resize(image, dsize=(3280, 2464))

                # output: H x W x 1
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # typing for open cv
                u_frame = cv2.UMat(image)

                # detect the aruco marker
                corners, ids, _ = aruco.detectMarkers(u_frame, marker_dict)

                # no markers were found
                if ids.get() is None:
                    print('[AC, t] no aruco marker detected')
                    raw_capture.truncate(0)  # clean
                # marker(s) were found
                else:
                    # before we do any work, check if marker with given id is present
                    if config.QR_code in ids.get().flatten():

                        # do conversion to UMat for opencv
                        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
                        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))

                        # get rotation, translation for each single aruco marker
                        rvecs, tvecs, obj_pts = \
                             aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

                        # get the index of the marker id we want for the rest of the arrays
                        flat = ids.get().flatten()
                        id_index, = np.where(flat == int(QR_code))

                        # reshape t and r vectors for matrix multiplication
                        rvec = rvecs.get()[id_index][0].reshape((3,1))
                        tvec = tvecs.get()[id_index][0].reshape((3,1))

                        # Rodrigues baby, look it up, and convert to matrix
                        R, _ = cv2.Rodrigues(rvec)
                        R = np.mat(R).T

                        # get the camera pose
                        cam_pose = -R * np.mat(tvec)
                        cam_pose = np.squeeze(np.asarray(cam_pose))

                        # extract x offset and z camera to marker distance
                        z = cam_pose[-1]
                        x = tvec[0][0]

                        z = round(z, 3)
                        x = round(x, 3)

                        # critical section begin
                        condition.acquire()

                        # set to global
                        config.offset = x
                        config.distance = z

                        # critical section end
                        condition.release()

                        if debugging:
                            z_fmt = 'z: {0} meters'.format(z)
                            x_fmt = 'x: {0} meters'.format(x)

                            # draw detected markers on frame
                            aruco.drawDetectedMarkers(u_frame, corners, ids)
                            posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)

                            # draw x and z distance calculation on frame
                            cv2.namedWindow('aruco', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('aruco', 600, 600)
                            cv2.putText(posed_img, z_fmt, (20, 2200), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0,0,0))
                            cv2.putText(posed_img, x_fmt, (20, 2350), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0,0,0))
                            # save image so we can see what it looks like
                            save_fname = './live_images/' + str(config.offset) + 'x_' + str(config.distance) + 'z.jpg'
                            cv2.imwrite(save_fname, posed_img)

                            # show the image
                            cv2.imshow('aruco', posed_img)
                            cv2.resizeWindow('aruco', 600, 600)
                            cv2.waitKey(0)

                            # confirm with console
                            #print(z_fmt)
                            #print(x_fmt)

                            # clean up and exit condition
                            key = cv2.waitKey(1) & 0XFF
                            raw_capture.truncate(0)
                            if key == ord("q"):
                                break

                    # schleep, *sheets rustle*
                    time.sleep(self.continual_capture_seconds)

                # clean up
                cv2.destroyAllWindows()
                raw_capture.truncate(0)

                #print('[AC tf] global x: {0}, z: {1}'.format(offset, distance))


def load_camera_calibration(filename):
    with open(filename) as f:
        calib_json = json.load(f)
    camera_mtx = calib_json["camera_matrix"]
    dist_coeffs = calib_json["dist_coeff"]

    return camera_mtx, dist_coeffs


if __name__ == "__main__":
    seconds_per_capture = 0.5
    ac = ArucoCalculator(seconds_per_capture)
    mt = MiguelsThread()

    ac.start()
    mt.start()
