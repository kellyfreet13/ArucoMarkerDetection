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


class ArucoCalculator(threading.Thread):
    def __init__(self, secs):
        threading.Thread.__init__(self)
        self.continual_capture_seconds = secs

    def run(self):
        print('[AC] using pi camera single frame capture')

        # determining whether to save to files and draw axis
        # anything besides distance and offset calculation
        debugging = False

        # declare as global to share between my and miguel's thread
        global offset
        global distance
        global marker_id_to_find

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
            c.acquire()

            # check if we've been given a marker to find.
            # if not, wait until the other thread notifies us for an id
            if marker_id_to_find is None:
                print('[AC, f] no marker set to find. waiting')
                c.wait()

            else:
                print('[AC, t] trying to find marker with id {0}'.format(marker_id_to_find))
                camera.capture(raw_capture, format="bgr") # shape W x H x 3
                image = raw_capture.array  # shape H x W x 3

                # input:  W x H (desired)
                # output: H x W x 3
                image = cv2.resize(image, dsize=(3280, 2464))

                # output: H x W x 1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
                    if marker_id_to_find in ids.get().flatten():

                        # do conversion to UMat for opencv
                        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
                        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))

                        # get rotation, translation for each single aruco marker
                        rvecs, tvecs, obj_pts = \
                             aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

                        # get the index of the marker id we want for the rest of the arrays
                        flat = ids.get().flatten()
                        id_index, = np.where(flat == int(marker_id_to_find))

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
                        z_fmt = 'z: {0} meters'.format(z)
                        x_fmt = 'x: {0} meters'.format(x)

                        # set to global
                        offset = x
                        distance = z

                        if debugging:
                            # draw detected markers on frame
                            aruco.drawDetectedMarkers(u_frame, corners, ids)
                            posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvec, tvec, 0.1)

                            # draw x and z distance calculation on frame
                            cv2.namedWindow('aruco', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('aruco', 600, 600)
                            cv2.putText(posed_img, z_fmt, (260, 290), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (10,10,10))
                            cv2.putText(posed_img, x_fmt, (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (10,10,10))

                            # save image so we can see what it looks like
                            save_fname = './live_images/' + str(offset) + 'x_' + str(distance) + 'z.jpg'
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

                # clean up
                cv2.destroyAllWindows()
                raw_capture.truncate(0)

                # let everyone know we're done
                c.notify_all()

                #print('[AC tf] global x: {0}, z: {1}'.format(offset, distance))

            # free the lock
            c.release()

            # schleep, *sheets rustle*
            time.sleep(self.continual_capture_seconds)


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


def load_camera_calibration(filename):
    with open(filename) as f:
        calib_json = json.load(f)
    camera_mtx = calib_json["camera_matrix"]
    dist_coeffs = calib_json["dist_coeff"]

    return camera_mtx, dist_coeffs


if __name__ == "__main__":
    seconds_per_capture = 2
    ac = ArucoCalculator(seconds_per_capture)
    mt = MiguelsThread()

    ac.start()
    mt.start()

