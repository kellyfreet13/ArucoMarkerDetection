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

            # check if we've been given a marker to find.
            # if not, wait until the other thread notifies us for an id
            if marker_id_to_find is None:
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

                # no markers were found
                if ids.get() is None:
                    print('[AC, t] no aruco marker detected')
                    raw_capture.truncate(0)  # clean
                    time.sleep(self.continual_capture_seconds)
                # marker(s) were found
                else:
                    # before we do any work, check if marker with given id is present
                    if int(marker_id_to_find) in ids.get():
                        # do conversion to UMat for opencv
                        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
                        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))

                        # get rotation, translation for each single aruco marker
                        rvecs, tvecs, obj_pts = \
                             aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_camera_mtx, u_dist_coeffs)

                        # get the index of the marker id we want for the rest of the arrays
                        id_index = ids.get().index(int(marker_id_to_find))

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
