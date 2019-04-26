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
aruco_square_len = 0.141  # formerly 0.152

def init_camera():
    # init the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow camera to warmup
    time.sleep(0.1)

    # read camera calibration from json
    calib_fname = 'camera_calib.json'
    with open(calib_fname) as f:
        calib_json = json.load(f)

    # extract values from json
    camera_mtx = calib_json["camera_matrix"]
    dist_coeffs = calib_json["dist_coeff"]

    # and create an aruco marker dictionary
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


    use_video = True

    if use_video:
        print('using pi camera video')

        rawCapture.truncate(0)  # clean

        # test continuous video first, easier debugging
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            image = frame.array
            u_frame = cv2.UMat(image)

            # detectMarkers with frame, marker dict, marker corners, and marker ids
            corners, ids, rejectedImgPoints = aruco.detectMarkers(u_frame, marker_dict)
            if ids.get() is None:
                print('no aruco marker detected')
                rawCapture.truncate(0)  # clean up
                time.sleep(1)  # don't overload the lil pi
                continue

            u_camera_mtx = cv2.UMat(np.array(camera_mtx))
            u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))
            rvecs, tvecs, obj_pts = \
                aruco.estimatePoseSingleMarkers(corners, aruco_square_len, u_camera_mtx, u_dist_coeffs)

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
            rawCapture.truncate(0)
            if key == ord("q"):
                break

    # temporary solution, consider using a thread or something later
    else:
        print('using pi camera single frame capture')

        while 1:

            # grab an image from the camera and get numpy arr
            camera.capture(rawCapture, format="bgr")
            image = rawCapture.array
            u_frame = cv2.UMat(image)

            # detect the aruco marker
            corners, ids, rejectedImgPoints = aruco.detectMarkers(u_frame, marker_dict)
            if ids.get() is None:
                print('no aruco marker detected')
                rawCapture.truncate(0)  # clean
                time.sleep(0.2)
                continue

            u_camera_mtx = cv2.UMat(np.array(camera_mtx))
            u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))
            rvecs, tvecs, obj_pts = \
                 aruco.estimatePoseSingleMarkers(corners, aruco_square_len, u_camera_mtx, u_dist_coeffs)

            # get the number of markers detected
            num_ids = ids.get().shape[0]
            for i in range(num_ids):
                posed_img = aruco.drawAxis(u_frame, u_camera_mtx, u_dist_coeffs, rvecs, tvecs, 0.1)

            cv2.imshow("aruco_detection", posed_img)

            # get the camera's location
            rotM = cv2.Rodrigues(rvecs)[0].get()
            print('rotM shape', rotM.shape)

            print('tvecs.get(): ', tvecs.get())
            print('tvecs.get() shape: ', tvecs.get().shape)
            print('tvecs.get()[0] shape: ', tvecs.get()[0].shape)
            print('tvecs.get()[0][0] shape: ', tvecs.get()[0][0].shape)

            cam_pos = -np.matrix(rotM).T * np.matrix(tvecs.get()[0][0].reshape((3,1)))
            cam_pos = np.array(cam_pos).reshape((1,3))
            print('cam_pos: ', cam_pos)
            print('x: ', cam_pos[0][0])
            print('y: ', cam_pos[0][1])
            print('z: ', cam_pos[0][2])
            print('\n')

            # clean up and exit condition
            key = cv2.waitKey(0) & 0XFF
            rawCapture.truncate(0)
            if key == ord("q"):
                break

            print(time.time())
            # execute this every 4 seconds, naive way
            time.sleep(4)
            print(time.time())


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

def test_aruco_offset():

    # read camera calibration from json
    calib_fname = 'camera_calib.json'
    with open(calib_fname) as f:
        calib_json = json.load(f)

    # extract values from json
    camera_mtx = calib_json["camera_matrix"]
    dist_coeffs = calib_json["dist_coeff"]

    # and create an aruco marker dictionary
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    images = glob.glob('./aruco_imgs/x_offset*.jpg')
    images = sorted(images)
    for fname in images:
        print(fname)
        image = cv2.imread(fname, 0)
        u_frame = cv2.UMat(image)

        # detectMarkers with frame, marker dict, marker corners, and marker ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(u_frame, marker_dict)

        if ids.get() is None:
            print('no aruco marker detected')
            rawCapture.truncate(0)  # clean up
            time.sleep(1)  # don't overload the lil pi
            continue

        u_camera_mtx = cv2.UMat(np.array(camera_mtx))
        u_dist_coeffs = cv2.UMat(np.array(dist_coeffs))
        rvecs, tvecs, obj_pts = \
            aruco.estimatePoseSingleMarkers(corners, aruco_square_len, u_camera_mtx, u_dist_coeffs)

        rvec = rvecs.get()[0][0].reshape((3,1))
        tvec = tvecs.get()[0][0].reshape((3,1))

        R, _ = cv2.Rodrigues(rvec)
        R = np.mat(R).T
        cam_pose = -R * np.mat(tvec)

        x = cam_pose[0][0]
        y = cam_pose[1][0]
        z = cam_pose[2][0]

        x = np.squeeze(np.asarray(cam_pose))
        y = np.squeeze(np.asarray(cam_pose))
        z = np.squeeze(np.asarray(cam_pose))

        x = x[-3]
        y = y[-2]
        z = z[-1]
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

        # ignore below for now
        continue

        # estimate center of marker from corner positions
        print('corners: ', corners[0].reshape((4,2)))
        r_corners = corners[0].reshape((4,2))

        mid_x0 = (r_corners[0][0]+r_corners[2][0]) / 2
        mid_y0 = (r_corners[0][1]+r_corners[2][1]) / 2

        mid_x1 = (r_corners[1][0]+r_corners[3][0]) / 2
        mid_y1 = (r_corners[1][1]+r_corners[3][1]) / 2

        avg_mid_x = (mid_x0 + mid_x1) / 2
        avg_mid_y = (mid_y0 + mid_y1) / 2

        print('tvecs: ', tvecs.get())

        # get distance of a side
        side_len_px = r_corners[1][0]-r_corners[0][0]
        print('dist of a side: ', side_len_px)

        x_scale = 343  # pixels
        dist = x_scale / side_len_px
        print('estimated distance in meters: ', dist)

        to_plot = corners[0][0]
        x = to_plot[:,0]
        y = to_plot[:,1]
        x = np.append(x, avg_mid_x)
        y = np.append(y, avg_mid_y)
        plt.xlim(0,3000)
        plt.ylim(0,3000)
        plt.xticks(np.arange(0, 3000+1, 500))
        plt.yticks(np.arange(0, 3000+1, 500))
        plt.scatter(x,y)
        plt.show()


def load_camera_calibration():

    # read camera calibration from json
    calib_fname = 'camera_calib.json'
    with open(calib_fname) as f:
        calib_json = json.load(f)
    camera_mtx = calib_json["camera_matrix"]
    dist_coeffs = calib_json["dist_coeff"]

    return camera_mtx, dist_coeffs



def translate_to_origin(corners):
    r_corners = corners[0].reshape((4,2))

    mid_x0 = (r_corners[0][0]+r_corners[2][0]) / 2
    mid_y0 = (r_corners[0][1]+r_corners[2][1]) / 2

    mid_x1 = (r_corners[1][0]+r_corners[3][0]) / 2
    mid_y1 = (r_corners[1][1]+r_corners[3][1]) / 2

    avg_mid_x = (mid_x0+mid_x1) / 2
    avg_mid_y = (mid_y0+mid_y1) / 2

    return avg_mid_x, avg_mid_y

def test_rotations():

    # get camera calibration and marker dictionary
    camera_mtx, dist_coeffs = load_camera_calibration()
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    images = glob.glob('./testing_rotation_aruco_imgs/*.jpg')
    for fname in images:
        print('testing image ', fname)
        image = cv2.imread(fname, 0)
        u_frame = cv2.UMat(image)

        corners, ids, _ = aruco.detectMarkers(u_frame, marker_dict)

        if ids.get() is None:
            print('no aruco marker detected for ', fname)

        u_cam_mtx = cv2.UMat(np.array(camera_mtx))
        u_dist_cf = cv2.UMat(np.array(dist_coeffs))
        rvecs, tvecs, obj_pts = \
            aruco.estimatePoseSingleMarkers(corners, aruco_square_len, u_cam_mtx, u_dist_cf)

        print('rotational vecs: ', rvecs.get())
        print('translational vecs: ', tvecs.get())

        # estimate center of marker from corner positions
        print('corners:\n', corners[0].reshape((4,2)))
        avg_mid_x, avg_mid_y = translate_to_origin(corners)

        # get distance of a side
        r_corners = corners[0].reshape((4,2))
        side_len_px = r_corners[1][0]-r_corners[0][0]
        print('dist of a side: ', side_len_px)

        x_scale = 343  # pixels
        dist = x_scale / side_len_px
        print('estimated distance in meters: ', dist)

        to_plot = corners[0][0]
        x1 = to_plot[:,0]
        y1 = to_plot[:,1]
        x1 = np.append(x1, avg_mid_x)
        y1 = np.append(y1, avg_mid_y)

        # translate by the estimated center of the aruco marker
        for i in range(len(x1)):
            x1[i] -= r_corners[0][0]
            y1[i] -= r_corners[0][1]

        # get rodrigues and use that for euler angles
        rodrigues, _ = cv2.Rodrigues(rvecs)
        euler_angles = mat2euler(rodrigues.get(), axes='rzxy')
        print('euler angles: ', euler_angles)

        rot_z = lambda theta: \
              np.array([[math.cos(theta),    -math.sin(theta),   0],
                        [math.sin(theta),    math.cos(theta),    0],
                        [0,                  0,                  1]
                        ])

        rot_y = lambda theta: \
                np.array([[math.cos(theta),     0, math.sin(theta)],
                          [0,                   1, 0              ],
                          [-math.sin(theta),    0, math.cos(theta)]
                        ])

        rot_x = lambda theta: \
                np.array([[1,       0,               0               ],
                          [0,       math.cos(theta), -math.sin(theta)],
                          [0,       math.sin(theta), math.cos(theta) ]
                        ])

        # each euler angle
        theta_z = -euler_angles[0]
        theta_y = -euler_angles[2]
        theta_x = -euler_angles[1]

        # trying all the possible dimensions
        rot1 = np.dot(rot_z(theta_z), np.dot(rot_y(theta_y), rot_x(theta_x)))  # zyx
        rot2 = np.dot(rot_z(theta_z), np.dot(rot_x(theta_x), rot_y(theta_y)))  # zxy
        rot3 = np.dot(rot_x(theta_x), np.dot(rot_y(theta_y), rot_z(theta_z)))  # xyz
        rot4 = np.dot(rot_x(theta_x), np.dot(rot_z(theta_z), rot_y(theta_y)))  # xzy
        rot5 = np.dot(rot_y(theta_y), np.dot(rot_x(theta_x), rot_z(theta_z)))  # yxz
        rot6 = np.dot(rot_y(theta_y), np.dot(rot_z(theta_z), rot_x(theta_x)))  # yzx

        # testing z dimension
        R_z1 = rot_z(euler_angles[0])
        R_y1 = rot_y(-euler_angles[2])
        R_x1 = rot_x(euler_angles[1])
        R_a = np.dot(rot_z(-euler_angles[0]),  rot_y(-euler_angles[2]))
        R_b = rot_x(-euler_angles[1])
        R_c = np.dot(R_x1, R_z1)


        # add a z-dimension ?
        ext = np.array([])
        for i in range(len(x1)):
            extended = np.append([x1[i], y1[i]], [1]).reshape((3,1))
            rotated = np.dot(rot2, extended)
            ext = np.append(ext, rotated)

        ext = ext.reshape((5,3))
        print('ext:\n', ext)
        x2 = ext[:,0]
        y2 = ext[:,1]

        fig, ax = plt.subplots()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.grid(True, which='both')

        plt.scatter(x2,y2,color="red")
        plt.scatter(x1,y1,color="blue")
        plt.show()

        print('--------------------------------------------\n')


# rvecs_corrected = rvecs.get()[0][0]
# if rvecs_corrected[0] < 0:
#   rvecs_cor = -(math.pi_rvecs_cor[0])
# else
#   rvecs_cor = math.pi-rvecs_cor[0]
# rvecs_cor = np.array([rvecs_cor.flatten().reshape((1,3))])
# rvecs_cor = cv2.UMat(rvecs_cor)

# temp_z = rvecs_cor.get()[0][0][1]
# temp_y = rvecs_cor.get()[0][0][2]
# temp_x = rvecs_cor.get()[0][0][0]
# temp_zyx = np.array([[[temp_z, temp_y, temp_x]]])
# zyx_corrected = cv2.UMat(temp_zyx)


# note: camera looks into z-axis, so
#   yaw rotates image
#   pitch turns image
#   roll tilts forwards/backwards

# calculate Rotation Matrix given euler angles
def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([[1,         0,                  0                  ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0])],
                    [0,         math.sin(theta[0]), math.cos(theta[0]) ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I-shouldBeIdentity)
    return n < 1e-6


def MY_mat2euler(R):
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([z, y, x])


if __name__ == "__main__":
    test_aruco_offset()
    #test_rotations()
    #init_camera()
    #camera_calibration()
