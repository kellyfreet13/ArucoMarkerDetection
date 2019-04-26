import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv2.aruco as aruco
import glob
import math
from transforms3d.euler import mat2euler, euler2mat

def from_frame_capture():
    # get the camera's location
    rotM = cv2.Rodrigues(rvecs)[0].get()
    print('rotM shape', rotM.shape)

    print('tvecs.get(): ', tvecs.get())
    print('tvecs.get() shape: ', tvecs.get().shape)
    print('tvecs.get()[0] shape: ', tvecs.get()[0].shape)
    print('tvecs.get()[0][0] shape: ', tvecs.get()[0][0].shape)

    cam_pos = -np.matrix(rotM).T * np.matrix(tvecs.get()[0][0].reshape((3, 1)))
    cam_pos = np.array(cam_pos).reshape((1, 3))
    print('cam_pos: ', cam_pos)
    print('x: ', cam_pos[0][0])
    print('y: ', cam_pos[0][1])
    print('z: ', cam_pos[0][2])
    print('\n')

# with a single marker for now
# rvec = rvecs.get()[0][0]
# tvec = tvecs.get()[0][0]

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


# essentially trying to do Rodrigues myself, so naive...
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
            aruco.estimatePoseSingleMarkers(corners, ARUCO_SQUARE_WIDTH, u_cam_mtx, u_dist_cf)

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


def plotting():
    # estimate center of marker from corner positions
    print('corners: ', corners[0].reshape((4, 2)))
    r_corners = corners[0].reshape((4, 2))

    mid_x0 = (r_corners[0][0] + r_corners[2][0]) / 2
    mid_y0 = (r_corners[0][1] + r_corners[2][1]) / 2

    mid_x1 = (r_corners[1][0] + r_corners[3][0]) / 2
    mid_y1 = (r_corners[1][1] + r_corners[3][1]) / 2

    avg_mid_x = (mid_x0 + mid_x1) / 2
    avg_mid_y = (mid_y0 + mid_y1) / 2

    print('tvecs: ', tvecs.get())

    # get distance of a side
    side_len_px = r_corners[1][0] - r_corners[0][0]
    print('dist of a side: ', side_len_px)

    x_scale = 343  # pixels
    dist = x_scale / side_len_px
    print('estimated distance in meters: ', dist)

    to_plot = corners[0][0]
    x = to_plot[:, 0]
    y = to_plot[:, 1]
    x = np.append(x, avg_mid_x)
    y = np.append(y, avg_mid_y)
    plt.xlim(0, 3000)
    plt.ylim(0, 3000)
    plt.xticks(np.arange(0, 3000 + 1, 500))
    plt.yticks(np.arange(0, 3000 + 1, 500))
    plt.scatter(x, y)
    plt.show()


def translate_to_origin(corners):
    r_corners = corners[0].reshape((4,2))

    mid_x0 = (r_corners[0][0]+r_corners[2][0]) / 2
    mid_y0 = (r_corners[0][1]+r_corners[2][1]) / 2

    mid_x1 = (r_corners[1][0]+r_corners[3][0]) / 2
    mid_y1 = (r_corners[1][1]+r_corners[3][1]) / 2

    avg_mid_x = (mid_x0+mid_x1) / 2
    avg_mid_y = (mid_y0+mid_y1) / 2

    return avg_mid_x, avg_mid_y
