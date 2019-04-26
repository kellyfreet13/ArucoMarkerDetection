import aruco_calculations as ac
import threading
import time
import random

if __name__ == "__main__":

    # create a thread, set to kill on app close
    marker_finder = threading.Thread(target=ac.single_frame_continuous_capture)
    marker_finder.daemon = True
    marker_finder.start()

    ac.marker_to_find_id = 1
    local_x = 0
    local_z = 0
    while True:
        local_x = X
        local_z = Z
        print('[main.py] x: {0} meters, z:{1} meters'.format(local_x, local_z))
        time.sleep(.2)
