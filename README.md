# ArucoMarkerDetection
Aruco marker tracking and subsequent movement towards a specified marker.
<br /> <br />

<h3><b>About</b></h3>
<br />
This project is a component of a larger, ongoing application to pick up and move a box with multiple autonomous vehicles.
This component seeks to locate an Aruco marker with a specified id, and calculate the z distance to the marker as well as the x offset, and send the subsequent movement commands to the vehicle.
<br />

<h3><b>Software Libraries</b></h3>
<br />
OpenCV was the computer vision library used, as well as numpy for numerical processing of course.
<br />

<h3><b>Hardware</b></h3>
<br />
This is performed on a Raspberry Pi 3 Model B V1.2 with a Raspberry Camera module V2.1.
<br />

<h3><b>Running this Project</b></h3>
<br />
You must have the main and contrib libraries of opencv 3.4 or greater installed on your machine. <br />
You also must execute these commands to source .profile and run in the cv virtual env <br />
$ source ~/.profile <br />
$ workon cv
