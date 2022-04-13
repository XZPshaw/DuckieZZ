#!/usr/bin/env python3.8

import os
import numpy as np
import yaml
import cv2
import os
import rospy
from tag import Tag
import math
#from pupil_apriltags import Detector
from dt_apriltags import Detector
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Empty, EmptyResponse
from cv_bridge import CvBridgeError,CvBridge
import rospkg
from scipy.spatial.transform import Rotation as R

class MyNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.undistort_img = None
        self.host = str(os.environ['VEHICLE_NAME'])
        self.feet_to_meters = 0.3048
        # TODO: add your subsribers or publishers here
        self.image_sub = rospy.Subscriber("/%s/camera_node/image/compressed"%self.host, CompressedImage, self.camera_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",CompressedImage,queue_size=1)

        # TODO: add information about tags
        TAG_SIZE = .08
        FAMILIES = "tagStandard41h12"
        self.tags = Tag(TAG_SIZE, FAMILIES)

        # Add information about tag locations
        # Function Arguments are id, x, y, z, theta_x, theta_y, theta_z (euler) 
        # for example, self.tags.add_tag( ... 

        feet_to_meters = 0.3048

        """
        self.tags.add_tag(0, 0, 0, 1*feet_to_meters, 0, 3*np.pi/2, 0)
        self.tags.add_tag(1, 1*feet_to_meters, 0, 2*feet_to_meters, 0, 0, 0)
        self.tags.add_tag(2, 2*feet_to_meters, 0, 1*feet_to_meters, 0, np.pi/2, 0)
        self.tags.add_tag(3, 1*feet_to_meters, 0, 0, 0, np.pi, 0)
        """
        """
        self.tags.add_tag(0, 0, 0, 1*feet_to_meters, 0, np.pi/2, 0)
        self.tags.add_tag(1, 1*feet_to_meters, 0, 2*feet_to_meters, 0, 0, 0)
        self.tags.add_tag(2, 2*feet_to_meters, 0, 1*feet_to_meters, 0, -1*np.pi/2, 0)
        self.tags.add_tag(3, 1*feet_to_meters, 0, 0, 0, -1*np.pi, 0)
        """
        """
        self.tags.add_tag(0, 0, 0.4*feet_to_meters, 1*feet_to_meters, 0, np.pi/2, 0)
        self.tags.add_tag(1, 1*feet_to_meters, 0.4*feet_to_meters, 2*feet_to_meters, 0, 0, 0)
        self.tags.add_tag(2, 2*feet_to_meters, 0.4*feet_to_meters, 1*feet_to_meters, 0, -1*np.pi/2, 0)
        self.tags.add_tag(3, 1*feet_to_meters, 0.4*feet_to_meters, 0, 0, -1*np.pi, 0)
        """
        height = 0.1065
        self.tags.add_tag(0, 0, height, 1*feet_to_meters, 0, np.pi/2, 0)
        self.tags.add_tag(1, 1*feet_to_meters, height, 2*feet_to_meters, 0, 0, 0)
        self.tags.add_tag(2, 2*feet_to_meters, height, 1*feet_to_meters, 0, 3*np.pi/2, 0)
        self.tags.add_tag(3, 1*feet_to_meters, height, 0, 0, np.pi, 0)
        

        # Load camera parameters
        # TODO: change with your robots name
        with open("/data/config/calibrations/camera_intrinsic/%s.yaml"% (self.host)) as file:

        #with open("/data/config/calibrations/camera_intrinsic/MYROBOT.yaml") as file:
                camera_list = yaml.load(file,Loader = yaml.FullLoader)

        self.camera_intrinsic_matrix = np.array(camera_list['camera_matrix']['data']).reshape(3,3)
        self.distortion_coeff = np.array(camera_list['distortion_coefficients']['data']).reshape(5,1)

    def camera_callback(self, data):

        #np_arr = np.fromstring(data.data, np.uint8)
        np_arr = np.frombuffer(data.data, np.uint8)
        #print('-----------------', np_arr)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image_gray =  cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        undistort_img = self.undistort(cv_image_gray)
        
        self.undistort_img = undistort_img
        #self.undistort_img = cv_image_gray

    def undistort(self, img):
        '''
        Takes a fisheye-distorted image and undistorts it

        Adapted from: https://github.com/asvath/SLAMDuck
        '''
        height = img.shape[0]
        width = img.shape[1]

        newmatrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_intrinsic_matrix,
            self.distortion_coeff, 
            (width, height),
            1, 
            (width, height))

        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_intrinsic_matrix, 
            self.distortion_coeff,  
            np.eye(3), 
            newmatrix, 
            (width, height), 
            cv2.CV_16SC2)

        undistorted_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
       
        return undistorted_image   
             

    def detect(self, img):
        '''
        Takes an images and detects AprilTags
        '''
        PARAMS = [
            self.camera_intrinsic_matrix[0,0],
            self.camera_intrinsic_matrix[1,1],
            self.camera_intrinsic_matrix[0,2],
            self.camera_intrinsic_matrix[1,2]] 


        TAG_SIZE = 0.08 
        detector = Detector(families="tagStandard41h12", nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
        detected_tags = detector.detect(
            img, 
            estimate_tag_pose=True, 
            camera_params=PARAMS, 
            tag_size=TAG_SIZE)

        return detected_tags


    def localize(self):
        if self.undistort_img is None:
            pass
        else:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', self.undistort_img)[1]).tostring()
            self.image_pub.publish(msg)

            detected_tag = self.detect(self.undistort_img)
            #print("what is detected tag:",detected_tag)
            print('..............................')

            #print('$$$$$$$$$$$$$$$$$$$$$$$$', detected_tag)
            
            if detected_tag != []:
                total_angles_in_degress = 0.0
                my_tags = []
                finalGlobalFrame = 0.0
                for eachTag in detected_tag:
                    print("detected_tag:",eachTag.tag_id)

                    detected_tag_id = eachTag.tag_id
                    my_tags.append(detected_tag_id)
                    tag_position = self.tags.locations[detected_tag_id] ###???
                    #print("tag_position:",tag_position)

                    pose_R = np.array(eachTag.pose_R)
                    pose_t = np.array(eachTag.pose_t)
                    #print("pose_R:",pose_R)
                    print("pose_t:",pose_t)
                    
                    camera_R_tag = pose_R.T         
                    camera_t_tag = -1 * pose_t  

                    #print("camera_R_tag:",camera_R_tag)
                    #print("camera_t_tag:",camera_t_tag)

                    #### calculate the final camera location
                    camera_position = camera_R_tag.dot(camera_t_tag)
                    #print(camera_position)


                    P = np.array([[1,0,0],
                                [0,-1,0],
                                [0,0,1]])


                    #position_unofficial = P.dot(tag_position)
                    position_unofficial = P @ camera_position


                    R_g = self.tags.orientations[detected_tag_id]
                    t_g = self.tags.locations[detected_tag_id]
                    

                    position_global = R_g @ position_unofficial + t_g

                    #print("global frame:",position_global)
                    finalGlobalFrame += position_global


                    angleMatrix = R_g @ camera_R_tag
                    #angleMatrix = camera_R_tag @ R_g

                    theta_x = np.arctan2(angleMatrix[2][1],angleMatrix[2][2])
                    theta_y = np.arctan2(-1*angleMatrix[2][0], np.sqrt(angleMatrix[2][1]**2+angleMatrix[2][2]**2))
                    theta_z = np.arctan2(angleMatrix[1][0], angleMatrix[0][0])


                    euler_angles = rotationMatrixToEulerAngles(angleMatrix)
                    print("angles in radian: ", euler_angles)

                    euler_angles_in_degree = euler_angles * 180/np.pi
                    #euler_angles_in_degree = euler_angles.as_euler('zyx', degrees=True)
                    if detected_tag_id == 3:
                        print('Tag 3')
                        """
                        if euler_angles_in_degree[1] > 0:
                            euler_angles_in_degree[1] += 180
                        """
                    if detected_tag_id ==3:
                        print("---------3-----------:\n",)
                        robot_euler_angles = rotationMatrixToEulerAngles(pose_R)
                        robot_euler_angles_in_degree = robot_euler_angles * 180/np.pi
                        print(robot_euler_angles_in_degree)
                        if robot_euler_angles_in_degree[1]<0: ##robot observe april tag from left
                            euler_angles_in_degree[1] = 180-euler_angles_in_degree[1]
                        if robot_euler_angles_in_degree[1]>=0: #robot observe april tag from right
                            euler_angles_in_degree[1] = 180-euler_angles_in_degree[1] 
                    if detected_tag_id ==2:
                        print("---------2-----------:\n",)
                        robot_euler_angles = rotationMatrixToEulerAngles(pose_R)
                        robot_euler_angles_in_degree = robot_euler_angles * 180/np.pi
                        print(robot_euler_angles_in_degree)
                        if robot_euler_angles_in_degree[1]<0:
                            euler_angles_in_degree[1] = 180-euler_angles_in_degree[1]
                    if detected_tag_id == 0:
                        print("---------0----2-------:\n",)
                        robot_euler_angles = rotationMatrixToEulerAngles(pose_R)
                        robot_euler_angles_in_degree = robot_euler_angles * 180/np.pi
                        print(robot_euler_angles_in_degree)
                        if robot_euler_angles_in_degree[1]>=0:
                            euler_angles_in_degree[1] = 180-euler_angles_in_degree[1]

                    
                    total_angles_in_degress += euler_angles_in_degree

                if len(detected_tag) > 0:
                    print("tags found:",my_tags)
                    print('*******Final Position*******\n', finalGlobalFrame/len(detected_tag))
                    print('*******Final Position in Feet*******\n', finalGlobalFrame/len(detected_tag)/self.feet_to_meters)
                    print("*******Final angles:\n",total_angles_in_degress/len(detected_tag))



    def run(self):
        # publish message every 1 second
        rate = rospy.Rate(20) # 40Hz
        print("I m runing---------------------------------------------------------------")
        #self.localize()
        #rate.sleep()
        while not rospy.is_shutdown():
            self.localize()
            rate.sleep()
        print('11111111111111111')
# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def main():
    print("Lab5 Start")
    #time.sleep(5)

    try:
        my_node = MyNode(node_name = "location_finder")
        my_node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Lab5 terminated.")
if __name__ == '__main__':
    main()
