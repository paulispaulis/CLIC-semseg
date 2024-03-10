#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node('image_segmentation_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher("/camera/image_segmented", Image, queue_size=1)
        
        # Variable to store the time of the last processed frame
        self.last_processed_frame_time = rospy.Time.now()

    def image_callback(self, data):
        try:
            # Convert the ROS image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        # Simulated image processing (e.g., segmentation)
        # This is where you'd integrate your segmentation software
        processed_image = self.simulate_image_processing(cv_image)
        
        try:
            # Convert the CV2 image back to a ROS image message
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, "bgr8"))
        except CvBridgeError as e:
            print(e)
    
    def simulate_image_processing(self, image):
        # Dummy "processing" for demonstration
        # For real applications, replace this with the call to your segmentation software
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return processed_image

if __name__ == '__main__':
    try:
        ImageSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
