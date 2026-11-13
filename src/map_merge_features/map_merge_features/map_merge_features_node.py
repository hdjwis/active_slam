import rclpy
from rclpy.node import Node
from  nav_msgs.msg import OccupancyGrid, MapMetaData
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from rclpy.executors import ExternalShutdownException

class MapMergeFeaturesNode(Node):
    def __init__(self):
        super().__init__('map_merge_features_node')
        self.global_map = None
        self.global_map_pub = self.create_publisher(OccupancyGrid, '/global_map', 10)
        self.map1_sub = self.create_subscription(OccupancyGrid, '/burger_1/map', self.map1_callback, 10)
        self.map2_sub = self.create_subscription(OccupancyGrid, '/burger_2/map', self.map2_callback, 10)
        self.map1 = None
        self.map2 = None
        self.map1_ready = False
        self.map2_ready = False
        self.metadata = None
        self.kaze = cv2.KAZE_create(
            extended=False,
            upright=False,
            threshold=0.001,
            nOctaves=4,          
            nOctaveLayers=4,     
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
    def convert(self, grid):
        img = np.full(grid.shape, 127, dtype=np.uint8)
        img[grid == 0] = 255 
        img[grid == 100] = 0 
        return img
        
    def publish_global_map(self):
        if self.global_map is None or self.metadata is None:
            return

        # Convert 0.0..1.0 probabilities back to ROS OccupancyGrid int8 (-1, 0..100)
        grid_int8 = np.full(self.global_map.shape, -1, dtype=np.int8)
        known = np.abs(self.global_map - 0.5) > 0.05
        grid_int8[known] = np.round(np.clip(self.global_map[known] * 100.0, 0, 100)).astype(np.int8)

        # Build the OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.metadata.header.frame_id
        msg.info = self.metadata.info
        msg.data = grid_int8.flatten().tolist()

        self.global_map_pub.publish(msg)


    def merge(self):
        self.get_logger().info("here merge")
        self.map1_ready = False
        self.map2_ready = False
        kp1, desc1 = self.kaze.detectAndCompute(self.map1, None)
        kp2, desc2 = self.kaze.detectAndCompute(self.map2, None)
        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            self.get_logger().info("here desc")
            return
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desc1,desc2,k=2)
        matches = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(matches) < 4:
            self.get_logger().info("here len")
            return
        self.get_logger().info("here merge")
        matches = sorted(matches, key = lambda x:x.distance)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        if M is None:
            self.get_logger().info("here M is None")
            return
        # check validtity of feature correspondence
        h, w = self.map1.shape
        self.get_logger().info("here merge")

        warped_map2 = cv2.warpPerspective(self.map2, M, (w, h), borderValue=127)
        # may need to pad matricies with 0.5 to make them align
        p1 = (255.0 - self.map1) /255.0
        p2 = (255.0 - warped_map2) / 255.0
        numerator = p1*p2 
        denominator = numerator + (1 - p1)*(1 - p2)
        self.global_map = numerator / np.maximum(denominator, 1e-6)
        self.publish_global_map()




    def map1_callback(self, msg):
        self.map1 = self.convert(np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width)))
        self.map1_ready = True
        self.metadata = msg
        if self.map1_ready and self.map2_ready:
            self.get_logger().info(f"map1 here and {str(self.map1_ready)}")
            self.merge()
    
    def map2_callback(self, msg):
        
        self.map2 = self.convert(np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width)))
        self.map2_ready = True
        self.get_logger().info(f"map2 here and {str(self.map2_ready)}")
        if self.map1_ready and self.map2_ready:
            self.merge()

def main() -> None:
    try:
        # 1. Initialize ROS 2 communications
        rclpy.init()
        
        # 2. Instantiate your custom node
        node = MapMergeFeaturesNode()
        
        # 3. Keep the node alive to process callbacks
        rclpy.spin(node)
        
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 4. Clean up the node and shut down ROS 2
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
