import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np
from rclpy.executors import ExternalShutdownException
import math

class MapMergeFeaturesNode(Node):
    def __init__(self):
        super().__init__('map_merge_features_node')
        self.map1 = None
        self.map2 = None

        self.metadata1 = None
        self.metadata2 = None

        self.map1_ready = False
        self.map2_ready = False

        self.global_map = None
        self.global_known = None

        self.common_start_x = 0
        self.common_start_y = 0
        self.common_height = 0
        self.common_width = 0

        self.threshold = 0.99

        self.min_matches = 8
        self.min_inliers = 6
        self.min_inlier_ratio = 0.30

        self.ratio_threshold = 0.75

        self.ransac_threshold = 3.0

        self.min_scale = 0.90
        self.max_scale = 1.10

        self.max_rotation = 25.0

        self.min_omega = 0.70
        self.min_overlap_cells = 100

        self.kaze = cv2.KAZE_create(
            extended=False,
            upright=False,
            threshold=0.001,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )

        self.sift = cv2.SIFT_create()

        # FLANN matcher
        FLANN_INDEX_KDTREE = 1

        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=5
        )

        search_params = dict(checks=100)

        self.matcher = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

        self.global_map_pub = self.create_publisher(
            OccupancyGrid,
            '/global_map',
            10
        )

        self.map1_sub = self.create_subscription(
            OccupancyGrid,
            '/burger_1/map',
            self.map1_callback,
            10
        )

        self.map2_sub = self.create_subscription(
            OccupancyGrid,
            '/burger_2/map',
            self.map2_callback,
            10
        )

        self.get_logger().info(
            'Map Merge Features Node initialized.')

    # ==============================================
    # MAP CALLBACKS
    # ==============================================

    def map1_callback(self, msg):

        self.map1 = msg
        self.metadata1 = msg
        self.map1_ready = True

        self.try_merge()

    def map2_callback(self, msg):

        self.map2 = msg
        self.metadata2 = msg
        self.map2_ready = True

        self.try_merge()

    # ==============================================
    # VALIDATE INPUT MAPS
    # ==============================================

    def validate_maps(self):

        if not self.map1_ready or not self.map2_ready:
            return False

        if self.map1.info.resolution <= 0:
            return False

        if self.map2.info.resolution <= 0:
            return False

        if abs(
            self.map1.info.resolution -
            self.map2.info.resolution
        ) > 1e-5:

            self.get_logger().warning(
                'Map resolutions differ. Cannot merge.'
            )

            return False

        # Check map-origin rotations
        for msg in (self.map1, self.map2):

            q = msg.info.origin.orientation

            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y*q.y + q.z*q.z)
            )

            if abs(yaw) > 1e-3:

                self.get_logger().warning(
                    'Rotated map origins are not supported.'
                )

                return False

        return True

    # ==============================================
    # OCCUPANCY GRID CONVERSION
    # ==============================================

    def occupancy_to_probability(self, msg):

        raw = np.array(
            msg.data,
            dtype=np.int16
        ).reshape(
            (msg.info.height, msg.info.width)
        )

        known = raw >= 0

        probability = np.full(
            raw.shape,
            0.5,
            dtype=np.float32
        )

        probability[known] = (
            raw[known].astype(np.float32) / 100.0
        )

        # ROS OccupancyGrid starts at bottom-left.
        # OpenCV images start at top-left.

        probability = np.flipud(probability)
        known = np.flipud(known)

        return probability, known

    # ==============================================
    # MATLAB threshold_grid()
    # ==============================================

    def threshold_grid(self, probability):

        # MATLAB:
        # p_free = 1 - p_occ
        # binaryMapImage = 1 if p_free >= threshold

        p_free = 1.0 - probability

        binary_image = np.zeros(
            probability.shape,
            dtype=np.uint8
        )

        binary_image[
            p_free >= self.threshold
        ] = 255

        return p_free, binary_image

    # ==============================================
    # MATLAB local_to_global()
    # ==============================================

    def prepare_common_maps(self):

        p1, known1 = self.occupancy_to_probability(
            self.map1
        )

        p2, known2 = self.occupancy_to_probability(
            self.map2
        )

        _, img1 = self.threshold_grid(p1)
        _, img2 = self.threshold_grid(p2)

        h1, w1 = p1.shape
        h2, w2 = p2.shape

        # Create a common canvas with room around
        # both maps.

        max_h = max(h1, h2)
        max_w = max(w1, w2)

        canvas_h = max_h + 2 * math.ceil(max_h / 4)
        canvas_w = max_w + 2 * math.ceil(max_w / 4)

        # Center each map independently.
        start_y1 = (canvas_h - h1) // 2
        start_x1 = (canvas_w - w1) // 2

        start_y2 = (canvas_h - h2) // 2
        start_x2 = (canvas_w - w2) // 2

        # Probability canvases
        global_p1 = np.full(
            (canvas_h, canvas_w),
            0.5,
            dtype=np.float32
        )

        global_p2 = np.full(
            (canvas_h, canvas_w),
            0.5,
            dtype=np.float32
        )

        # Known-cell masks
        global_known1 = np.zeros(
            (canvas_h, canvas_w),
            dtype=bool
        )

        global_known2 = np.zeros(
            (canvas_h, canvas_w),
            dtype=bool
        )

        # Binary feature images
        global_img1 = np.zeros(
            (canvas_h, canvas_w),
            dtype=np.uint8
        )

        global_img2 = np.zeros(
            (canvas_h, canvas_w),
            dtype=np.uint8
        )

        # Insert map 1
        global_p1[
            start_y1:start_y1+h1,
            start_x1:start_x1+w1
        ] = p1

        global_known1[
            start_y1:start_y1+h1,
            start_x1:start_x1+w1
        ] = known1

        global_img1[
            start_y1:start_y1+h1,
            start_x1:start_x1+w1
        ] = img1

        # Insert map 2
        global_p2[
            start_y2:start_y2+h2,
            start_x2:start_x2+w2
        ] = p2

        global_known2[
            start_y2:start_y2+h2,
            start_x2:start_x2+w2
        ] = known2

        global_img2[
            start_y2:start_y2+h2,
            start_x2:start_x2+w2
        ] = img2

        # Save common map information
        self.common_start_x = start_x1
        self.common_start_y = start_y1

        self.common_height = canvas_h
        self.common_width = canvas_w

        return (
            global_p1,
            global_p2,
            global_known1,
            global_known2,
            global_img1,
            global_img2
        )

    # ==============================================
    # KAZE DETECTION + SIFT DESCRIPTORS
    # ==============================================

    def extract_features(self, image):

        keypoints = self.kaze.detect(
            image,
            None
        )

        if len(keypoints) < self.min_matches:
            return None, None

        keypoints, descriptors = self.sift.compute(
            image,
            keypoints
        )

        return keypoints, descriptors

    # ==============================================
    # FEATURE REGISTRATION
    # ==============================================

    def register_maps(self, img1, img2):

        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)

        if desc1 is None or desc2 is None:
            return None

        if len(desc1) < 2 or len(desc2) < 2:
            return None

        # Match moving descriptors against fixed descriptors.
        # queryIdx = moving
        # trainIdx = fixed

        try:

            matches = self.matcher.knnMatch(
                desc2,
                desc1,
                k=2
            )

        except cv2.error:

            return None

        # Lowe ratio test
        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        self.get_logger().info(
            f'Feature matches: {len(good_matches)}'
        )

        if len(good_matches) < self.min_matches:
            return None

        # Moving -> Fixed
        src_pts = np.float32([
            kp2[m.queryIdx].pt
            for m in good_matches
        ])

        dst_pts = np.float32([
            kp1[m.trainIdx].pt
            for m in good_matches
        ])

        # Similarity transformation using RANSAC
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=3000,
            confidence=0.995,
            refineIters=10
        )

        if transform is None or inlier_mask is None:
            return None

        inlier_mask = inlier_mask.ravel().astype(bool)

        num_inliers = int(np.sum(inlier_mask))

        inlier_ratio = (
            num_inliers / len(good_matches)
        )

        # Minimum inlier validation
        if num_inliers < self.min_inliers:

            self.get_logger().warning(
                f'Rejected: insufficient inliers ({num_inliers})'
            )

            return None

        if inlier_ratio < self.min_inlier_ratio:

            self.get_logger().warning(
                f'Rejected: low inlier ratio ({inlier_ratio:.3f})'
            )

            return None

        # Check inlier spatial distribution
        inlier_src = src_pts[inlier_mask]
        inlier_dst = dst_pts[inlier_mask]

        def spatial_coverage(points, shape):

            hull = cv2.convexHull(
                points.astype(np.float32)
            )

            area = cv2.contourArea(hull)

            return area / (shape[0] * shape[1])

        coverage1 = spatial_coverage(
            inlier_dst,
            img1.shape
        )

        coverage2 = spatial_coverage(
            inlier_src,
            img2.shape
        )

        if min(coverage1, coverage2) < 0.001:

            self.get_logger().warning(
                'Rejected: inliers are spatially concentrated.'
            )

            return None

        # Check scale and rotation
        a = transform[0, 0]
        b = transform[1, 0]

        scale = math.sqrt(a*a + b*b)

        rotation = math.degrees(
            math.atan2(b, a)
        )

        if not self.min_scale <= scale <= self.max_scale:

            self.get_logger().warning(
                f'Rejected: invalid scale ({scale:.3f})'
            )

            return None

        if abs(rotation) > self.max_rotation:

            self.get_logger().warning(
                f'Rejected: excessive rotation ({rotation:.2f})'
            )

            return None

        self.get_logger().info(
            f'Registration accepted: '
            f'inliers={num_inliers}, '
            f'ratio={inlier_ratio:.3f}, '
            f'scale={scale:.3f}, '
            f'rotation={rotation:.2f}'
        )

        return transform

    # ==============================================
    # EXPANDED MAP TRANSFORMATION
    # ==============================================

    def transform_maps(
        self,
        p1,
        p2,
        known1,
        known2,
        transform
    ):

        h, w = p1.shape
        hm, wm = p2.shape

        # Moving map corners
        corners = np.float32([
            [0, 0],
            [wm - 1, 0],
            [0, hm - 1],
            [wm - 1, hm - 1]
        ]).reshape(-1, 1, 2)

        transformed_corners = cv2.transform(
            corners,
            transform
        ).reshape(-1, 2)

        # Determine bounds of both maps
        min_x = min(
            0,
            int(np.floor(transformed_corners[:, 0].min()))
        )

        min_y = min(
            0,
            int(np.floor(transformed_corners[:, 1].min()))
        )

        max_x = max(
            w - 1,
            int(np.ceil(transformed_corners[:, 0].max()))
        )

        max_y = max(
            h - 1,
            int(np.ceil(transformed_corners[:, 1].max()))
        )

        out_w = max_x - min_x + 1
        out_h = max_y - min_y + 1

        if out_w <= 0 or out_h <= 0:
            return None

        if out_w * out_h > 100_000_000:

            self.get_logger().warning(
                'Rejected: transformed map is too large.'
            )

            return None

        # Shift maps to expanded output canvas
        shift = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y]
        ], dtype=np.float64)

        # Convert 2x3 affine transformations to 3x3
        # homogeneous transformation matrices

        shift_h = np.vstack([
            shift,
            [0.0, 0.0, 1.0]
        ])

        transform_h = np.vstack([
            transform,
            [0.0, 0.0, 1.0]
        ])

        # Fixed map transformation
        fixed_transform = shift_h[:2, :]

        # Moving map transformation
        moving_transform = (shift_h @ transform_h)[:2, :]

        # Warp fixed probabilities
        fixed_canvas = cv2.warpAffine(
            p1,
            fixed_transform,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.5
        )

        fixed_known = cv2.warpAffine(
            known1.astype(np.uint8),
            fixed_transform,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ).astype(bool)

        # Warp moving probabilities
        moving_canvas = cv2.warpAffine(
            p2,
            moving_transform,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.5
        )

        moving_known = cv2.warpAffine(
            known2.astype(np.uint8),
            moving_transform,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ).astype(bool)

        return (
            fixed_canvas,
            moving_canvas,
            fixed_known,
            moving_known,
            min_x,
            min_y,
            max_x,
            max_y
        )

    # ==============================================
    # REGISTRATION AGREEMENT VALIDATION
    # ==============================================

    def validate_alignment(
        self,
        p1,
        p2,
        known1,
        known2
    ):

        both_known = known1 & known2

        overlap = int(np.sum(both_known))

        if overlap < self.min_overlap_cells:

            self.get_logger().warning(
                f'Rejected: insufficient known overlap ({overlap})'
            )

            return False

        # Classify occupied and free cells
        occ1 = p1 > 0.65
        occ2 = p2 > 0.65

        free1 = p1 < 0.35
        free2 = p2 < 0.35

        agreement = np.sum(
            both_known & (
                (occ1 & occ2) |
                (free1 & free2)
            )
        )

        disagreement = np.sum(
            both_known & (
                (occ1 & free2) |
                (free1 & occ2)
            )
        )

        total = agreement + disagreement

        if total == 0:
            return False

        omega = agreement / total

        self.get_logger().info(
            f'Alignment validation: '
            f'overlap={overlap}, omega={omega:.3f}'
        )

        if omega < self.min_omega:

            self.get_logger().warning(
                f'Rejected: low alignment agreement ({omega:.3f})'
            )

            return False

        return True

    # ==============================================
    # BAYESIAN GRID FUSION
    # ==============================================

    def bayesian_fusion(
        self,
        fixed,
        moving,
        fixed_known,
        moving_known
    ):

        # Cells known by both maps
        both = fixed_known & moving_known

        # Cells known by only one map
        only_fixed = fixed_known & ~moving_known
        only_moving = ~fixed_known & moving_known

        result = np.full(
            fixed.shape,
            0.5,
            dtype=np.float32
        )

        # Preserve individual known observations
        result[only_fixed] = fixed[only_fixed]
        result[only_moving] = moving[only_moving]

        # MATLAB Bayesian equation
        p1 = fixed[both]
        p2 = moving[both]

        numerator = p1 * p2

        denominator = (
            p1 * p2 +
            (1.0 - p1) * (1.0 - p2)
        )

        valid = denominator > 1e-8

        fused = np.full(
            p1.shape,
            0.5,
            dtype=np.float32
        )

        fused[valid] = (
            numerator[valid] / denominator[valid]
        )

        result[both] = fused

        result_known = fixed_known | moving_known

        return result, result_known

    # ==============================================
    # PUBLISH GLOBAL MAP
    # ==============================================

    def publish_global_map(
        self,
        probability,
        known,
        min_x,
        max_y
    ):

        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()

        # Use Robot 1's map frame
        msg.header.frame_id = self.metadata1.header.frame_id

        resolution = self.metadata1.info.resolution

        height, width = probability.shape

        msg.info.resolution = resolution
        msg.info.width = width
        msg.info.height = height

        # Calculate output origin.
        # Assumes zero map-origin yaw.

        msg.info.origin.position.x = (
            self.metadata1.info.origin.position.x
            + (self.common_start_x + min_x) * resolution
        )

        msg.info.origin.position.y = (
            self.metadata1.info.origin.position.y
            + (
                self.common_start_y
                + self.map1.info.height
                - 1
                - max_y
            ) * resolution
        )

        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Convert probability grid to ROS occupancy grid
        occupancy = np.full(
            probability.shape,
            -1,
            dtype=np.int8
        )

        occupancy[known] = np.clip(
            probability[known] * 100.0,
            0,
            100
        ).astype(np.int8)

        # OpenCV top-to-bottom -> ROS bottom-to-top
        occupancy = np.flipud(occupancy)

        msg.data = occupancy.flatten().tolist()

        self.global_map = probability
        self.global_known = known

        self.global_map_pub.publish(msg)

        self.get_logger().info(
            f'Published global map: {width} x {height}'
        )

    # ==============================================
    # MAIN MERGE PIPELINE
    # ==============================================

    def try_merge(self):

        if not self.validate_maps():
            return

        self.merge()

    def merge(self):

        # ------------------------------------------
        # 1. Prepare common maps
        # ------------------------------------------

        (
            p1,
            p2,
            known1,
            known2,
            img1,
            img2
        ) = self.prepare_common_maps()

        # ------------------------------------------
        # 2. Feature registration
        # ------------------------------------------

        transform = self.register_maps(
            img1,
            img2
        )

        if transform is None:

            self.get_logger().warning(
                'No reliable feature registration found.'
            )

            return

        # ------------------------------------------
        # 3. Transform maps
        # ------------------------------------------

        transformed = self.transform_maps(
            p1,
            p2,
            known1,
            known2,
            transform
        )

        if transformed is None:
            return

        (
            fixed_canvas,
            moving_canvas,
            fixed_known,
            moving_known,
            min_x,
            min_y,
            max_x,
            max_y
        ) = transformed

        # ------------------------------------------
        # 4. Validate alignment
        # ------------------------------------------

        accepted = self.validate_alignment(
            fixed_canvas,
            moving_canvas,
            fixed_known,
            moving_known
        )

        if not accepted:
            return

        # ------------------------------------------
        # 5. Bayesian fusion
        # ------------------------------------------

        merged, merged_known = self.bayesian_fusion(
            fixed_canvas,
            moving_canvas,
            fixed_known,
            moving_known
        )

        # ------------------------------------------
        # 6. Publish
        # ------------------------------------------

        self.publish_global_map(
            merged,
            merged_known,
            min_x,
            max_y
        )

        self.get_logger().info(
            'Map fusion completed successfully.'
        )


# ==============================================
# MAIN
# ==============================================

def main(args=None):

    rclpy.init(args=args)

    node = MapMergeFeaturesNode()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
