import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    slam_params_file = os.path.join(
        get_package_share_directory('slam_toolbox'),
        'config',
        'mapper_params_online_async.yaml'
    )

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value="/opt/ros/lyrical/share/turtlebot3_gazebo/models",
    )

    # Added '-s' for headless server mode over SSH
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', 'new_multi_robot.sdf'],
        output='screen',
    )

    bridge_topics = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

        # Burger 1
        '/burger_1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/burger_1/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        '/burger_1/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        '/burger_1/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/burger_1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/burger_1/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        '/burger_1/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',

        # Burger 2
        '/burger_2/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/burger_2/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        '/burger_2/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        '/burger_2/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/burger_2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/burger_2/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        '/burger_2/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
    ]

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_topics,
        output='screen',
    )

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # --- Burger 1 SLAM ---
    slam_node_1 = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='burger_1',
        output='screen',
        parameters=[
            slam_params_file,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': '/burger_1/scan',
                'odom_frame': 'burger_1/odom',
                'base_frame': 'burger_1/base_footprint',
                'map_frame': 'burger_1/map',

                # Publish occupancy grid more frequently
                'map_update_interval': 1.0,

                # Process every scan
                'throttle_scans': 1,
                'minimum_time_interval': 0.1,
            }
        ],
        remappings=[
            ('/tf', '/burger_1/tf'),
            ('/tf_static', '/burger_1/tf_static'),
            ('/map', '/burger_1/map'),
            ('/map_metadata', '/burger_1/map_metadata'),
        ]
    )

    configure_slam_1 = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_node_1),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    activate_slam_1 = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node_1,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(slam_node_1),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )

    # --- Burger 2 SLAM ---
    slam_node_2 = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='burger_2',
        output='screen',
        parameters=[
            slam_params_file,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': '/burger_2/scan',
                'odom_frame': 'burger_2/odom',
                'base_frame': 'burger_2/base_footprint',
                'map_frame': 'burger_2/map',

                # Publish occupancy grid more frequently
                'map_update_interval': 1.0,

                # Process every scan
                'throttle_scans': 1,
                'minimum_time_interval': 0.1,
            }
        ],
        remappings=[
            ('/tf', '/burger_2/tf'),
            ('/tf_static', '/burger_2/tf_static'),
            ('/map', '/burger_2/map'),
            ('/map_metadata', '/burger_2/map_metadata'),
        ]
    )

    configure_slam_2 = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_node_2),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    activate_slam_2 = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node_2,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(slam_node_2),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )

    burger1_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='burger1_base_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0.010',
            '--frame-id', 'burger_1/base_footprint',
            '--child-frame-id', 'burger_1/base_link'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf_static', '/burger_1/tf_static')
        ]
    )

    burger1_lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='burger1_lidar_tf',
        arguments=[
            '--x', '-0.032', '--y', '0', '--z', '0.171',
            '--frame-id', 'burger_1/base_link',
            '--child-frame-id', 'burger_1/base_scan'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf_static', '/burger_1/tf_static')
        ]      
    )

    burger1_imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='burger1_imu_tf',
        arguments=[
            '--x', '-0.032', '--y', '0', '--z', '0.068',
            '--frame-id', 'burger_1/base_link',
            '--child-frame-id', 'burger_1/imu_link'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf_static', '/burger_1/tf_static')
        ]
    )

    burger2_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='burger2_base_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0.010',
            '--frame-id', 'burger_2/base_footprint',
            '--child-frame-id', 'burger_2/base_link'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf_static', '/burger_2/tf_static')
        ]
    )

    burger2_lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='burger2_lidar_tf',
        arguments=[
            '--x', '-0.032', '--y', '0', '--z', '0.171',
            '--frame-id', 'burger_2/base_link',
            '--child-frame-id', 'burger_2/base_scan'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf_static', '/burger_2/tf_static')
        ]      
    )

    burger2_imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='burger2_imu_tf',
        arguments=[
            '--x', '-0.032', '--y', '0', '--z', '0.068',
            '--frame-id', 'burger_2/base_link',
            '--child-frame-id', 'burger_2/imu_link'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf_static', '/burger_2/tf_static')
        ]
    )

    map_merge = Node(
            package='map_merge_features',
            executable='map_merge_features_node',
            name='map_merge',
            output='screen'
        )

    return LaunchDescription([
        gz_resource_path,
        gz_sim,
        bridge_node,
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),
        burger1_base_tf,
        burger1_lidar_tf,
        burger1_imu_tf,
        burger2_base_tf,
        burger2_lidar_tf,
        burger2_imu_tf,
        slam_node_1,
        configure_slam_1,
        activate_slam_1,
        slam_node_2,
        configure_slam_2,
        activate_slam_2,
        map_merge
    ])
