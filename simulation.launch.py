from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import SetEnvironmentVariable
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value="/opt/ros/lyrical/share/turtlebot3_gazebo/models",
    )

    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', 'multi_robot.sdf'],
        output='screen',
    )

    bridge_topics = [
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


    return LaunchDescription([
        gz_resource_path,
        gz_sim,
        bridge_node
        # front_camera_bridge,
    ])