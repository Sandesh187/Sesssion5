import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped


class WaypointPatrol(Node):

    def __init__(self):
        super().__init__("waypoint_patrol")

        self.client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose"
        )

        self.get_logger().info("Waiting for Nav2...")

        self.client.wait_for_server()

        self.get_logger().info("Nav2 Ready!")

        self.waypoints = [
            (1.04,2.30, 0.0),
            (-1.45, 2.40, 0.0),
            (-1.94, -1.29, 0.0)
        ]

        self.current_index = 0

        self.send_next_goal()

    def make_pose(self, x, y, yaw):

        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        return pose

    def send_next_goal(self):

        if self.current_index >= len(self.waypoints):

            self.get_logger().info("Mission Complete!")
            rclpy.shutdown()
            return

        x, y, yaw = self.waypoints[self.current_index]

        self.get_logger().info(
            f"Navigating to Waypoint {self.current_index + 1}..."
        )

        goal = NavigateToPose.Goal()
        goal.pose = self.make_pose(x, y, yaw)

        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):

        self.goal_handle = future.result()

        if not self.goal_handle.accepted:

            self.get_logger().error("Goal Rejected")
            rclpy.shutdown()
            return

        result_future = self.goal_handle.get_result_async()

        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):

        distance = feedback_msg.feedback.distance_remaining

        self.get_logger().info(
            f"Remaining Distance : {distance:.2f} m"
        )

    def result_callback(self, future):

        result = future.result()

        if result.status == 4:

            self.get_logger().info(
                f"Waypoint {self.current_index + 1} Reached!"
            )

            self.current_index += 1

            self.send_next_goal()

        else:

            self.get_logger().error(
                f"Navigation Failed at Waypoint {self.current_index + 1}"
            )

            rclpy.shutdown()


def main():

    rclpy.init()

    node = WaypointPatrol()

    rclpy.spin(node)


if __name__ == "__main__":
    main()