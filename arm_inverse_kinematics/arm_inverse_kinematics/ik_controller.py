

import argparse
import math
import sys
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ---- Link dimensions (from URDF / robot_dimensions.py) ----
H  = 0.075   # shoulder_height: base_link -> shoulder_lift axis
L1 = 0.20    # upper_arm: shoulder_lift -> elbow
L2 = 0.25    # forearm: elbow -> wrist
L3 = 0.175   # wrist -> end_effector

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_joint",
]

JOINT_LIMITS = {
    "shoulder_pan_joint":  (-3.14, 3.14),
    "shoulder_lift_joint": (-1.5708, 1.5708),
    "elbow_joint":         (-2.3562, 2.3562),
    "wrist_joint":         (-2.3562, 2.3562),
}


def solve_ik(x, y, z, phi_deg=180.0, elbow_up=False):
    """
    Analytic IK for the 4-DOF arm. See Q2_IK_Derivation.pdf for the full
    derivation. Returns (theta1, theta2, theta3, theta4) in radians.

    phi_deg: desired end-effector pitch from vertical (0=up, 180=straight down).
             180 is used by default since the arm grasps objects on a table.
    elbow_up: choose between the two elbow solution branches.
    """
    # Step 1: base yaw
    theta1 = math.atan2(y, x)

    r = math.hypot(x, y)
    z_prime = z - H

    # Step 2: desired approach angle
    phi = math.radians(phi_deg)

    # Step 3: wrist-center point (back off L3 along the approach direction)
    r_w = r - L3 * math.sin(phi)
    z_w = z_prime - L3 * math.cos(phi)

    d = math.hypot(r_w, z_w)
    if d > (L1 + L2) or d < abs(L1 - L2):
        raise ValueError(
            f"Target unreachable: wrist-center distance d={d:.3f} m, "
            f"reachable range is [{abs(L1 - L2):.3f}, {L1 + L2:.3f}] m"
        )

    # Step 4: law of cosines for the elbow
    cos_t3 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t3 = max(-1.0, min(1.0, cos_t3))  # clamp for float safety
    theta3 = math.acos(cos_t3)
    if elbow_up:
        theta3 = -theta3

    theta2 = math.atan2(r_w, z_w) - math.atan2(
        L2 * math.sin(theta3), L1 + L2 * math.cos(theta3)
    )

    # Step 5: wrist makes up the remaining pitch
    theta4 = phi - theta2 - theta3

    return theta1, theta2, theta3, theta4


def check_limits(angles):
    for name, val in zip(JOINT_NAMES, angles):
        lo, hi = JOINT_LIMITS[name]
        if not (lo <= val <= hi):
            raise ValueError(
                f"{name} = {math.degrees(val):.2f} deg is outside limits "
                f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg"
            )


# Approach angles to try, in preferred order: straight-down first (best for
# grasping objects on a table), then progressively shallower. Each is tried
# with both elbow branches.
AUTO_PHI_CANDIDATES = [180, 160, 140, 120, 100, 90, 70, 50, 30, 10]


def auto_solve_ik(x, y, z):
    """
    Tries a range of approach angles / elbow branches and returns the first
    one that is both geometrically reachable and within joint limits.
    Returns (theta1, theta2, theta3, theta4, phi_used, elbow_up_used).
    """
    last_error = None
    for phi_deg in AUTO_PHI_CANDIDATES:
        for elbow_up in (False, True):
            try:
                angles = solve_ik(x, y, z, phi_deg=phi_deg, elbow_up=elbow_up)
                check_limits(angles)
                return (*angles, phi_deg, elbow_up)
            except ValueError as e:
                last_error = e
                continue
    raise ValueError(
        f"No reachable solution found for ({x}, {y}, {z}) across "
        f"phi in {AUTO_PHI_CANDIDATES} and both elbow branches. "
        f"Last error: {last_error}"
    )


class CustomIKNode(Node):
    def __init__(self, target_xyz, phi_deg=None, elbow_up=False, duration_sec=3.0):
        super().__init__('custom_ik_node')

        if phi_deg is None:
            # Auto-search for a reachable approach angle - no need to pass --phi.
            theta1, theta2, theta3, theta4, phi_used, elbow_used = auto_solve_ik(*target_xyz)
            angles = (theta1, theta2, theta3, theta4)
            self.get_logger().info(
                f"Auto-selected approach angle phi={phi_used} deg, elbow_up={elbow_used}"
            )
        else:
            angles = solve_ik(*target_xyz, phi_deg=phi_deg, elbow_up=elbow_up)
            check_limits(angles)

        self.angles = angles
        self.duration_sec = duration_sec

        self.get_logger().info(
            f"Target {target_xyz} -> joint angles (deg): "
            f"{[round(math.degrees(a), 2) for a in angles]}"
        )

        # Publish straight to the JointTrajectoryController's plain topic
        # interface. No action handshake / goal-acceptance validation here -
        # this is the "publish this message directly to the arm's hardware
        # controllers" interface the assignment describes, and it completely
        # bypasses MoveIt / move_group.
        self._pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )
        # give the publisher a moment to match with the controller's subscriber
        self.create_timer(1.0, self.publish_once)
        self._published = False

    def publish_once(self):
        if self._published:
            return
        self._published = True

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = list(self.angles)
        point.velocities = [0.0] * len(self.angles)
        point.time_from_start = Duration(sec=int(self.duration_sec))
        traj.points = [point]

        self.get_logger().info('Publishing trajectory to /arm_controller/joint_trajectory ...')
        self._pub.publish(traj)
        self.get_logger().info(
            f'Published. Arm should reach the target in ~{self.duration_sec:.1f}s.'
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Custom IK node - drive the arm to an (x,y,z) target")
    parser.add_argument('--x', type=float, default=0.30, help="target x in metres (base_link frame)")
    parser.add_argument('--y', type=float, default=0.15, help="target y in metres (base_link frame)")
    parser.add_argument('--z', type=float, default=0.05, help="target z in metres (base_link frame)")
    parser.add_argument('--phi', type=float, default=None,
                         help="end-effector pitch from vertical, deg (0=up, 180=straight down). "
                              "If omitted, the node auto-searches for a reachable angle.")
    parser.add_argument('--elbow-up', action='store_true', help="use the elbow-up solution branch")
    parser.add_argument('--duration', type=float, default=3.0, help="trajectory execution time, seconds")
    # rclpy passes its own remap args through sys.argv; strip them before argparse sees them
    return parser.parse_known_args(rclpy.utilities.remove_ros_args(sys.argv[1:]))[0]


def main(args=None):
    rclpy.init(args=args)
    parsed = parse_args()
    target = (parsed.x, parsed.y, parsed.z)

    node = CustomIKNode(
        target,
        phi_deg=parsed.phi,
        elbow_up=parsed.elbow_up,
        duration_sec=parsed.duration,
    )
    # Spin long enough for the publisher to connect (1s delay), publish,
    # and for the arm to finish executing the trajectory in Gazebo.
    import time
    end_time = time.time() + 1.0 + parsed.duration + 1.0
    while rclpy.ok() and time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info('Done. Shutting down.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()