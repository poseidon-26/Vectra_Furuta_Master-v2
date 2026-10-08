"""
ai_controller_node.py — Hybrid swing-up + SAC balancing controller for the Furuta pendulum.

Reads raw joint state directly from /joint_states (MuJoCo ros2_control output).
The EKF estimator node has been removed; this node is the single subscriber.

State machine:
  Stage 1  |th2_ai| >= 0.4 rad  → Åström-Furuta energy-pumping swing-up
  Stage 2  |th2_ai| <  0.4 rad  → SAC neural-network balancing

Angle conventions:
  th2_raw   : encoder reading; initial_value = -π means hanging down.
  th2_ai    : wrapped to [-π, π] — used by the neural network (0 = upright).
  th2_phys  : th2_raw + π          — used by energy formula (0 = hanging, π = upright).

IMPORTANT (Two-Clocks Problem):
  This node MUST be launched with use_sim_time:=True so that self.create_timer()
  fires in sync with the MuJoCo simulation clock, not wall-clock time.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import numpy as np
import torch

from models import SACActor


class FurutaRLController(Node):

    def __init__(self):
        super().__init__('furuta_rl_controller')
        self.get_logger().info("Initialising Hybrid Controller (direct joint_states)...")

        # ---- Load trained policy ----
        self.actor = SACActor(state_dim=4, action_dim=1, max_action=10.0)
        try:
            self.actor.load_state_dict(
                torch.load("furuta_actor.pth", weights_only=True)
            )
            self.actor.eval()
            self.get_logger().info("AI policy loaded successfully.")
        except Exception as exc:
            self.get_logger().error(f"FATAL: Cannot load weights — {exc}")
            raise SystemExit from exc

        # ---- ROS 2 interfaces ----
        # Subscribe directly to MuJoCo's /joint_states — no EKF middleman.
        self.state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10
        )
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 10
        )
        # 50 Hz control loop — must match dt=0.02 s in environment.py and MuJoCo timestep.
        self.timer = self.create_timer(0.02, self._control_loop)

        # Cached state; populated by _joint_state_callback.
        self._yaw       = 0.0
        self._pitch     = 0.0
        self._yaw_dot   = 0.0
        self._pitch_dot = 0.0
        # Stays False until the first message that includes velocity data arrives,
        # preventing the control loop from running on an incomplete state.
        self._state_ready = False

        # ---- Physical constants (must match environment.py) ----
        self.M_P      = 0.164     # kg    pendulum mass
        self.L_COM    = 0.137     # m     pendulum CoM distance from joint
        self.J_P      = 2.595e-4  # kg·m² pendulum inertia about swing axis (iyy)
        self.G        = 9.81      # m/s²
        self.E_TARGET = 1.1 * (2.0 * self.M_P * self.G * self.L_COM)

    # ------------------------------------------------------------------
    # /joint_states callback
    # ------------------------------------------------------------------

    def _joint_state_callback(self, msg: JointState):
        """
        Parse the JointState message published by mujoco_ros2_control.

        Safety: MuJoCo emits messages with an empty velocity list for the first
        few ticks during initialisation.  We guard against IndexError and hold
        _state_ready = False until velocities are available, so the control loop
        does nothing until the state vector is fully populated.
        """
        has_vel = len(msg.velocity) == len(msg.position)

        for i, name in enumerate(msg.name):
            pos = msg.position[i]
            vel = msg.velocity[i] if has_vel else None

            if name == "Joint_yaw":
                self._yaw = pos
                if vel is not None:
                    self._yaw_dot = vel
            elif name == "joint_pitch":
                self._pitch = pos
                if vel is not None:
                    self._pitch_dot = vel

        if has_vel:
            self._state_ready = True

    # ------------------------------------------------------------------
    # Control loop (50 Hz timer)
    # ------------------------------------------------------------------

    def _control_loop(self):
        if not self._state_ready:
            return

        th1_raw = self._yaw
        th2_raw = self._pitch
        th1_dot = self._yaw_dot
        th2_dot = self._pitch_dot

        # -- AI FRAME: angles wrapped to [-π, π] (neural network input) --
        th1_ai = (th1_raw + np.pi) % (2.0 * np.pi) - np.pi
        th2_ai = (th2_raw + np.pi) % (2.0 * np.pi) - np.pi

        # -- PHYSICS FRAME: offset so 0 = hanging, π = upright (energy formula) --
        th2_phys = th2_raw + np.pi

        if abs(th2_ai) < 0.4:
            # ============================================================
            # STAGE 2 — SAC neural-network balancing
            # ============================================================
            state_t = torch.FloatTensor(
                [th1_ai, th2_ai, th1_dot, th2_dot]
            ).unsqueeze(0)
            torque = self.actor.get_action(state_t).item()

        else:
            # ============================================================
            # STAGE 1 — Åström-Furuta energy-pumping swing-up
            # ============================================================
            E_current = (0.5 * self.J_P * (th2_dot ** 2)
                         + self.M_P * self.G * self.L_COM * (1.0 - np.cos(th2_phys)))
            E_error   = self.E_TARGET - E_current

            # Negative sign compensates for the angle convention offset.
            k_E         = 25.0
            pump_torque = -k_E * E_error * th2_dot * np.cos(th2_phys)

            # Dead-zone kick when pendulum is stationary and cold.
            if abs(th2_dot) < 0.05 and E_current < 0.1:
                pump_torque = -5.0

            # Yaw stabiliser (uses wrapped angle to prevent wind-up).
            k_p    = 5.0
            k_d    = 0.5
            torque = pump_torque - (k_p * th1_ai) - (k_d * th1_dot)

        torque = float(np.clip(torque, -10.0, 10.0))

        out_msg      = Float64MultiArray()
        out_msg.data = [torque]
        self.torque_pub.publish(out_msg)

        self.get_logger().info(
            f"[CTRL] Yaw: {th1_ai:+.2f} rad | Pitch: {th2_ai:+.2f} rad | "
            f"Torque: {torque:+.2f} Nm",
            throttle_duration_sec=0.2,
        )


def main(args=None):
    rclpy.init(args=args)
    node = FurutaRLController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()