import numpy as np

class CustomFurutaEnv:
    def __init__(self):
        # ==========================================
        # PARAMETERS UPDATED FROM URDF
        # ==========================================
        
        # Pendulum (horizontal_vertical_rod_link)
        # Mass: 0.164162 kg 
        # Length: COM is at 0.12837m, so total rod length is ~0.25674m 
        self.base_m_p = 0.164162145283487  
        self.base_l_p = 0.256741556853168  
        
        # Rotary Arm (motor_encoder_link)
        # Mass: 0.086537 kg [cite: 3]
        # Length: Joint_pitch Y-offset is 0.04m from the rotation axis 
        self.base_m_r = 0.0865375317171068 
        self.base_l_r = 0.04               
        
        self.g = 9.81
        self.dt = 0.02         # 50 Hz control loop
        self.max_torque = 100.0 # Updated to match URDF effort limit 
        
        # Friction/Damping (Extracted from joint dynamics)
        # Joint_yaw damping: 0.005 
        # Joint_pitch damping: 0.005 
        self.base_b_r = 0.005 
        self.base_b_p = 0.005 

        self.reset()

    def reset(self):
        # Innovation 1: Domain Randomization
        random_factor = lambda: np.random.uniform(0.8, 1.2)
        
        self.m_p = self.base_m_p * random_factor()
        self.l_p = self.base_l_p * random_factor()
        self.m_r = self.base_m_r * random_factor()
        self.l_r = self.base_l_r * random_factor()
        
        self.b_r = self.base_b_r * random_factor()
        self.b_p = self.base_b_p * random_factor()

        # Innovation 2: Rigid Body Inertia Tensors
        # Using solid rod inertia (1/3 * m * L^2) as per your simulation logic
        self.J_r = (1.0 / 3.0) * self.m_r * (self.l_r ** 2)
        self.J_p = (1.0 / 3.0) * self.m_p * (self.l_p ** 2)

        # State: [th1 (arm), th2 (pendulum), th1_dot, th2_dot]
        # Start near the top (0 radians is UP)
        self.state = np.array([
            0.0, 
            np.random.uniform(-0.1, 0.1), 
            0.0, 
            0.0
        ], dtype=np.float32)
        
        return self._get_obs()

    def _get_obs(self):
        # Strict state wrapping (-pi to pi)
        th1 = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
        th2 = (self.state[1] + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([th1, th2, self.state[2], self.state[3]], dtype=np.float32)

    def _compute_derivatives(self, state, torque):
        th1, th2, th1_dot, th2_dot = state
        
        sin_th2 = np.sin(th2)
        cos_th2 = np.cos(th2)
        
        # Lagrangian Mass Matrix Elements M(theta)
        M11 = self.J_r + self.m_p * (self.l_r**2) + self.J_p * (sin_th2**2)
        M12 = -self.m_p * self.l_r * (self.l_p / 2.0) * cos_th2
        M21 = M12
        M22 = self.J_p + self.m_p * ((self.l_p / 2.0)**2)
        
        # Coriolis and Centrifugal Forces C(theta, theta_dot)
        C1 = self.J_p * np.sin(2 * th2) * th1_dot * th2_dot \
             + self.m_p * self.l_r * (self.l_p / 2.0) * sin_th2 * (th2_dot**2)
        C2 = -0.5 * self.J_p * np.sin(2 * th2) * (th1_dot**2)
        
        # Gravity Vector G(theta)
        G2 = -self.m_p * self.g * (self.l_p / 2.0) * sin_th2
        
        # Friction
        F1 = self.b_r * th1_dot
        F2 = self.b_p * th2_dot
        
        # Solve M * q_ddot + C + G = Tau
        RHS1 = torque - C1 - F1
        RHS2 = -C2 - G2 - F2
        
        det_M = (M11 * M22) - (M12 * M21)
        
        th1_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
        th2_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M
        
        return np.array([th1_dot, th2_dot, th1_ddot, th2_ddot])

    def step(self, action):
        torque = np.clip(action[0], -self.max_torque, self.max_torque)
        
        # Runge-Kutta 4th Order Integration
        k1 = self._compute_derivatives(self.state, torque)
        k2 = self._compute_derivatives(self.state + 0.5 * self.dt * k1, torque)
        k3 = self._compute_derivatives(self.state + 0.5 * self.dt * k2, torque)
        k4 = self._compute_derivatives(self.state + self.dt * k3, torque)
        
        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        obs = self._get_obs()
        th1, th2, th1_dot, th2_dot = obs
        
        # Reward Function
        alive_bonus = 1.0
        angle_penalty = (th2 ** 2) * 2.0
        velocity_penalty = (th1_dot ** 2) * 0.01 + (th2_dot ** 2) * 0.01
        
        reward = alive_bonus - angle_penalty - velocity_penalty
        done = bool(abs(th2) > 0.8)
        
        return obs, reward, done, {}