import numpy as np


class uav_param:
    def __init__(self):
        self.m: float = 0.8                                                 # 无人机质量
        self.g: float = 9.8                                                 # 重力加速度
        self.J: np.ndarray = np.array([4.212e-3, 4.212e-3, 8.255e-3])       # 转动惯量
        self.d: float = 0.12                                                # 机臂长度 'X'构型
        self.CT: float = 2.168e-6                                           # 螺旋桨升力系数
        self.CM: float = 2.136e-8                                           # 螺旋桨力矩系数
        self.J0: float = 1.01e-5                                            # 电机和螺旋桨的转动惯量
        self.kr: float = 1e-3                                               # 旋转阻尼系数
        self.kt: float = 1e-3                                               # 平移阻尼系数
        self.pos0: np.ndarray = np.array([0, 0, 0])
        self.vel0: np.ndarray = np.array([0, 0, 0])
        self.angle0: np.ndarray = np.array([0, 0, 0])
        self.pqr0: np.ndarray = np.array([0, 0, 0])
        self.dt = 0.01
        self.time_max = 30  # 每回合最大时间


class UAV:
    def __init__(self, param: uav_param):
        self.m = param.m
        self.g = param.g
        self.J = param.J
        self.d = param.d
        self.CT = param.CT
        self.CM = param.CM
        self.J0 = param.J0
        self.kr = param.kr
        self.kt = param.kt

        self.x = param.pos0[0]
        self.y = param.pos0[1]
        self.z = param.pos0[2]
        self.vx = param.vel0[0]
        self.vy = param.vel0[1]
        self.vz = param.vel0[2]
        self.phi = param.angle0[0]
        self.theta = param.angle0[1]
        self.psi = param.angle0[2]
        self.p = param.pqr0[0]
        self.q = param.pqr0[1]
        self.r = param.pqr0[2]

        self.dt = param.dt
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.time_max = param.time_max

        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩

    def ode(self, xx: np.ndarray, dis: np.ndarray):
        """
        @param xx:      state of the uav
        @param dis:     disturbances
        @return:        dot_xx
        """
        [_x, _y, _z, _vx, _vy, _vz, _phi, _theta, _psi, _p, _q, _r] = xx[0:12]
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        self.J0 = 0.  # 不考虑陀螺力矩，用于分析观测器的效果
        dp = (-self.kr * _p - _q * _r * (self.J[2] - self.J[1]) + self.torque[0]) / self.J[0]
        dq = (-self.kr * _q - _p * _r * (self.J[0] - self.J[2]) + self.torque[1]) / self.J[1]
        dr = (-self.kr * _r - _p * _q * (self.J[1] - self.J[0]) + self.torque[2]) / self.J[2]
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''

        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''
        _R_pqr2diner = np.array([[1, np.tan(_theta) * np.sin(_phi), np.tan(_theta) * np.cos(_phi)],
                                 [0, np.cos(_phi), -np.sin(_phi)],
                                 [0, np.sin(_phi) / np.cos(_theta), np.cos(_phi) / np.cos(_theta)]])
        [dphi, dtheta, dpsi] = np.dot(_R_pqr2diner, [_p, _q, _r]).tolist()
        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''

        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        [dx, dy, dz] = [_vx, _vy, _vz]
        dvx = (self.throttle * (np.cos(_psi) * np.sin(_theta) * np.cos(_phi) + np.sin(_psi) * np.sin(_phi))
               - self.kt * _vx + dis[0]) / self.m
        dvy = (self.throttle * (np.sin(_psi) * np.sin(_theta) * np.cos(_phi) - np.cos(_psi) * np.sin(_phi))
               - self.kt * _vy + dis[1]) / self.m
        dvz = -self.g + (self.throttle * np.cos(_phi) * np.cos(_theta)
                         - self.kt * _vz + dis[2]) / self.m
        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''

        return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

    def rk44(self, action: np.ndarray, dis: np.ndarray, n: int = 10, att_only: bool = False):
        self.throttle = action[0]
        self.torque = action[1: 4]
        h = self.dt / n  # RK-44 解算步长
        cc = 0
        xx = self.uav_state_call_back()
        while cc < n:
            K1 = h * self.ode(xx, dis)
            K2 = h * self.ode(xx + K1 / 2, dis)
            K3 = h * self.ode(xx + K2 / 2, dis)
            K4 = h * self.ode(xx + K3, dis)
            xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            cc += 1
        if att_only:
            xx[0:6] = np.zeros(6)[:]
        self.set_state(xx)
        self.time += self.dt
        if self.psi > np.pi:  # 如果角度超过 180 度
            self.psi -= 2 * np.pi
        if self.psi < -np.pi:  # 如果角度小于 -180 度
            self.psi += 2 * np.pi
        self.n += 1  # 拍数 +1

    def uav_state_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def uav_pos_vel_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz])

    def uav_att_pqr_call_back(self):
        return np.array([self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def uav_pos(self):
        return np.array([self.x, self.y, self.z])

    def uav_vel(self):
        return np.array([self.vx, self.vy, self.vz])

    def uav_att(self):
        return np.array([self.phi, self.theta, self.psi])

    def uav_pqr(self):
        return np.array([self.p, self.q, self.r])

    def set_state(self, xx: np.ndarray):
        [self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r] = xx[:]

    def f1(self) -> np.ndarray:
        """
        :brief:  [1  sin(phi)tan(theta)      cos(phi)tan(theta)]
                 [0       cos(phi)               -sin(phi)     ]
                 [0  sin(phi)/cos(theta)   -cos(phi)/cos(theta)]
        :return: f1(rho_1)
        """
        _f1_rho1 = np.zeros((3, 3)).astype(float)
        _f1_rho1[0][0] = 1.
        _f1_rho1[0][1] = np.sin(self.phi) * np.tan(self.theta)
        _f1_rho1[0][2] = np.cos(self.phi) * np.tan(self.theta)
        _f1_rho1[1][1] = np.cos(self.phi)
        _f1_rho1[1][2] = -np.sin(self.phi)
        _f1_rho1[2][1] = np.sin(self.phi) / np.cos(self.theta)
        _f1_rho1[2][2] = np.cos(self.phi) / np.cos(self.theta)
        return _f1_rho1

    def f2(self) -> np.ndarray:
        """
        :brief:  [(kr * p + qr * (Iyy - Izz)) / Ixx]
                 [(kr * q + pr * (Izz - Ixx)) / Iyy]
                 [(kr * r + pq * (Ixx - Iyy)) / Izz]
        :return: f2(rho_2)
        """
        _f2_rho2 = np.array([0, 0, 0]).astype(float)
        _f2_rho2[0] = (self.kr * self.p + self.q * self.r * (self.J[1] - self.J[2])) / self.J[0]
        _f2_rho2[1] = (self.kr * self.q + self.p * self.r * (self.J[2] - self.J[0])) / self.J[1]
        _f2_rho2[2] = (self.kr * self.r + self.p * self.q * (self.J[0] - self.J[1])) / self.J[2]
        return _f2_rho2

    def h(self) -> np.ndarray:
        """
        :brief:  [        0             1/Jxx    0       0 ]
                 [        0               0    1/Jyy     0 ]
                 [        0               0      0    1/Jzz]
        :return: h(rho_1)
        """
        _g = np.zeros((3, 3)).astype(float)
        _g[0][0] = 1 / self.J[0]
        _g[1][1] = 1 / self.J[1]
        _g[2][2] = 1 / self.J[2]
        return _g

    def rho1(self) -> np.ndarray:
        return np.array([self.phi, self.theta, self.psi])

    def rho2(self) -> np.ndarray:
        return np.array([self.p, self.q, self.r])

    def dot_rho1(self) -> np.ndarray:
        return np.dot(self.f1(), self.rho2())

    def dot_rho2(self) -> np.ndarray:
        return self.f2() + np.dot(self.h(), self.torque)

    def F(self) -> np.ndarray:
        dot_rho1 = self.dot_rho1()  # dphi dtheta dpsi
        _dot_f1_rho1 = np.zeros((3, 3)).astype(float)
        _dot_f1_rho1[0][1] = dot_rho1[0] * np.tan(self.theta) * np.cos(self.phi) + dot_rho1[1] * np.sin(self.phi) / np.cos(self.theta) ** 2
        _dot_f1_rho1[0][2] = -dot_rho1[0] * np.tan(self.theta) * np.sin(self.phi) + dot_rho1[1] * np.cos(self.phi) / np.cos(self.theta) ** 2

        _dot_f1_rho1[1][1] = -dot_rho1[0] * np.sin(self.phi)
        _dot_f1_rho1[1][2] = -dot_rho1[0] * np.cos(self.phi)

        _temp1 = dot_rho1[0] * np.cos(self.phi) * np.cos(self.theta) + dot_rho1[1] * np.sin(self.phi) * np.sin(self.theta)
        _dot_f1_rho1[2][1] = _temp1 / np.cos(self.theta) ** 2

        _temp2 = -dot_rho1[0] * np.sin(self.phi) * np.cos(self.theta) + dot_rho1[1] * np.cos(self.phi) * np.sin(self.theta)
        _dot_f1_rho1[2][2] = _temp2 / np.cos(self.theta) ** 2
        return _dot_f1_rho1

    def second_order_att_dynamics(self) -> np.ndarray:
        return np.dot(self.F(), self.rho2()) + np.dot(self.f1(), self.f2())

    def att_control_matrix(self) -> np.ndarray:
        return np.dot(self.f1(), self.h())

    def eta(self):
        return np.array([self.x, self.y, self.z])

    def dot_eta(self):
        return np.array([self.vx, self.vy, self.vz])
