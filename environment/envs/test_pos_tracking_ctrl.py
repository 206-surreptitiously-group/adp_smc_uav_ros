#!/usr/bin/python3

import datetime
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import rospy

from UAV.uav_visualization import UAV_Visualization
from UAV.FNTSMC import fntsmc_param
from UAV.ref_cmd import *
from UAV.uav import uav_param
from UAV.uav_pos_ctrl import uav_pos_ctrl
from environment.Color import Color
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from common.common_func import *

'''Parameter list of the quadrotor'''
DT = 0.01
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 60
uav_param.pos_zone = np.atleast_2d([[-3, 3], [-3, 3], [0, 3]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-45), deg2rad(45)], [deg2rad(-45), deg2rad(45)], [deg2rad(-120), deg2rad(120)]])
'''Parameter list of the quadrotor'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param()
att_ctrl_param.k1 = np.array([25, 25, 40])
att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
att_ctrl_param.alpha = np.array([2.5, 2.5, 2.5])
att_ctrl_param.beta = np.array([0.99, 0.99, 0.99])
att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
att_ctrl_param.dim = 3
att_ctrl_param.dt = DT
att_ctrl_param.ctrl0 = np.array([0., 0., 0.])
att_ctrl_param.saturation = np.array([0.3, 0.3, 0.3])
'''Parameter list of the attitude controller'''

'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param()
pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''

if __name__ == '__main__':
    rospy.init_node(name='test_pos_tracking_ctrl', anonymous=False)
    rate = rospy.Rate(1 / DT)

    '''1. Define a controller'''
    pos_ctrl = uav_pos_ctrl(uav_param, att_ctrl_param, pos_ctrl_param)
    quad_vis = UAV_Visualization()

    '''2. Define parameters for signal generator'''
    NUM_OF_SIMULATION = 10
    cnt = 0

    # '''3. Control'''
    while (not rospy.is_shutdown()) and (cnt < NUM_OF_SIMULATION):
        ref_amplitude, ref_period, ref_bias_a, ref_bias_phase = pos_ctrl.generate_random_circle(yaw_fixed=False)
        ref, _, _, _ = ref_uav(pos_ctrl.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)

        phi_d = phi_d_old = 0.
        theta_d = theta_d_old = 0.
        dot_phi_d = (phi_d - phi_d_old) / pos_ctrl.dt
        dot_theta_d = (theta_d - theta_d_old) / pos_ctrl.dt
        throttle = pos_ctrl.m * pos_ctrl.g

        uav_param.time_max = 30
        uav_param.pos0 = pos_ctrl.set_random_init_pos(pos0=ref[0:3], r=0.3 * np.ones(3))

        quad_vis.reset()
        pos_ctrl.uav_reset_with_new_param(new_uav_param=uav_param)  # 无人机初始参数，只变了初始位置
        pos_ctrl.controller_reset_with_new_param(new_att_param=att_ctrl_param, new_pos_param=pos_ctrl_param)  # 控制器参数，一般不变
        pos_ctrl.collector_reset(round(uav_param.time_max / uav_param.dt))
        ref_traj = pos_ctrl.generate_ref_pos_trajectory(ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
        pos_ctrl.draw_3d_trajectory_projection(ref_traj)
        pos_ctrl.draw_init_image()
        pos_ctrl.draw_3d_points_projection(np.atleast_2d([pos_ctrl.uav_pos()]), [Color().Red])
        pos_ctrl.show_image(True)

        if cnt % 1 == 0:
            print('Current:', cnt)
        writer = cv.VideoWriter('record.mp4', cv.VideoWriter_fourcc(*'mp4v'), 300, (pos_ctrl.width, pos_ctrl.height), True)

        while pos_ctrl.time < pos_ctrl.time_max - DT / 2:
            # if pos_ctrl.n % 1000 == 0:
            #     print('time: ', pos_ctrl.n * pos_ctrl.dt)

            '''3.1 generate '''
            ref, dot_ref, dot2_ref, _ = ref_uav(pos_ctrl.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # xd yd zd psid
            uncertainty = generate_uncertainty(time=pos_ctrl.time, is_ideal=True)
            obs = np.zeros(3)

            '''3.2 outer-loop control'''
            phi_d_old = phi_d
            theta_d_old = theta_d
            phi_d, theta_d, throttle = pos_ctrl.pos_control(ref[0:3], dot_ref[0:3], dot2_ref[0:3], uncertainty, obs)
            dot_phi_d = (phi_d - phi_d_old) / pos_ctrl.dt
            dot_theta_d = (theta_d - theta_d_old) / pos_ctrl.dt

            '''3.3 inner-loop control'''
            rho_d = np.array([phi_d, theta_d, ref[3]])
            dot_rho_d = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])
            torque = pos_ctrl.att_control(rho_d, dot_rho_d, np.zeros(3), att_only=False)

            '''3.4 update state'''
            action_4_uav = np.array([throttle, torque[0], torque[1], torque[2]])
            pos_ctrl.update(action=action_4_uav)

            '''3.5. publish'''
            quad_vis.render(uav_pos=pos_ctrl.uav_pos(),
                            target_pos=None,
                            uav_pos_ref=pos_ctrl.pos_ref,
                            uav_att=pos_ctrl.uav_att(),
                            uav_att_ref=pos_ctrl.att_ref,
                            d=8 * pos_ctrl.d,
                            tracking=True)

            # pos_ctrl.image = pos_ctrl.image_copy.copy()
            # pos_ctrl.draw_3d_points_projection(np.atleast_2d([pos_ctrl.uav_pos(), ref[0: 3]]), [Color().Red, Color().Green])
            # pos_ctrl.draw_error(pos_ctrl.uav_pos(), ref[0:3])
            # pos_ctrl.show_image(False)
            # writer.write(pos_ctrl.image)

            # rate.sleep()
        # print(cnt, '  Finish...')
        writer.release()
        rise = pos_ctrl.RISE()
        # print('RISE calculation:')
        # print('  x: %.2f,  y: %.2f,  z: %.2f' % (rise[0], rise[1], rise[2]))
        if max(rise) > 0.03:
            print('WARNING......')
            _str = (str(cnt) + ',' + '[%.2f, %.2f, %.2f]' % (rise[0], rise[1], rise[2]) + '\n' +
                    '  amplitude: [%.2f, %.2f, %.2f, %.2f]' % (ref_amplitude[0], ref_amplitude[1], ref_amplitude[2], ref_amplitude[3]) + '\n' +
                    '  period: [%.2f, %.2f, %.2f, %.2f]' % (ref_period[0], ref_period[1], ref_period[2], ref_period[3]) + '\n' +
                    '  bias_a: [%.2f, %.2f, %.2f, %.2f]' % (ref_bias_a[0], ref_bias_a[1], ref_bias_a[2], ref_bias_a[3]) + '\n' +
                    '  bias_phase: [%.2f, %.2f, %.2f, %.2f]' % (ref_bias_phase[0], ref_bias_phase[1], ref_bias_phase[2], ref_bias_phase[3]) + '\n\n')
            print(_str)
        cnt += 1
        SAVE = False
        if SAVE:
            new_path = (os.path.dirname(os.path.abspath(__file__)) +
                        '/../../datasave/' +
                        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/')
            os.mkdir(new_path)
            pos_ctrl.collector.package2file(path=new_path)
        # plt.ion()
        # pos_ctrl.collector.plot_att()
        # pos_ctrl.collector.plot_pqr()
        # pos_ctrl.collector.plot_torque()
        # pos_ctrl.collector.plot_pos()
        # pos_ctrl.collector.plot_vel()
        # pos_ctrl.collector.plot_throttle()
        # pos_ctrl.collector.plot_outer_obs()
        # plt.pause(2)
        # plt.ioff()
        # plt.show()
