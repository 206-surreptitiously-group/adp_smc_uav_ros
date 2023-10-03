# ADP-SMC-UAV
A repository about ADP-based SMC controller for a UAV.
 
To satisfy the requirements of physical experiments, we use SMC to design both inner and outer-loop controllers.
However, we fix the inner-loop controller parameters and just use RL to train some hyperparameters of the outer-loop controller.

Therefore, the uav model, inner-loop controller, and outer-loop controller are integrated together as the "environment" of the RL.

We use Rviz to display 3D UAV control performance.
Noting that Rviz is only utilized for graphical display.
Hope we can get satisfactory performance as soon as possible.

One can download the package to your own ROS workspace and run the SMC tests for UAV attitude
and position control demo by using 
```commandline
catkin_make
source devel/setup.bash
roslaunch adp_smc_uav_ros att_ctrl.launch
roslaunch adp_smc_uav_ros pos_ctrl.launch
```
Tips: One can comment out the "rate.sleep()" in "test_att_ctrl.py" or "test_pos_ctrl.py" for
faster simulation.
