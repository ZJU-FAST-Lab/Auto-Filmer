# all information for a replan

Header header

int16 state
float64[] iniState
float64[] iniAngle
nav_msgs/Odometry target
quadrotor_msgs/OccMap3d occmap

#shot debug
float64[] shot_d
float64[] shot_angle
float64[] shot_img_x
float64[] shot_img_y

# other temporal variables
float64[] path
time replan_stamp