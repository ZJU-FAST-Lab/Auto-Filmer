#! /usr/bin/env python
import math
import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry


class map_generator(object):

    def __init__(self):
        """ Constructor """
        rospy.init_node('random_map_sensing', anonymous=True)
        self.map = None
        self.points = []
        self.has_odom = False
        self.has_map = False
        self.init_params()
        self.floor_bias = 0.50

    def init_params(self):
        """ Initializes ros parameters """
        
        # rospack = rospkg.RosPack()
        # package_path = rospack.get_path("yolov4_trt_ros")
        self.dimx = rospy.get_param("map/x_size", default=10.0)
        self.dimy = rospy.get_param("map/y_size", default=10.0)
        self.dimz = rospy.get_param("map/z_size", default=3.0)

        self.resolution = rospy.get_param("map/resolution", default=0.05)

        self.add_floor = rospy.get_param("map/add_floor", default=True)
        self.add_ceiling = rospy.get_param("map/add_ceiling", default=True)
        
        # self.all_map_topic = rospy.get_param("all_map_topic", default="/map_generator/global_cloud")
        self.all_map_topic = rospy.get_param("all_map_topic", default="/global_map")
        self.all_map_pub = rospy.Publisher(self.all_map_topic, PointCloud2, queue_size=1)
        # self.odom_sub = rospy.Subscriber( "odometry", Odometry, self.odom_callback, queue_size=50);

        self.rate = rospy.get_param("sensing/rate", default=1.0)
        self.rate = rospy.Rate(self.rate)

        print("dimx: ", self.dimx)
        print("dimy: ", self.dimy)
        print("dimz: ", self.dimz)

    def add_box(self, size, position):
        ''' size: [x,y,z]
            position: [x,y,z] --- center position
        '''
        position[2] -= self.floor_bias
        x_low = math.floor((position[0] - size[0] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        x_high = math.floor((position[0] + size[0] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        y_low = math.floor((position[1] - size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        y_high = math.floor((position[1] + size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        z_low = math.floor((position[2] - size[2] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        z_high = math.floor((position[2] + size[2] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution

        x = x_low
        while x <= x_high:
            y = y_low
            while y <= y_high:
                z = z_low
                while z <= z_high:
                    if (math.fabs(x - x_low) < self.resolution) or (math.fabs(x - x_high) < self.resolution) \
                        or (math.fabs(y - y_low) < self.resolution) or (math.fabs(y - y_high) < self.resolution) \
                        or (math.fabs(z - z_low) < self.resolution) or (math.fabs(z - z_high) < self.resolution):
                        self.points.append([x,y,z])
                    z += self.resolution
                y += self.resolution
            x += self.resolution

        return

    def add_epplisoid(self, size, position):
        ''' size: [x,y,z]
            position: [x,y,z] --- center position
        '''
        # position[2] -= self.floor_bias
        x_low = math.floor((position[0] - size[0] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        x_high = math.floor((position[0] + size[0] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        y_low = math.floor((position[1] - size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        y_high = math.floor((position[1] + size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        z_low = math.floor((position[2] - size[2] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        z_high = math.floor((position[2] + size[2] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution

        x = x_low
        while x <= x_high:
            y = y_low
            while y <= y_high:
                z = z_low
                while z <= z_high:
                    a = math.fabs(x - position[0]) / (size[0] / 2)
                    b = math.fabs(y - position[1]) / (size[1] / 2)
                    c = math.fabs(z - position[2]) / (size[2] / 2)
                    if math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2) < 1:
                        self.points.append([x,y,z])
                    # if (math.fabs(x - x_low) < self.resolution) or (math.fabs(x - x_high) < self.resolution) \
                    #     and (math.fabs(y - y_low) < self.resolution) or (math.fabs(y - y_high) < self.resolution) \
                    #     and (math.fabs(z - z_low) < self.resolution) or (math.fabs(z - z_high) < self.resolution):
                    #     self.points.append([x,y,z])
                    z += self.resolution
                y += self.resolution
            x += self.resolution

        return

    def add_cylinder(self, size, position):
        ''' size: [r, h]
            position: [x,y,z] --- center position
        '''
        position[2] -= self.floor_bias
        center_x = position[0]
        center_y = position[1]
        z_low = math.floor((position[2] - size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        z_high = math.floor((position[2] + size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution

        radius_num = math.floor(size[0] / self.resolution)
        x = - radius_num
        while x <= radius_num:
            y = - radius_num
            while y <= radius_num:
                radius2 = x ** 2 + y ** 2
                if radius2 < (radius_num + 0.5) ** 2:
                    z = z_low
                    while z <= z_high:
                        if radius2 > (radius_num - 0.5) ** 2 or \
                            (math.fabs(z - z_low) < self.resolution) or (math.fabs(z - z_high) < self.resolution):
                            self.points.append([center_x + x * self.resolution, center_y + y * self.resolution, z])
                        z += self.resolution
                y += 1
            x += 1

        return

    def add_layer(self, size, position):
        ''' size: [x,y]
            position: [x,y,z] --- center position
        '''        
        x_low = math.floor((position[0] - size[0] / 2) / self.resolution) * self.resolution + 0.5 * self.resolution
        x_high = math.floor((position[0] + size[0] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        y_low = math.floor((position[1] - size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        y_high = math.floor((position[1] + size[1] / 2) / self.resolution) * self.resolution  + 0.5 * self.resolution
        z = position[2]

        x = x_low
        while x <= x_high:
            y = y_low
            while y <= y_high:
                self.points.append([x,y,z])
                y += self.resolution
            x += self.resolution
        
        return

    def add_chair(self, position):
        '''position: [x, y] '''
        x = np.random.rand() * 0.2 + 0.4
        h = np.random.rand() * 0.1 + 0.4

        self.add_box_on_floor([x, x, h], [position[0], position[1], h/2])
        direction = np.random.randint(0, 5)
        h_ratio = 2 # 1.5 + np.random.rand() * 0.5
        if direction == 0:
            return
        elif direction == 1:
            self.add_box_on_floor([x, 0.1, h * h_ratio], [position[0], position[1] + x / 2, h * h_ratio / 2])
        elif direction == 2:
            self.add_box_on_floor([x, 0.1, h * h_ratio], [position[0], position[1] - x / 2, h * h_ratio / 2])
        elif direction == 3:
            self.add_box_on_floor([0.1, x, h * h_ratio], [position[0] + x / 2, position[1] , h * h_ratio / 2])
        elif direction == 4:
            self.add_box_on_floor([0.1, x, h * h_ratio], [position[0] - x / 2, position[1], h * h_ratio / 2])

    def add_bed(self, position):
        ''' position[x,y]'''
        x = np.random.rand() * 0.6 + 1.6
        y = np.random.rand() * 0.6 + 1.2
        z = np.random.rand() * 0.2 + 0.3
        # left right
        if np.random.rand() > 0.5:
            self.add_box_on_floor([x, y, z], [-self.dimx / 2 + x / 2, position[1], z/2])
            if np.random.rand() > 0.3:
                self.add_box_on_floor([0.1, y, z*1.5], [-self.dimx / 2 + 0.1, position[1], z*1.5/2])
        else:
            self.add_box_on_floor([x, y, z], [self.dimx / 2 - x / 2, position[1], z/2])
            if np.random.rand() > 0.3:
                self.add_box_on_floor([0.1, y, z*1.5], [self.dimx / 2 - 0.1, position[1], z*1.5/2])

    def add_table(self, position):
        ''' position[x,y]'''
        x, y, z = np.random.rand(3)
        ratio = 0.3 + np.random.rand() * 0.5

        x = np.random.rand() * 1.0 + 0.4
        y = np.random.rand() * 1.0 + 0.4
        z = z * 0.6 + 0.4
        # upper part
        if np.random.rand() > 0.5:
            self.add_cylinder([x/2, 0.1], [position[0] * 0.6, position[1], z])
        else:
            self.add_box([x, y, 0.1], [position[0] * 0.6, position[1], z])
        # lower part
        if np.random.rand() > 0.5:
            self.add_cylinder_on_floor([x/2 * ratio, z], [position[0] * 0.6, position[1]])

        else:
            self.add_box_on_floor([x * ratio, y * ratio, z], [position[0] * 0.6, position[1]])

    def add_long_door(self, position):
        ''' position[x,y]'''
        # warning: need to check map
        x = np.random.rand() * 0.4 + 0.8
        y = np.random.rand() * 0.4 + 0.4

        pos_x = (position[0] + x / 2) + np.random.rand() * (self.dimx - x)
        pos_y = position[1] + self.step_size / 2.0
        # left side
        self.add_box_on_floor([pos_x - x / 2 + self.dimx / 2, y, self.dimz], [-self.dimx / 2 + (pos_x - x / 2 + self.dimx / 2) / 2, pos_y, self.dimz / 2])
        # right side
        self.add_box_on_floor([self.dimx / 2 - pos_x - x / 2, y, self.dimz], [self.dimx / 2 - (self.dimx / 2 - pos_x - x / 2) / 2, pos_y, self.dimz / 2])

    def add_box_on_floor(self, size, position):
        ''' size: [x, y, z]
        position: [x, y] '''
        self.add_box(size, position + [size[-1] / 2])

    def add_random_box(self, position):
        ''' position: [x, y] '''
        x = 10
        y = 10
        z = 10
        while x + y + z > 4:
            x = np.random.rand() * 0.7 + 0.4
            y = np.random.rand() * 0.7 + 0.5
            z = np.random.rand() * 1.4 + 0.4

        self.add_box_on_floor([x,y,z], position)

    def add_random_cylinder(self, position):
        ''' position: [x, y] '''
        r = 10
        h = 10
        while r * 2 + h > 3:
            r = np.random.rand() * 0.35 + 0.15
            h = np.random.rand() * 1.2 + 0.8

        self.add_cylinder_on_floor([r,h], position)

    def add_stride_on_ceiling(self, position):
        ''' 
        position: [x, y] '''
        y = np.random.rand() * 0.2 + 0.4
        z = np.random.rand() * 0.4 + 0.2
        self.add_box([self.dimx, y, z], [0, position[1], self.dimz - z / 2])

    def add_stride_on_floor(self, position):
        '''
        position: [x, y] '''
        y = np.random.rand() * 0.2 + 0.4
        z = np.random.rand() * 0.4 + 0.2
        self.add_box([self.dimx, y, z], [0, position[1], 0 + z / 2])

    def add_cylinder_on_floor(self, size, position):
        ''' size: [r, h]
        position: [x, y] '''
        self.add_cylinder(size, [position[0], position[1], size[1] / 2])

    def publish_map(self):
        if not self.has_map:
            return False
        self.all_map_pub.publish(self.map)
        return True

    def odom_callback(self, odom):
        self.has_odom = True
        self.publish_map()

    def make_random_corridor(self):
        self.dimx = 3.0
        self.dimy = 50.0
        self.dimz = 2.0
        self.step_size = 1.8
        steps = math.floor(self.dimy / self.step_size)
        for i in range(int(steps)):
            # clear start and end
            if i < 1 or i > steps - 2:
                continue
            obs_num = np.random.randint(2, 4)

            center_x = - self.dimx / 2
            center_y = - self.dimy / 2 + self.step_size * i

            if np.random.rand() < 0.2:
                self.add_long_door([center_x, center_y])
                continue

            for num in range(obs_num):
                pos_x = center_x + 0.2 + np.random.rand() * (self.dimx - 0.4)
                pos_y = center_y + 0.3 + np.random.rand() * (self.step_size - 0.6)
                object_type = np.random.randint(0, 8)

                if object_type < 3:
                    self.add_random_box([pos_x, pos_y])
                # elif object_type == 1:
                #     self.add_chair([pos_x, pos_y])
                # elif object_type == 2:
                #     self.add_bed([pos_x, pos_y])
                elif object_type < 4:
                    self.add_stride_on_ceiling([center_x, center_y])
                elif object_type < 5:
                    self.add_stride_on_floor([pos_x, pos_y])
                # elif object_type == 5:
                #     self.add_table([pos_x, pos_y])
                # elif object_type == 6:
                #     self.add_random_box([pos_x, pos_y])
                elif object_type < 7:
                    self.add_random_cylinder([pos_x, pos_y])
                else:
                    self.add_random_box([pos_x, pos_y])


    def make_map(self):
        rospy.loginfo("start making map")

        self.add_box([2.0, 4.0, 3.0], [-2.0, -5.5, 1.5])

        self.add_box([1.5, 4.0, 3.0], [-3.0, 4.0, 1.5])
        
        self.add_box([1.5, 1.0, 3.0], [10.0, -2.0, 1.5])

        self.add_box([1.0, 1.5, 3.0], [5.0, -8.0, 1.5])

        self.add_box([2.0, 3.0, 3.0], [-10.0, 0.0, 1.5])

        self.add_box([5.0, 2.0, 3.0], [5.0, 14.0, 1.5])

        self.add_box([2.0, 5.0, 3.0], [-4.0, 13.0, 1.5])
        
        self.add_box([2.0, 4.0, 3.0], [3.0, 0.0, 1.5])

        self.add_box([2.0, 6.0, 3.0], [-10.0, 15.0, 1.5])

        self.add_box([4.0, 2.0, 3.0], [6.0, 6.0, 1.5])

        self.add_box([3.0, 2.0, 3.0], [-8.0, -10.0, 1.5])

        # transfer to pcl
        self.map = PointCloud2()
        self.map.height = 1
        self.map.width = len(self.points)
        self.map.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        self.map.point_step = 12 #12
        self.map.row_step = 12 * len(self.points)
        self.map.is_bigendian = False
        self.map.is_dense = False
        self.map.data = np.asarray(self.points, np.float32).tostring()
        self.map.header.frame_id = "world"

        self.has_map = True
        rospy.loginfo("finish making map")

        return True
    
def main():
    

    map_maker = map_generator()
    map_maker.make_map()
    while not rospy.is_shutdown():
        map_maker.publish_map()
        map_maker.rate.sleep()

if __name__ == '__main__':
    main()

