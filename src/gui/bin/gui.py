#! /usr/bin/env python3
# the following lines are run on ubuntu 18.04 & 20.04
import tkinter as tk
from tkinter import ttk
# import tkFont

# the following lines are run on manifold2 ubuntu 16.04
# import Tkinter as tk
# import ttk

# import tkFont

import numpy as np

import rospy, rospkg
import math
import os
import time
import sys

import cv2 as cv
# import pyscreenshot as ImageGrab

from sensor_msgs.msg import Image
from PIL import Image as imagePIL
from PIL import ImageTk
from cv_bridge import CvBridge
from quadrotor_msgs.msg import ShotParams

class CustomScale(ttk.Scale):
    def __init__(self, master=None, style=None, init_value=None, **kw):
        kw.setdefault("orient", "vertical")
        self.variable = kw.pop('variable', tk.DoubleVar(master))
        self.variable.set(init_value)
        ttk.Scale.__init__(self, master, variable=self.variable, **kw)
        self._style_name = '{}.custom.{}.TScale'.format(self, kw['orient'].capitalize()) # unique style name to handle the text
        self['style'] = self._style_name
        self.style = style
        self.variable.trace('w', self._update_text)
        self._update_text()

    def _update_text(self, *args):
        self.style.configure(self._style_name, text="{:.1f}".format(self.variable.get()))

class GUI():
    def __init__(self):
        #NOTE important parameters
        self.min_dis = 2.0
        self.max_dis = 5.0
        self.min_time = 3.0
        self.max_time = 10.0
        #0314
        self.init_theta = 0.0 # 3.14
        self.f = 346.74048
        # object length in meter, 250 drone in sim
        self.obj_size = 0.33 * 2

        # ros
        rospy.init_node('gui', anonymous=True)
        self.image_sub = rospy.Subscriber("image", Image, self.imageCallback, tcp_nodelay=True)
        self.shot_pub = rospy.Publisher("shot", ShotParams,queue_size=10)
        self.ui_pub = rospy.Publisher("gui", Image, queue_size=10)
        self.bridge = CvBridge()
        
        # tkinter 
        self.root = tk.Tk()
        self.root.geometry('880x480')
        self.root.configure(background='white')
        self.root.title('GUI')
        self.cnt = 0

        # fpv image
        self.canvas_img = tk.Canvas(self.root, height=480, width=640, bg='white', highlightthickness=0)
        self.canvas_img.place(x=0, y=0)
        self.line1 = self.canvas_img.create_line(213, 0, 213, 480, fill='silver')
        self.line2 = self.canvas_img.create_line(426, 0, 426, 480, fill='silver')
        self.line3 = self.canvas_img.create_line(0, 160, 640, 160, fill='silver')
        self.line4 = self.canvas_img.create_line(0, 320, 640, 320, fill='silver')
        self.canvas_img.bind("<ButtonPress-1>", self.startMove)
        self.canvas_img.bind("<B1-Motion>", self.onMove)
        self.canvas_img.bind("<ButtonRelease-1>", self.stopMove)
        self.rect_x = 320
        self.rect_y = 240
        self.rect_size = self.obj_size * self.f / ((self.min_dis + self.max_dis)/2 )
        coord = self.getRectCoordinates(self.rect_x, self.rect_y, self.rect_size, self.rect_size)
        self.rect = self.canvas_img.create_rectangle(coord[0], coord[1], coord[2], coord[3], 
                                            outline='#1D6CD4', activeoutline='#EE0000' ,width=3,)

        # view angle
        self.canvas_upright = tk.Canvas(self.root, height=240, width=240, bg='white')
        self.canvas_upright.place(x=640, y=0)
        self.circle_x = 120
        self.circle_y = 120
        self.R_min = 40
        self.R_max = 80
        self.circle_R = (self.R_min + self.R_max)/2
        self.circle = self.canvas_upright.create_oval(self.circle_x - self.circle_R, self.circle_y - self.circle_R, 
                                            self.circle_x + self.circle_R, self.circle_y + self.circle_R, 
                                            width=1, dash=(3,7), outline='silver')
        # self.canvas_upright.configure(state=tk.DISABLED)
              
        rospack = rospkg.RosPack()
        path = os.path.join(rospack.get_path('gui'),'assets','man.png')
        man_image = tk.PhotoImage(file=path, format='png')
        self.label_man = tk.Label(self.root)
        man_size = 50
        self.label_man.place(x=760-man_size/2, y=120-man_size/2)
        self.label_man.configure(image=man_image,bg='white')

        path = os.path.join(rospack.get_path('gui'),'assets','drone.png')
        drone_image = tk.PhotoImage(file=path, format='png')
        
        self.drone_size = 24
        self.label_drone = tk.Label(self.root, image=drone_image)
        self.drone_theta = self.init_theta
        self.drone_x = 640 + self.circle_x - math.sin(self.drone_theta)*self.circle_R - self.drone_size/2
        self.drone_y = self.circle_y - math.cos(self.drone_theta) * self.circle_R - self.drone_size/2
        self.label_drone.place(x=self.drone_x, y=self.drone_y)
        self.label_drone.configure(bg='white')
        self.label_drone.bind("<ButtonPress-1>", self.onCilckDrone)
        self.label_drone.bind("<B1-Motion>", self.onMoveDrone)
        self.label_drone.bind("<ButtonRelease-1>", self.onReleaseDrone)
        self.move_drone = False

        self.text_aov = tk.Label(self.root, text='camera angle', bg='white', font=('Calibri','10','italic'))
        self.text_aov.place(x=715, y=210)

        # transition and distance
        self.canvas_downright = tk.Canvas(self.root, height=240, width=240, bg='white')
        self.canvas_downright.place(x=640, y=240)

        self.text_time = tk.Label(self.root, text='transition', bg='white', font=('Calibri','10','italic'))
        self.text_time.place(x=690, y=430)
        self.text_slow = tk.Label(self.root, text='SLOW', bg='white', font=('Calibri','7'))
        self.text_slow.place(x=661, y=400)
        self.text_fast = tk.Label(self.root, text='QUICK', bg='white', font=('Calibri','7'))
        self.text_fast.place(x=661, y=250)

        self.text_dis = tk.Label(self.root, text='distance', bg='white', font=('Calibri','10','italic'))
        self.text_dis.place(x=770, y=430)
        self.text_big = tk.Label(self.root, text='CLOSE-UP', bg='white', font=('Calibri','7'))
        self.text_big.place(x=827, y=250)
        self.text_small = tk.Label(self.root, text='LONG SHOT', bg='white', font=('Calibri','7'))
        self.text_small.place(x=823, y=400)

        trough = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00r\xb6\r$\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d\x00\x00\x00\x19tEXtSoftware\x00www.inkscape.org\x9b\xee<\x1a\x00\x00\x00\x15IDAT\x08\x99c\\\xb5j\xd5\x7f\x06\x06\x06\x06&\x06(\x00\x00.\x08\x03\x01\xa5\\\x04^\x00\x00\x00\x00IEND\xaeB`\x82'
        slider = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x000\x00\x00\x000\x08\x06\x00\x00\x00W\x02\xf9\x87\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\x00\x00\tpHYs\x00\x00\x1d\x87\x00\x00\x1d\x87\x01\x8f\xe5\xf1e\x00\x00\x00\x19tEXtSoftware\x00www.inkscape.org\x9b\xee<\x1a\x00\x00\x05\xa4IDATh\x81\xd5\x9a\xcfo\x1bE\x14\xc7?3\xcez\x13\xb7k7\xb1\x9c4\x07\xdaT \xe8\x11*\x0e\xe5\x04A\xaa\x80\x1cP\x13\xe2\xcd\x85\x1eB\xab\x1e\x81\x0b\xe2ohOpD\xa8p(\x95\xaa\x8d\xa0\xad\x04F\xfc\x90\xc2\xad\x1c\xa0\xaa8\x15\xd1\xaa\xa4\x97\xd6?b\xe2\x9f\xb1\xb3\xf6>\x0e\xf6\x86%q\x9a8mj\xf7s\xb2gv\xc7\xdf7\x9e\x9d7\xef\xbdU<\x06\x16\x17\x17\x07\xd2\xe9\xf4\xcb\xa1P\xe8\xb8\xe7yG\x95R\xcf\x01\x07\x80X\xfb\x92\x02\xb0""\xb7\xb5\xd6\xb7\x94R\xd7\xe3\xf1\xf8\xef\x93\x93\x93\x8dG\xfdm\xf5(\xa2s\xb9\xdc\x9b\xc0)\x11y\x0b\xb0\xba\x1c\xa2(")\xa5\xd4W\x89D\xe2\x87\xdd\x1a\xd3\xb5\x01\x8e\xe3\x0c\x01\xa7\x81\x8f\x80C~\xbba\x18\x0c\r\ra\x9a&\x86a\x10\n\x85\xd0Z\x03\xe0y\x1e\xcdf\x13\xd7u\xa9\xd7\xeb\xd4j5\xd6\xd6\xd6\x82\xc3\xde\x13\x91\xf3\xd5j\xf5\xc2\xfc\xfc|m\xcf\x0cXXX\x98\x11\x91O\x80g|\xd1\xd1h\x14\xcb\xb2\x18\x18\x18\xe8f(\x1a\x8d\x06\xa5R\x89b\xb1\x88\xeb\xba~\xf3\x92R\xea\x83d2ym\xa7\xe3\xec\xc8\x80K\x97.\r\x1b\x86q\x01\x98\x060M\x93\x91\x91\x11\xf6\xed\xdb\xd7\x95\xe8N\x88\x08\xd5j\x95|>O\xbd^\xf7\x9b\xbf6\x0c\xe3\xcc\xf4\xf4\xf4\xcav\xf7ok\xc0\xe5\xcb\x97\x8fi\xad\xbf\x01\x0ek\xad\x89\xc7\xe3D\xa3Q\x94\xda\xf5\xe3\xb3%\x85B\x81\xe5\xe5e<\xcf\x03\xb8\x0b\xcc\xd8\xb6}\xf3a\xf7<T\xc5\xc2\xc2\xc2k"r\r\x88\x9a\xa6\xc9\xf8\xf8x\xd7K\xa5[\x1a\x8d\x06\x0f\x1e<\xa0V\xab\x01\x94\x95R\xef$\x93\xc9\x1f\xb7\xba~K\x03\x1c\xc7y\x1dH\x01\xa6eY\x8c\x8e\x8e\xee\xc9\xacwBD\xc8d2\x94J%\x80\x9a\x88L\xcd\xcd\xcd-v\xba\xb6\xa3\xa2\xf6\xb2\xf9\x05\xb0b\xb1\x18\x89Db\xef\xd4n\x81\x88\x90\xcb\xe5(\x14\n\x00E\xe0\xd5N\xcbi\x93\x01W\xae\\9\xe0\xba\xee\r\xe0\xc8\x93\x9e\xf9Nd2\x19\x8a\xc5"\xc0=\xe0%\xdb\xb6\xf3\xc1~\xbd\xf1\x06\xd7u\xbf\x00\x8e\x98\xa6\xd9s\xf1\x00\x89D\x82\xc1\xc1Ah\xf9\x9c\xcf6\xf6\xff\xcf\x00\xc7q\xa6\x81i\xad5\xe3\xe3\xe3=\x17\x0f\xa0\x94bll\xccw\x8a\xb3\x8e\xe3\xbc\x1d\xec_7\xa0\xeda?\x01\x88\xc7\xe3{\xbe\xdbt\x83a\x18\xc4\xe3q\xff\xeb\xa7\xa9T\xca\xf4\xbf\x04\xff\x81\xd3\xc0!\xd34\x89F\xa3OR\xdf\x8e\x88\xc5b\x98\xa6\t0Q.\x97\xdf\xf3\xdb5\xb4\x0ef\xb4\xce6\x0c\x0f\x0f\xf7\xc5\xd2\xe9\xc4\xc8\xc8\x88\xff\xf1\xe3\xb6\xe6\x96\x01\xd9l\xf6\r\xe0\x90a\x18\xec\xdf\xbf\xbfG\xf2\xb6\'\x12\x89\x10\x0e\x87\x01\x0e\xe7r\xb9\x13\xf0\xdf\x12:\x05\xf4\xe5\xd2\t\xa2\x94\nN\xf0)\x00\xb5\xb8\xb88\x90\xcdf\x97\x81\xe8\xc4\xc4D_=\xbc\x9dp]\x97\xa5\xa5%h\x05Iq\x9dN\xa7_\x06\xa2\x86a\xf4\xbdxh\xedH\x86a\x00\xc4<\xcf;\xa6C\xa1\xd0q\x80\xa1\xa1\xa1\xde*\xeb\x82H$\x02\x80R\xea\x15\xedy\xdeQ\xc0\xdf\xa2\x9e\n\xda\xff\x00\xc0\x0b\xba\x1d\x80\x07\x1b\xfb\x9e\xf6N\x84R\xeay\r\x0c\x03\x84B\xa1^j\xea\x8a\x80\xd6\x03\x1a\xd8\x0f\xf4\xad\xf3\xeaD@\xab\xb5\xe94\xfa\xb4\xa1\x812\xb4\x02\x88\xa7\x85\x80\xd6\x92\x06\xfe\x01h6\x9b=\x13\xd4-\x01\xad\xffh\x11\xb9\r\x04s3}\x8f\xafUD\xfe\xd2Z\xeb[@0\'\xd3\xf7\x04\xb4\xfe\xa9\x9b\xcd\xe6\xaf\x00\xab\xab\xab\xbdS\xd4%\xbeV\x11\xb9\xae\xc7\xc6\xc6~\x03\n\xae\xeb\xd2h<r\xb2x\xcfi4\x1a\xfe\x12Z\xd1Z\xdf\xd0\x93\x93\x93\r\x11\xf9\x1e\xf0\xf30}M@c\xca\xb6\xed\xa6\xef\x07.\x02~\xfa\xa2o\x11\x91\xa0\x01\x17\xa1\x1d\xd0\x8c\x8e\x8e\xfe\x08\xdcs]\x97J\xa5\xd2#y\xdbS\xa9T\xfc\xb4\xfc\xdf\x89D\xe2gh\x1b\xd0^F\xe7\x01\xf2\xf9\xfc\xd6#\xf4\x18_\x9b\x88\x9c\xf3\x0b"\xebG\x89j\xb5z\x01X\xaa\xd7\xeb~:\xaf\xafXYY\xf1g\xff\xaeeY_\xfa\xed\xeb\x06\xcc\xcf\xcf\xd7D\xe4C\x80\xe5\xe5\xe5\xberl\x8dFc}\xf6\x95R\xefOMM\xad;\x82\xff\x1d\xe6\xe6\xe6\xe6\xae\x02_{\x9eG:\x9d\xee\x8b\xf3\x91\x88p\xff\xfe}<\xcfC)\xe5$\x93\xc9o\x83\xfd\x9bN\xa3\x86a\x9c\x01\xee\xd6j\xb5\xbe0"\x93\xc9\xf8\x9e\xf7\x8e\x88\x9c\xdd\xd8\xbf\xc9\x80vYg\x06(\x96\xcber\xb9\\O\x8c\x10\x11\xb2\xd9\xac\xbfm\x16\xb4\xd63\xb6moz8;\xc6\x03\xb6m\xdf\x14\x91\x93@\xadP(\x90\xc9d\x9e\xa8\x11"B:\x9d\xf67\x93\x9aR\xea\xe4\xec\xec\xec\x1f\x9d\xae\xddI\x89\xe9*\x103M\x93\x83\x07\x0f\xeey\xec\xbc\xb1\xc4D\xabN\xf6\xd3V\xd7o\x1bG:\x8e\xf3"\xf0\rp\xc4/\xf2\xc5b\xb1\xedn\xdb\x15\x1b\x8a|w\xb4\xd63[\xcd\xbc\xcf\x8e\x02\xe1v\xd5\xe6s`\x16\xfe+\xb3F"\x91G\x8e\xa5E\x84J\xa5B>\x9f_/~+\xa5\x1c\x119\xdbi\xcdo\xa4\xab_o\x17\x17>\x05&\xa0\x95\xde\xb0,kW\x85n\xd7u)\x97\xcb\x94J\xa5`\xd5\xfe\xaeR\xea\xfd\x8d[\xe5\xc3\xe8z\xfaR\xa9\x94\xd9\xce\xcf\x7f\x0c\x1c\xf6\xdb\xc3\xe10\x83\x83\x83\x84\xc3a\xc2\xe1\xf0\xfa\xab\x06"\x82\x88\xac\x1f\x83\xd7\xd6\xd6X]]\xdd\xe8(\xff\x16\x91s\x96e}\x19tR{b\x80O\xfbe\x8f\x13"\xf2.0E\xeb\xed\x94nX\x11\x91\xef\x94R_\x01?\xd9\xb6\xbd\xab\xa0\xfc\xb1$\x83\x1c\xc7\ty\x9ewL)\xf5\x8aR\xea(\xf0,\x10\x07\xfc|}\x11X\xa6\xe5\x8cn\x89\xc8u\xad\xf5\x8d\xdd\x8a\x0e\xf2/#\xf8\x81 \xf2;_\x08\x00\x00\x00\x00IEND\xaeB`\x82'
        style = ttk.Style(self.root)
        path_slider = os.path.join(rospack.get_path('gui'),'assets','circle1.png')
        img_slider = tk.PhotoImage(file=path_slider, format='png')
        img_trough = tk.PhotoImage(master=self.root, data=trough)
        style.element_create('custom.Scale.trough', 'image', img_trough)
        style.element_create('custom.Scale.slider', 'image', img_slider)
        style.layout('custom.Vertical.TScale',
             [('custom.Scale.trough', {'sticky': 'ns'}),
              ('custom.Scale.slider',
               {'side': 'top', 
                'sticky': '',
                # uncomment the following line to show value on the slider
                # 'children': [('custom.Vertical.Scale.label', {'sticky': ''})]
                })])
        # style.configure('custom.Vertical.TScale', background='white', font=('Calibri','10'))
        style.configure('custom.Vertical.TScale', background='white')
        style.map('custom.Vertical.TScale', background=[('active','white')])
        
        self.scale_time = CustomScale(self.root, style=style, init_value=(self.min_time+self.max_time)/2,from_=self.min_time, to=self.max_time,
                        length=160)
        self.scale_time.place(x=696, y=260)

        self.scale_distance = CustomScale(self.root, style=style,init_value=(self.min_dis+self.max_dis)/2,
                    from_=self.min_dis, to=self.max_dis, length=160)
        self.scale_distance.place(x=776, y=260)

        self.scale_distance.bind("<ButtonPress-1>", self.onClickScale)
        self.scale_distance.bind("<B1-Motion>", self.onMoveScale)
        self.scale_distance.bind("<ButtonRelease-1>", self.onReleaseScale)
        self.move_scale = False

        # default shot params
        self.shot_param = ShotParams()
        self.shot_param.image_x = self.rect_x
        self.shot_param.image_y = self.rect_y
        self.shot_param.theta = self.drone_theta
        self.shot_param.transition = (self.min_time+self.max_time)/2
        self.shot_param.distance = 2.0 #(self.min_dis+self.max_dis)/2
        
        #bool variables
        self.rcv_image = False
        self.is_moving = False

        # begin loop
        self.showImage()
        self.root.mainloop()

    def showImage(self):
        if self.rcv_image:
            frame = self.img
            self.canvas_img.create_image(320, 240, image=frame)
            self.canvas_img.image = frame
            self.canvas_img.tag_raise(self.line1)
            self.canvas_img.tag_raise(self.line2)
            self.canvas_img.tag_raise(self.line3)
            self.canvas_img.tag_raise(self.line4)
            self.canvas_img.tag_raise(self.rect)
        self.cnt = self.cnt + 1
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        x1 = x + 880
        y1 = y + 480
        # sc = ImageGrab.grab().crop((x,y,x1,y1))
        # sc_arr = np.array(sc)
        # sc_cv = cv.cvtColor(sc_arr, cv.COLOR_RGB2BGR)
        # sc_msg = self.bridge.cv2_to_imgmsg(sc_cv, encoding="bgr8")
        # if self.cnt is 5:
        #     self.ui_pub.publish(sc_msg)
        #     self.cnt = 0
        # sc.save('./screen.png')
        # print(time.time())
        self.root.after(10, self.showImage)
    
    def imageCallback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, "rgb8")
        pil_img = imagePIL.fromarray(cv_img)
        pil_img = pil_img.resize((640, 480))
        self.img = ImageTk.PhotoImage(pil_img)
        if not self.rcv_image:
            self.rcv_image = True

    def getRectCoordinates(self, cx, cy, width, height):
        x0 = cx - width/2
        y0 = cy - height/2
        x1 = cx + width/2 +1
        y1 = cy + height/2
        return (x0, y0, x1, y1)

    def inRectangle(self, x, y):
        if abs(x - self.rect_x) < self.rect_size and abs(y - self.rect_y) < self.rect_size:
            return True
        else:
            return False
    
    def startMove(self, event):
        if self.inRectangle(event.x, event.y):
            self.is_moving = True
            self.first_x = event.x
            self.first_y = event.y

    def onMove(self, event):
        if self.is_moving:
            self.canvas_img.move(self.rect, event.x - self.first_x, event.y - self.first_y)
            self.rect_x += event.x - self.first_x
            self.rect_y += event.y - self.first_y
            self.first_x = event.x
            self.first_y = event.y
    
    def stopMove(self, event):
        if self.is_moving:
            self.canvas_img.move(self.rect, event.x - self.first_x, event.y - self.first_y)
            self.rect_x += event.x - self.first_x
            self.rect_y += event.y - self.first_y
            self.first_x = event.x
            self.first_y = event.y
            self.is_moving = False
            # publish a message
            self.shot_param.time = rospy.Time.now()
            self.shot_param.image_x = self.rect_x
            self.shot_param.image_y = self.rect_y
            self.shot_param.transition = self.scale_time.variable.get()
            self.shot_pub.publish(self.shot_param)
    
    def onCilckDrone(self, event):
        self.move_drone = True

    def onMoveDrone(self, event):
        if self.move_drone:
            event_x = event.x + self.drone_x
            event_y = event.y + self.drone_y
            center_x = 640 + self.circle_x
            center_y = self.circle_y
            dx = event_x - center_x
            dy = event_y - center_y
            theta = math.atan2(dy, dx)
            new_x = 640 + self.circle_x + math.cos(theta) * self.circle_R - self.drone_size/2
            new_y = self.circle_y + math.sin(theta) * self.circle_R - self.drone_size/2
            self.label_drone.place(x=new_x,y=new_y)
            self.drone_theta = - theta - 0.5 * math.pi
            if self.drone_theta  < - math.pi:
                self.drone_theta += 2 * math.pi
            self.drone_x = new_x
            self.drone_y = new_y
    
    def onReleaseDrone(self, event):
        if self.move_drone:
            center_x = 640 + self.circle_x
            center_y = self.circle_y
            event_x = event.x + self.drone_x
            event_y = event.y + self.drone_y
            dx = event_x - center_x
            dy = event_y - center_y
            theta = math.atan2(dy, dx)
            new_x = 640 + self.circle_x + math.cos(theta) * self.circle_R - self.drone_size/2
            new_y = self.circle_y + math.sin(theta) * self.circle_R - self.drone_size/2
            self.label_drone.place(x=new_x,y=new_y)
            self.drone_theta = - theta - 0.5 * math.pi
            if self.drone_theta  < - math.pi:
                self.drone_theta += 2 * math.pi
            self.drone_x = new_x
            self.drone_y = new_y
            self.move_drone = False
            # publish a message
            self.shot_param.time = rospy.Time.now()
            self.shot_param.theta = self.drone_theta
            self.shot_param.transition = 4.0 #self.scale_time.variable.get()
            self.shot_pub.publish(self.shot_param)
        
    def onClickScale(self, event):
        self.move_scale = True

    def onMoveScale(self, event):
        if self.move_scale:
            dis = self.scale_distance.variable.get()
            #update rectangle
            new_size = self.obj_size * self.f / dis
            scale = new_size / self.rect_size
            self.canvas_img.scale(self.rect, self.rect_x, self.rect_y, scale, scale)
            self.rect_size = new_size
            #update drone 
            ratio = (dis - self.min_dis) / (self.max_dis - self.min_dis)
            R = ratio * (self.R_max - self.R_min) + self.R_min
            scale = R / self.circle_R
            self.canvas_upright.scale(self.circle, self.circle_x, self.circle_y ,scale, scale)
            self.circle_R = R
            self.drone_x = 640 + self.circle_x - math.sin(self.drone_theta) * self.circle_R - self.drone_size/2
            self.drone_y = self.circle_y - math.cos(self.drone_theta) * self.circle_R - self.drone_size/2
            self.label_drone.place(x=self.drone_x, y=self.drone_y)
    
    def onReleaseScale(self, event):
        if self.move_scale:
            dis = self.scale_distance.variable.get()
            #update rectangle
            new_size = self.obj_size * self.f / dis
            scale = new_size / self.rect_size
            self.canvas_img.scale(self.rect, self.rect_x, self.rect_y, scale, scale)
            self.rect_size = new_size
            #update drone 
            ratio = (dis - self.min_dis) / (self.max_dis - self.min_dis)
            R = ratio * (self.R_max - self.R_min) + self.R_min
            scale = R / self.circle_R
            self.canvas_upright.scale(self.circle, self.circle_x, self.circle_y ,scale, scale)
            self.circle_R = R
            self.drone_x = 640 + self.circle_x - math.sin(self.drone_theta) * self.circle_R - self.drone_size/2
            self.drone_y = self.circle_y - math.cos(self.drone_theta) * self.circle_R - self.drone_size/2
            self.label_drone.place(x=self.drone_x, y=self.drone_y)
            self.move_scale = False
            # publish a message
            self.shot_param.time = rospy.Time.now()
            self.shot_param.distance = dis
            self.shot_param.transition = self.scale_time.variable.get()
            self.shot_pub.publish(self.shot_param)
    
def main():
    gui = GUI()

if __name__ == '__main__':
    main()

