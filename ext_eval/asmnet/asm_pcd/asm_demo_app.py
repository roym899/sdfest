"""
GUI demo for Active Shape Model
 Shuichi Akizuki, Chukyo Univ.
 Email: s-akizuki@sist.chukyo-u.ac.jp
"""
import sys
sys.path.append("../")
import open3d as o3
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
import argparse
import cv2
import glob
from math import *
from asm import *
import tkinter
from tkinter import ttk
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import common3Dfunc as c3D


def get_argumets():
    """ Parse arguments from command line
    """

    parser = argparse.ArgumentParser( description='Statistical Shape Model')
    parser.add_argument('--dir', type=str, default='data/',
                        help='path to data')
    parser.add_argument('--intrin', type=str, default='intrinsic.json',
                        help='file name of the camera intrinsic(.json).')
    
    return parser.parse_args()

class App:
    def __init__(self, pcds, intrin_name):
        """ Demo application of asm deformation
        Args:
           pcds(list): a list of o3d point cloud
           intrin_name: filename of intrinsic
        """

        self.window = tkinter.Tk()
        self.window.title("ASM deformation")
        
        # prepare deformation parameters
        self.asm = ActiveShapeModel( pcds )
        ap = self.asm.get_all_projection()
        self.dparam = np.zeros(self.asm.get_n_pcd()) # deformation
        self.scale = 1.0                             # scaling
        
        # prepare image mapper
        self.mapping = c3D.Mapping(intrin_name)

        self.width = 640
        self.height = 480
        
        self.im_l_size = 200
        self.im_l = generate_latent_space_image(ap, self.im_l_size)
        
        # pose
        self.rpy = np.array([0.,0.,0.])
        
        ######################################
        #  Widget layout
        ######################################
        """
           0         1        2
        0| Open  | dparam |  pose |
        1|  pcd  |   =0=  |   =r= |
        2|   :  |   =1=  |   =p= |
        3|   :  |   =2=  |   =y= |
        4|   :  |   =3=  |       |
        5| im_l|   =4=  |       |
        6|  :  |  Reset | close |
        """
        
        # 1st column
        self.dir = None
        self.update_asm = False
        self.b1 = ttk.Button(
                    self.window, text='Open dir', width=15,
                    command=self.button1_clicked).grid(row=0,
                                                       column=0,
                                                       padx=5,
                                                       sticky=tkinter.W)
        
        self.canvas = tkinter.Canvas(self.window, 
                                     width=self.width, 
                                     height=self.height)
        self.canvas.grid(row=1, column=0, rowspan=4)
        
        self.canvas_l = tkinter.Canvas(self.window, 
                                       width=self.im_l_size, 
                                       height=self.im_l_size)
        self.canvas_l.grid(row=5, column=0, rowspan=2)
        self.canvas_l.bind('<ButtonPress-1>', self.start_pickup)


        
        # 2nd colmun (dparam)
        label1 = ttk.Label(
                            self.window,
                            text='Deformation param',
                            #background='#0000aa',
                            foreground='#000000',
                            padding=(5, 10))
        label1.grid(row=0, column=1)
        
        # slider dparam
        min_param = -40
        max_param = 40
        self.val0 = tkinter.DoubleVar()
        self.val0.set(0)
        self.sc_v0 = ttk.Scale(
                    self.window,
                    variable=self.val0,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=min_param,
                    to=max_param,
                    command=self.get_p).grid(row=1, column=1)


        self.val1 = tkinter.DoubleVar()
        self.val1.set(0)
        self.sc_v1 = ttk.Scale(
                    self.window,
                    variable=self.val1,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=min_param,
                    to=max_param,
                    command=self.get_p).grid(row=2, column=1)  
        
        self.val2 = tkinter.DoubleVar()
        self.val2.set(0)
        self.sc_v2 = ttk.Scale(
                    self.window,
                    variable=self.val2,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=min_param,
                    to=max_param,
                    command=self.get_p).grid(row=3, column=1) 
        
        label2 = ttk.Label(
                    self.window,
                    text='Scaling param',
                    #background='#0000aa',
                    foreground='#000000',
                    padding=(5, 10))
        label2.grid(row=4, column=1)
        
        self.val4 = tkinter.DoubleVar()
        self.val4.set(1)
        self.sc_v4 = ttk.Scale(
                    self.window,
                    variable=self.val4,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=0.8,
                    to=1.2,
                    command=self.get_p).grid(row=5, column=1)
        

        
        self.label_description = ttk.Label(self.window, textvariable=self.dparam[0])
        self.label_description.grid(row=6, column=1)
        
        # Close button
        self.close_btn = tkinter.Button(self.window, text="Reset")
        self.close_btn.grid(row=6, column=1) 
        self.close_btn.configure(command=self.reset)
        
        # 3rd column (Pose)
        label2 = ttk.Label(
                            self.window,
                            text='Pose',
                            #background='#0000aa',
                            foreground='#000000',
                            padding=(5, 10))
        label2.grid(row=0, column=2)
        
        # slider (pose)
        self.val_r = tkinter.DoubleVar()
        self.val_r.set(0)
        self.sc_r = ttk.Scale(
                    self.window,
                    variable=self.val_r,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=-np.pi,
                    to=np.pi,
                    command=self.get_p).grid(row=1, column=2)  
        
        self.val_p = tkinter.DoubleVar()
        self.val_p.set(0)
        self.sc_p = ttk.Scale(
                    self.window,
                    variable=self.val_p,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=-np.pi,
                    to=np.pi,
                    command=self.get_p).grid(row=2, column=2)  
        
        self.val_y = tkinter.DoubleVar()
        self.val_y.set(0)
        self.sc_y = ttk.Scale(
                    self.window,
                    variable=self.val_y,
                    orient=tkinter.HORIZONTAL,
                    length=200,
                    from_=-np.pi,
                    to=np.pi,
                    command=self.get_p).grid(row=3, column=2) 
        
        # Close button
        self.close_btn = tkinter.Button(self.window, text="Close")
        self.close_btn.grid(row=6, column=2) 
        self.close_btn.configure(command=self.destructor)
        

        self.delay = 10
        self.update()

        self.window.mainloop()

    def start_pickup(self, event):
        """ Pick up clicked coordinate
        Args:
          event: .x, .y is location of pointer on the canvas
        """
        offset = self.im_l_size/2 # offset (0,0) to image center
        self.dparam[0] = event.x-offset
        self.dparam[1] = self.im_l_size-event.y-offset
        self.val0.set(event.x-offset)
        self.val1.set(self.im_l_size-event.y-offset)
        
    def get_p(self,n):
        """ Update parameters
        """
        self.dparam[0] = self.val0.get()
        self.dparam[1] = self.val1.get()
        self.dparam[2] = self.val2.get()
        self.scale = self.val4.get()
        self.rpy[0] = self.val_r.get()
        self.rpy[1] = self.val_p.get()
        self.rpy[2] = self.val_y.get()
        #print(self.dparam)

    def update(self):
        # Generate another ASM when needed 
        if self.update_asm:
            self.update_asm = False
            fl = sorted(glob.glob( osp.join(self.dir,"*.pcd")))
            cloud_train = list()
            for name in fl:
                #print("load:",name)
                cloud_m = o3.io.read_point_cloud( name )
                cloud_train.append(cloud_m)
            if len(cloud_train) != 0:
                self.asm = ActiveShapeModel( cloud_train )
                ap = self.asm.get_all_projection()
                self.im_l = generate_latent_space_image(ap, self.im_l_size)
                self.reset()

        # Deformation 
        deformed = self.asm.deformation( self.dparam )
        deformed.scale( self.scale, [0,0,0])
        # Rotation
        y = c3D.RPY2Matrix4x4( 0, self.rpy[0], 0 )[:3,:3]
        x = c3D.RPY2Matrix4x4( self.rpy[1], 0, 0 )[:3,:3]
        rot = np.dot( x, y )
        v = np.array([0.,0.,-1.])
        rot_v = np.dot(x,v)
        q = np.hstack([self.rpy[2],rot_v]) 
        q = q/LA.norm(q) 
        pose = c3D.quaternion2rotation(q)
        rot = np.dot(pose,rot)
        deformed.rotate(rot)
        deformed.translate([0.,0.,5.])
        img = self.mapping.Cloud2Image(deformed, dbg_vis=True)

        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img), master=self.window)
        self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW )
        
        # Generate latent space image
        im_l2 = self.im_l.copy()
        pix = self.dparam[0:2] + self.im_l_size/2
        cv2.circle( im_l2, 
                    (int(pix[0]),int(self.im_l_size-pix[1])), 
                    3, (255,255,255), -1, cv2.LINE_AA )
        cv2.circle( im_l2, 
                    (int(pix[0]),int(self.im_l_size-pix[1])), 
                    2, (255,0,0), -1, cv2.LINE_AA )
        
        self.photo_l = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(im_l2), master=self.window)
        self.canvas_l.create_image(0, 0, image = self.photo_l, anchor = tkinter.NW )

        self.window.after(self.delay, self.update)
        

    def destructor(self):
        self.window.destroy()
        
    def reset(self):
        self.dparam = np.zeros(self.asm.get_n_pcd())
        self.rpy = np.zeros(3)
        self.val0.set(0)
        self.val1.set(0)
        self.val2.set(0)
        self.val4.set(1)
        self.val_r.set(0)
        self.val_p.set(0)
        self.val_y.set(0)
        
    def button1_clicked(self):
        v = filedialog.askdirectory(initialdir='./')
        if v:
            self.dir = v
            print(self.dir)
            self.update_asm = True

if __name__ == "__main__":

    args = get_argumets()

    # get file list
    fl = sorted(glob.glob( osp.join( args.dir,"*.pcd")))
    fl2 = sorted(glob.glob( osp.join( args.dir,"*.ply")))
    fl = fl+fl2

    cloud_train = []
    for name in fl:
        cloud_m = o3.io.read_point_cloud( name )
        cloud_train.append(cloud_m)
    print(len(cloud_train))

    if len(cloud_train) == 0:
        print("Point cloud is not found.")
        exit()

    App( cloud_train, args.intrin )
    

