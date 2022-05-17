import sys
import ctypes
import torch
from pprint import pprint
from PIL import Image
import glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import CppYCBRenderer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

try:
    from .get_available_devices import *
except:
    from get_available_devices import *

MAX_NUM_OBJECTS = 3
from glutils.utils import colormap

def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture


class TLessTensorRenderer:
    def __init__(self, width=512, height=512, gpu_id=0, render_rotation=False, render_marker=False):
        self.render_marker = render_marker
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.is_textured = []
        self.objects = []
        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.faces = []
        self.poses_trans = []
        self.poses_rot = []
        self.instances = []

        self.render_rotation = render_rotation

        self.r = CppYCBRenderer.CppYCBRenderer(width, height, get_available_devices()[gpu_id])
        self.r.init()

        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders

        self.shaders = shaders
        self.colors = [[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]]
        self.lightcolor = [1, 1, 1]

        vertexShader = self.shaders.compileShader("""
        #version 460
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        uniform vec3 instance_color; 
                
        layout (location=0) in vec3 position;
        layout (location=1) in vec3 normal;
        layout (location=2) in vec2 texCoords;
        out vec2 theCoords;
        out vec3 Normal;
        out vec3 FragPos;
        out vec3 Normal_cam;
        out vec3 Instance_color;
        out vec3 Pos_cam;
        out vec3 Pos_obj;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
            vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
            FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
            Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
            Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
            
            vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
            Pos_cam = pos_cam4.xyz / pos_cam4.w;
            Pos_obj = position;
            
            theCoords = texCoords;
            Instance_color = instance_color;
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader = self.shaders.compileShader("""
        #version 460
        uniform sampler2D texUnit;
        in vec2 theCoords;
        in vec3 Normal;
        in vec3 Normal_cam;
        in vec3 FragPos;
        in vec3 Instance_color;
        in vec3 Pos_cam;
        in vec3 Pos_obj;
        
        layout (location = 0) out vec4 outputColour;
        layout (location = 1) out vec4 NormalColour;
        layout (location = 2) out vec4 InstanceColour;
        layout (location = 3) out vec4 PCObject;
        layout (location = 4) out vec4 PCColour;

        uniform vec3 light_position;  // in world coordinate
        uniform vec3 light_color; // light color

        void main() {
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * light_color;
            vec3 lightDir = normalize(light_position - FragPos);
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * light_color;
        
            outputColour =  texture(texUnit, theCoords) * vec4(diffuse + ambient, 1);
            NormalColour =  vec4((Normal_cam + 1) / 2,1);
            InstanceColour = vec4(Instance_color,1);
            PCObject = vec4(Pos_obj,1);
            PCColour = vec4(Pos_cam,1);
        }
        """, GL.GL_FRAGMENT_SHADER)


        vertexShader_textureless = self.shaders.compileShader("""
        #version 460
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        uniform vec3 instance_color; 
                
        layout (location=0) in vec3 position;
        layout (location=1) in vec3 normal;
        layout (location=2) in vec3 color;
        out vec3 theColor;
        out vec3 Normal;
        out vec3 FragPos;
        out vec3 Normal_cam;
        out vec3 Instance_color;
        out vec3 Pos_cam;
        out vec3 Pos_obj;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
            vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
            FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
            Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
            Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
            
            vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
            Pos_cam = pos_cam4.xyz / pos_cam4.w;
            Pos_obj = position;
            
            theColor = color;
            Instance_color = instance_color;
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader_textureless = self.shaders.compileShader("""
        #version 460
        in vec3 theColor;
        in vec3 Normal;
        in vec3 Normal_cam;
        in vec3 FragPos;
        in vec3 Instance_color;
        in vec3 Pos_cam;
        in vec3 Pos_obj;
        
        layout (location = 0) out vec4 outputColour;
        layout (location = 1) out vec4 NormalColour;
        layout (location = 2) out vec4 InstanceColour;
        layout (location = 3) out vec4 PCObject;
        layout (location = 4) out vec4 PCColour;

        uniform vec3 light_position;  // in world coordinate
        uniform vec3 light_color; // light color

        void main() {
            float ambientStrength = 0.4;
            float diffuseStrength = 0.6;
            vec3 ambient_color = vec3(1,1,1);
            vec3 ambient = ambientStrength * ambient_color;
            vec3 lightDir = normalize(light_position - FragPos);
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = (diff * light_color) * diffuseStrength;
        
            outputColour =  vec4(theColor, 1) * vec4(diffuse + ambient, 1);
            NormalColour =  vec4((Normal_cam + 1) / 2,1);
            InstanceColour = vec4(Instance_color,1);
            PCObject = vec4(Pos_obj,1);
            PCColour = vec4(Pos_cam,1);
        }
        """, GL.GL_FRAGMENT_SHADER)


        vertexShader_simple = self.shaders.compileShader("""
            #version 460
            uniform mat4 V;
            uniform mat4 P;
            
            layout (location=0) in vec3 position;
            layout (location=1) in vec3 normal;
            layout (location=2) in vec2 texCoords;
 
            void main() {
                gl_Position = P * V * vec4(position,1);
            }
            """, GL.GL_VERTEX_SHADER)

        fragmentShader_simple = self.shaders.compileShader("""
            #version 460
            layout (location = 0) out vec4 outputColour;
            layout (location = 1) out vec4 NormalColour;
            layout (location = 2) out vec4 InstanceColour;
            layout (location = 3) out vec4 PCColour;
            void main() {
                outputColour = vec4(0.1, 0.1, 0.1, 1.0);
                NormalColour = vec4(0,0,0,0);
                InstanceColour = vec4(0,0,0,0);
                PCColour = vec4(0,0,0,0);

            }
            """, GL.GL_FRAGMENT_SHADER)

        vertexShader_simple2 = self.shaders.compileShader("""
            #version 450
            uniform mat4 V;
            uniform mat4 P;

            layout (location=0) in vec3 position;
            layout (location=1) in vec3 normal;
            layout (location=2) in vec2 texCoords;
            out vec3 Normal_cam;
            void main() {
                gl_Position = P * V * vec4(position,1);
                Normal_cam = normal;
            }
            """, GL.GL_VERTEX_SHADER)

        fragmentShader_simple2 = self.shaders.compileShader("""
            #version 450
            in vec3 Normal_cam;
            layout (location = 0) out vec4 outputColour;
            layout (location = 1) out vec4 NormalColour;
            layout (location = 2) out vec4 InstanceColour;
            layout (location = 3) out vec4 PCColour;
            void main() {
                outputColour = vec4(Normal_cam, 1.0);
                NormalColour = vec4(0,0,0,0);
                InstanceColour = vec4(0,0,0,0);
                PCColour = vec4(0,0,0,0);
            }
            """, GL.GL_FRAGMENT_SHADER)

        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.shaderProgram_textureless = self.shaders.compileProgram(vertexShader_textureless, fragmentShader_textureless)
        self.shaderProgram_simple = self.shaders.compileProgram(vertexShader_simple, fragmentShader_simple)
        self.texUnitUniform = GL.glGetUniformLocation(self.shaderProgram, 'texUnit')

        self.shaderProgram_simple2 = self.shaders.compileProgram(vertexShader_simple2, fragmentShader_simple2)

        self.lightpos = [0, 0, 0]

        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)
        self.color_tex_5 = GL.glGenTextures(1)

        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_5)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width, self.height, 0,
            GL.GL_DEPTH_STENCIL, GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.color_tex, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self.color_tex_2, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D, self.color_tex_3, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D, self.color_tex_4, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_TEXTURE_2D, self.color_tex_5, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_TEXTURE_2D, self.depth_tex, 0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(5, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1,
                             GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3, GL.GL_COLOR_ATTACHMENT4])

        assert GL.glCheckFramebufferStatus(
            GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) /
                        float(self.height), 0.01, 100)
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.grid = self.generate_grid()

        if render_rotation:
            initial_distribution = np.ones((37 * 72, ), dtype=np.float32)
            # initial_distribution /= np.sum(initial_distribution)

            self.views_vis = np.load('./code/views_vis.npy')

            print(self.views_vis)

            self.rotation_visualization = self.generate_rotation_visualization(initial_distribution)

    def generate_rotation_visualization(self, distribution):
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        vertexData = []

        for i in range(int(self.views_vis.shape[0])):
            if distribution[i] >  0.001:
                x, y, z = self.views_vis[i, 0], self.views_vis[i, 1], self.views_vis[i, 2]
                vertexData.append([-x * (1 - distribution[i]), -y * (1 - distribution[i]), -z * (1 - distribution[i]), 0.1, 0.5, 0.1, 0, 0])
                vertexData.append([-x, -y, -z, 0.5, 0.1, 0.1, 0, 0])

        if len(vertexData) > 0:
            vertexData = np.array(vertexData).astype(np.float32) * 3
            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData,
                            GL.GL_STATIC_DRAW)

            # enable array and set up data
            positionAttrib = GL.glGetAttribLocation(
                self.shaderProgram_simple2, 'position')
            normalAttrib = GL.glGetAttribLocation(
                self.shaderProgram_simple2, 'normal')

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                     None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                     ctypes.c_void_p(12))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

        return VAO

    def generate_grid(self):
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        vertexData = []
        for i in np.arange(-1, 1, 0.05):
            vertexData.append([i, 0, -1, 0, 0, 0, 0, 0])
            vertexData.append([i, 0, 1, 0, 0, 0, 0, 0])
            vertexData.append([1, 0, i, 0, 0, 0, 0, 0])
            vertexData.append([-1, 0, i, 0, 0, 0, 0, 0])

        vertexData = np.array(vertexData).astype(np.float32) * 3
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

        # enable array and set up data
        positionAttrib = GL.glGetAttribLocation(self.shaderProgram_simple, 'position')
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        return VAO

    def load_object(self, obj_path, texture_path, tless=True):
        if texture_path != '':
            is_texture = True
            texture = loadTexture(texture_path)
            self.textures.append(texture)
        else:
            is_texture = False
        self.is_textured.append(is_texture)

        scene = load(obj_path)
        mesh = scene.meshes[0]
        faces = mesh.faces
        pprint(vars(mesh))

        if tless:
            if is_texture:
                vertices = np.concatenate([mesh.vertices/1000.0, mesh.normals, mesh.texturecoords[0, :, :2]], axis=-1)
            else:
                if len(mesh.colors) == 0:
                    mesh.colors = np.expand_dims(np.ones_like(mesh.normals).astype(np.float32), axis=0)
                vertices = np.concatenate([mesh.vertices/1000.0, mesh.normals, mesh.colors[0, :, :3]], axis=-1)
        else:
            if is_texture:
                vertices = np.concatenate([mesh.vertices, mesh.normals, mesh.texturecoords[0, :, :2]], axis=-1)
            else:
                if len(mesh.colors) == 0:
                    mesh.colors = np.expand_dims(np.ones_like(mesh.normals).astype(np.float32), axis=0)
                vertices = np.concatenate([mesh.vertices, mesh.normals, mesh.colors[0, :, :3]], axis=-1)

        vertexData = vertices.astype(np.float32)

        release(scene)

        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

        # enable array and set up data
        if is_texture:
            positionAttrib = GL.glGetAttribLocation(self.shaderProgram, 'position')
            normalAttrib = GL.glGetAttribLocation(self.shaderProgram, 'normal')
            coordsAttrib = GL.glGetAttribLocation(self.shaderProgram, 'texCoords')
        else:
            positionAttrib = GL.glGetAttribLocation(self.shaderProgram_textureless, 'position')
            normalAttrib = GL.glGetAttribLocation(self.shaderProgram_textureless, 'normal')
            colorAttrib = GL.glGetAttribLocation(self.shaderProgram_textureless, 'color')

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        GL.glEnableVertexAttribArray(2)

        # the last parameter is a pointer
        if is_texture:
            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, ctypes.c_void_p(12))
            GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 32, ctypes.c_void_p(24))
        else:
            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, ctypes.c_void_p(12))
            GL.glVertexAttribPointer(colorAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, ctypes.c_void_p(24))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        self.VAOs.append(VAO)
        self.VBOs.append(VBO)
        self.faces.append(faces)
        self.objects.append(obj_path)
        self.poses_rot.append(np.eye(4))
        self.poses_trans.append(np.eye(4))

    def load_objects(self, obj_paths, texture_paths, colors=[[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0], [0.4, 0, 0]], tless=True):
        self.colors = colors
        for i in range(len(obj_paths)):
            self.load_object(obj_paths[i], texture_paths[i], tless=tless)
            self.instances.append(len(self.instances))

        print(self.instances)
        print(len(self.VAOs))

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)

    def set_camera_default(self):
        self.V = np.eye(4)

    def set_fov(self, fov):
        self.fov = fov
        # this is vertical fov
        P = perspective(self.fov, float(self.width) /
                        float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_projection_matrix(self, w, h, fu, fv, u0, v0, znear, zfar):
        L = -(u0) * znear / fu;
        R = +(w-u0) * znear / fu;
        T = -(v0) * znear / fv;
        B = +(h-v0) * znear / fv;

        P = np.zeros((4, 4), dtype=np.float32);
        P[0, 0] = 2 * znear / (R-L);
        P[1, 1] = 2 * znear / (T-B);
        P[2, 0] = (R+L)/(L-R);
        P[2, 1] = (T+B)/(B-T);
        P[2, 2] = (zfar +znear) / (zfar - znear);
        P[2, 3] = 1.0;
        P[3, 2] = (2*zfar*znear)/(znear - zfar);
        self.P = P

    def set_light_color(self, color):
        self.lightcolor = color

    def render(self, cls_indexes, image_tensor, seg_tensor, pc1_tensor=None, pc2_tensor=None):
        frame = 0
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        if self.render_marker:
            # render some grid and directions
            GL.glUseProgram(self.shaderProgram_simple)
            GL.glBindVertexArray(self.grid)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram_simple, 'V'), 1, GL.GL_TRUE, self.V)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram_simple, 'P'), 1, GL.GL_FALSE, self.P)
            GL.glDrawElements(GL.GL_LINES, 160,
                              GL.GL_UNSIGNED_INT, np.arange(160, dtype=np.int))
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            # end rendering markers

        if self.render_rotation:
            # render some grid and directions
            GL.glUseProgram(self.shaderProgram_simple2)
            GL.glBindVertexArray(self.rotation_visualization)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram, 'V'), 1, GL.GL_TRUE, self.V)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram, 'P'), 1, GL.GL_FALSE, self.P)
            GL.glDrawElements(GL.GL_LINES, 6000,
                              GL.GL_UNSIGNED_INT, np.arange(6000, dtype=np.int))
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)

        for i in range(len(cls_indexes)):
            index = cls_indexes[i]
            is_texture = self.is_textured[index]
            # active shader program
            if is_texture:
                GL.glUseProgram(self.shaderProgram)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'V'), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'P'), 1, GL.GL_FALSE, self.P)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'pose_trans'), 1, GL.GL_FALSE, self.poses_trans[i])
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'pose_rot'), 1, GL.GL_TRUE, self.poses_rot[i])
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram, 'light_position'), *self.lightpos)
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram, 'instance_color'), *self.colors[index])
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram, 'light_color'), *self.lightcolor)
            else:
                GL.glUseProgram(self.shaderProgram_textureless)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_textureless, 'V'), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_textureless, 'P'), 1, GL.GL_FALSE, self.P)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_textureless, 'pose_trans'), 1, GL.GL_FALSE, self.poses_trans[i])
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_textureless, 'pose_rot'), 1, GL.GL_TRUE, self.poses_rot[i])
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_textureless, 'light_position'), *self.lightpos)
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_textureless, 'instance_color'), *self.colors[index])
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_textureless, 'light_color'), *self.lightcolor)

            try:
                if is_texture:
                    # Activate texture
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[self.instances[index]])
                    GL.glUniform1i(self.texUnitUniform, 0)
                # Activate array
                GL.glBindVertexArray(self.VAOs[self.instances[index]])
                # draw triangles
                GL.glDrawElements(GL.GL_TRIANGLES, self.faces[self.instances[index]].size, GL.GL_UNSIGNED_INT, self.faces[self.instances[index]])
            finally:
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

        GL.glDisable(GL.GL_DEPTH_TEST)

        # mapping
        self.r.map_tensor(int(self.color_tex), int(self.width), int(self.height), image_tensor.data_ptr())
        self.r.map_tensor(int(self.color_tex_3), int(self.width), int(self.height), seg_tensor.data_ptr())
        if pc1_tensor is not None:
            self.r.map_tensor(int(self.color_tex_4), int(self.width), int(self.height), pc1_tensor.data_ptr())
        if pc2_tensor is not None:
            self.r.map_tensor(int(self.color_tex_5), int(self.width), int(self.height), pc2_tensor.data_ptr())
         
        '''
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #frame = np.frombuffer(frame,dtype = np.float32).reshape(self.width, self.height, 4)
        frame = frame.reshape(self.height, self.width, 4)[::-1, :]

        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        #normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #normal = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        #normal = normal[::-1, ]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        seg = seg.reshape(self.height, self.width, 4)[::-1, :]

        #pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        # seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)

        #pc = np.stack([pc,pc, pc, np.ones(pc.shape)], axis = -1)
        #pc = pc[::-1, ]
        #pc = (1-pc) * 10

        # points in object coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
        pc2 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc2 = pc2.reshape(self.height, self.width, 4)[::-1, :]
        pc2 = pc2[:,:,:3]

        # points in camera coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
        pc3 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc3 = pc3.reshape(self.height, self.width, 4)[::-1, :]
        pc3 = pc3[:,:,:3]

        return [frame, seg, pc2, pc3]
        '''


    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_poses(self, poses):
        self.poses_rot = [np.ascontiguousarray(
            quat2rotmat(item[3:])) for item in poses]
        self.poses_trans = [np.ascontiguousarray(
            xyz2mat(item[:3])) for item in poses]

    def set_allocentric_poses(self, poses):
        self.poses_rot = []
        self.poses_trans = []
        for pose in poses:
            x, y, z = pose[:3]
            quat_input = pose[3:]
            dx = np.arctan2(x, -z)
            dy = np.arctan2(y, -z)
            # print(dx, dy)
            quat = euler2quat(-dy, -dx, 0, axes='sxyz')
            quat = qmult(quat, quat_input)
            self.poses_rot.append(np.ascontiguousarray(quat2rotmat(quat)))
            self.poses_trans.append(np.ascontiguousarray(xyz2mat(pose[:3])))

    def release(self):
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        GL.glDeleteTextures([self.color_tex, self.color_tex_2,
                             self.color_tex_3, self.color_tex_4, self.depth_tex])
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None

        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        GL.glDeleteTextures(self.textures)
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.poses_trans = []  # GC should free things here
        self.poses_rot = []  # GC should free things here

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3]/v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v/v[-1]
        else:
            return None

    def transform_pose(self, pose):
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def get_num_instances(self):
        return len(self.instances)

    def get_poses(self):
        mat = [self.V.dot(self.poses_trans[i].T).dot(
            self.poses_rot[i]).T for i in range(self.get_num_instances())]
        poses = [np.concatenate(
            [mat2xyz(item), safemat2quat(item[:3, :3].T)]) for item in mat]
        return poses

    def get_egocentric_poses(self):
        return self.get_poses()

    def get_allocentric_poses(self):
        poses = self.get_poses()
        poses_allocentric = []
        for pose in poses:
            dx = np.arctan2(pose[0], -pose[2])
            dy = np.arctan2(pose[1], -pose[2])
            quat = euler2quat(-dy, -dx, 0, axes='sxyz')
            quat = qmult(qinverse(quat), pose[3:])
            poses_allocentric.append(np.concatenate([pose[:3], quat]))
            #print(quat, pose[3:], pose[:3])
        return poses_allocentric

    def get_centers(self):
        centers = []
        for i in range(len(self.poses_trans)):
            pose_trans = self.poses_trans[i]
            proj = (self.P.T.dot(self.V.dot(
                pose_trans.T).dot(np.array([0, 0, 0, 1]))))
            proj /= proj[-1]
            centers.append(proj[:2])
        centers = np.array(centers)
        centers = (centers + 1) / 2.0
        centers[:, 1] = 1 - centers[:, 1]
        centers = centers[:, ::-1]  # in y, x order
        return centers

# from PIL import Image
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import *
# import time
# from plyfile import *
# import glob
#
# def load_image( infilename ) :
#     img = Image.open( infilename )
#     img.load()
#     data = np.asarray( img, dtype="float" )
#     return data
#
#
# def pairwise_distances(x, y=None):
#     '''
#     Input: x is a Nxd matrix
#            y is an optional Mxd matirx
#     Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
#             if y is not given then use 'y=x'.
#     i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
#     '''
#     x_norm = (x ** 2).sum(1).view(-1, 1)
#     if y is not None:
#         y_t = torch.transpose(y, 0, 1)
#         y_norm = (y ** 2).sum(1).view(1, -1)
#     else:
#         y_t = torch.transpose(x, 0, 1)
#         y_norm = x_norm.view(1, -1)
#
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
#     # Ensure diagonal is zero if x=y
#     # if y is None:
#     #     dist = dist - torch.diag(dist.diag)
#     return torch.clamp(dist, 0.0, np.inf)
#
# import ruamel.yaml as yaml
#
# def load_info(path):
#     with open(path, 'r') as f:
#         info = yaml.load(f, Loader=yaml.CLoader)
#         for eid in info.keys():
#             if 'cam_K' in info[eid].keys():
#                 info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape(
#                     (3, 3))
#             if 'cam_R_w2c' in info[eid].keys():
#                 info[eid]['cam_R_w2c'] = np.array(
#                     info[eid]['cam_R_w2c']).reshape((3, 3))
#             if 'cam_t_w2c' in info[eid].keys():
#                 info[eid]['cam_t_w2c'] = np.array(
#                     info[eid]['cam_t_w2c']).reshape((3, 1))
#     return info
#
# def load_gt(path):
#     with open(path, 'r') as f:
#         gts = yaml.load(f, Loader=yaml.CLoader)
#         for im_id, gts_im in gts.items():
#             for gt in gts_im:
#                 if 'cam_R_m2c' in gt.keys():
#                     gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
#                 if 'cam_t_m2c' in gt.keys():
#                     gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
#     return gts
#
#
# if __name__ == '__main__':
#     model_path = sys.argv[1]
#     width = 400
#     height = 400
#
#     class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), \
#                               (255, 255, 0), (255, 0, 255), (0, 255, 255), \
#                               (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
#                               (64, 0, 0), (0, 64, 0), (0, 0, 64)]
#
#     # # renderer
#     # renderer = YCBRenderer(width=width, height=height, render_marker=False)
#     # renderer.set_camera_default()
#
#     for object_idx in [4]:
#
#         # renderer
#         renderer = TLessTensorRenderer(width=width, height=height, render_marker=False)
#         renderer.set_camera_default()
#
#         # object
#         models = ["obj_{:02}".format(object_idx+1)]
#         model_mesh_paths = ['{}/models_cad_color_hd/{}.ply'.format(model_path, cls) for cls in models]
#         model_texture_paths = ['' for cls in models]
#         model_colors = [np.array(class_colors_all[i]) / 255.0 for i in range(len(models))]
#         renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
#         # read the ply file
#         plydata = PlyData.read(model_mesh_paths[0])
#         model_vertices = np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).transpose() / 1000.0
#         color_vertices = np.zeros_like(model_vertices)
#         color_counter = np.zeros_like(model_vertices)
#         model_vertices_kdtree = cKDTree(model_vertices)
#
#         # folder for the images
#         photo_folder = '{}/train_kinect/{}/'.format(model_path, models[0][-2:])
#         object_info = load_info(photo_folder + 'info.yml')
#         object_gt = load_gt(photo_folder + 'gt.yml')
#         images_name = sorted(glob.glob(photo_folder + 'rgb/*.png'))
#
#         for i in range(len(images_name)):
#             K = object_info[i]['cam_K']
#             renderer.set_projection_matrix(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.01, 100)
#
#             R = object_gt[i][0]['cam_R_m2c']
#             t = np.squeeze(object_gt[i][0]['cam_t_m2c'], axis=1) / 1000.0
#             q = mat2quat(R)
#             pose = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
#
#             renderer.set_poses([pose])
#             renderer.set_light_pos([0, 0, 0])
#
#             image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
#             seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
#             pc1_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
#
#             cls_indexes = [0]
#             renderer.render(cls_indexes, image_tensor, seg_tensor, pc1_tensor)
#             image_tensor = image_tensor.flip(0)
#             pc1_tensor = pc1_tensor.flip(0)
#
#             image_np = image_tensor[:,:,:3].cpu().numpy().astype(np.float)
#             seg_np = seg_tensor.cpu().numpy()
#             real_photo = cv2.imread(images_name[i])[..., ::-1].astype(np.float)/255.0
#             real_color = torch.from_numpy(real_photo).view(-1, 3)
#             pcloud = pc1_tensor[:, :, :3].view(-1, 3)
#             pcloud_norm = torch.norm(pcloud, dim=1)
#             pcloud_eff = pcloud[pcloud_norm > 0]
#
#             dist, idx = model_vertices_kdtree.query(pcloud_eff)
#
#             idx_real = np.arange(0, idx.shape[0]) #[dist < 0.02]
#             # idx = idx[dist < 0.02]
#
#
#             color_vertices[idx] += real_color[pcloud_norm > 0][idx_real]
#             color_counter[idx] += [1.0, 1.0, 1.0]
#
#             plt.figure()
#             plt.imshow(np.concatenate((real_photo, image_np, 0.5 * real_photo + 0.5 * image_np), axis=1))
#             plt.show()
#
#             print('object: {:02}, finished {}/{}'.format(object_idx+1, i, len(images_name)))


        # color_vertices = np.divide(color_vertices, color_counter, out=np.zeros_like(color_vertices), where=color_counter != 0)
        #
        # color_vertices_uchar = (color_vertices * 255.0).astype(np.uint8)
        #
        # vertex_data = np.concatenate((np.array([plydata['vertex']['x'],
        #                                         plydata['vertex']['y'],
        #                                         plydata['vertex']['z'],
        #                                         plydata['vertex']['nx'],
        #                                         plydata['vertex']['ny'],
        #                                         plydata['vertex']['nz']]).transpose(),
        #                               color_vertices_uchar), axis=1)
        #
        # vertex_data_tuple = []
        # for j in range(vertex_data.shape[0]):
        #     vertex_data_tuple.append((vertex_data[j, 0],
        #                               vertex_data[j, 1],
        #                               vertex_data[j, 2],
        #                               vertex_data[j, 3],
        #                               vertex_data[j, 4],
        #                               vertex_data[j, 5],
        #                               vertex_data[j, 6],
        #                               vertex_data[j, 7],
        #                               vertex_data[j, 8]))
        #
        # save_dir = '/home/xinke/2ndSSD/TLess/t-less_v2/models_cad_color_hd/{}.ply'.format(models[0])
        #
        # vertex = plydata['vertex']
        # vertex.properties = ()
        # vertex.data = np.array(vertex_data_tuple,
        #     dtype=[('x', 'float'), ('y', 'float'), ('z', 'float'), ('nx', 'float'), ('ny', 'float'), ('nz', 'float'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
        # vertex.properties = (PlyProperty('x', 'float'),
        #                      PlyProperty('y', 'float'),
        #                      PlyProperty('z', 'float'),
        #                      PlyProperty('nx', 'float'),
        #                      PlyProperty('ny', 'float'),
        #                      PlyProperty('nz', 'float'),
        #                      PlyProperty('red', 'uchar'),
        #                      PlyProperty('green', 'uchar'),
        #                      PlyProperty('blue', 'uchar'))
        #
        # plydata.write(save_dir)





    # # test pose: obj 5: 401
    # R = [[-0.90722804, -0.42056504, -0.00791048,], [-0.24042845, 0.53389164, -0.81065029], [0.34515493, -0.73354282, -0.58547787]]
    # R = np.asarray(R, dtype=np.float)
    # t = np.asarray([-6.41927531, -10.69445110, 633.65775810], dtype=np.float)/1000.0
    # q = mat2quat(R)
    # pose = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
    #
    # renderer.set_poses([pose])
    # renderer.set_light_pos([0, 0, 0])
    #
    # image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    # seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    # pc1_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    #
    # cls_indexes = [0]
    #
    # renderer.render(cls_indexes, image_tensor, seg_tensor, pc1_tensor)
    #
    # image_tensor = image_tensor.flip(0)
    # pc1_tensor = pc1_tensor.flip(0)
    #
    # image_np = image_tensor[:,:,:3].cpu().numpy().astype(np.float)
    # seg_np = seg_tensor.cpu().numpy()
    #
    # real_photo = cv2.imread('/home/xinke/2ndSSD/TLess/t-less_v2/train_kinect/05/rgb/0401.png').astype(np.float)/255.0
    #
    # pcloud = pc1_tensor[:, :, :3].view(-1, 3)
    # pcloud_norm = torch.norm(pcloud, dim=1)
    # pcloud_eff = pcloud[pcloud_norm > 0]
    #
    #
    # # load object
    # model_vertices = load(model_mesh_paths[0]).meshes[0].vertices / 1000.0
    # model_color = np.zeros_like(model_vertices)
    #
    # model_vertices_kdtree = cKDTree(model_vertices)
    #
    # _, idx = model_vertices_kdtree.query(pcloud_eff)
    #
    # pcloud_associated = model_vertices[idx]
    #
    # print(pcloud_associated)
    #
    # print(model_vertices)
    #
    # plt.figure()
    # plt.imshow(np.concatenate((real_photo, image_np, 0.5 * real_photo + 0.5 * image_np), axis=1))
    #
    # fig = plt.figure()
    # pcloud = pc1_tensor[:, :, :3].reshape((-1, 3))
    # index = np.where(pcloud[:, 0] != 0)[0]
    # perm = np.random.permutation(np.arange(len(index)))
    # index = index[perm[:3000]]
    # pcloud = pcloud[index, :]
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], color='green')
    # ax.axis('equal')
    #
    # fig = plt.figure()
    # pcloud = pcloud_associated
    # index = np.where(pcloud[:, 0] != 0)[0]
    # perm = np.random.permutation(np.arange(len(index)))
    # index = index[perm[:3000]]
    # pcloud = pcloud[index, :]
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], color='green')
    # ax.axis('equal')
    #
    # fig = plt.figure()
    # pcloud = model_vertices
    # index = np.where(pcloud[:, 0] != 0)[0]
    # perm = np.random.permutation(np.arange(len(index)))
    # index = index[perm[:3000]]
    # pcloud = pcloud[index, :]
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], color='green')
    # ax.axis('equal')


    # plt.show()
