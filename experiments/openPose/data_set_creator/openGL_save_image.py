import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glm
import numpy as np
from PIL import Image, ImageOps
from pyrr import Matrix44, Vector4, Vector3, Quaternion
import pyrr
import argparse
import os
import xml.dom.minidom

VERT_DATA = np.array([1.0, 1.0, 0.0,
                        1.0, -1.0, 0.0,
                        -1.0, -1.0, 0.0,
                        -1.0, 1.0, 0.0],
                        dtype="float32")


TEXTURE_COORD_DATA = np.array([1.0, 1.0,
                                1.0, -1.0,
                                -1.0, -1.0,
                                -1.0, 1.0],
                                dtype="float32")

INDICES = np.array([0, 1, 3,
                    1, 2, 3],
                    dtype="int32")

WINDOW_WIDTH, WINDOW_HEIGHT = 768, 576

# camera params
FAR_CLIP = 2500.0
NEAR_CLIP = 2.0
FOV = 45.0

ORIGIN = np.array([-4.21425, 105.008, 327.119], dtype="float32")
TARGET = np.array([-4.1969, 104.951, 326.12], dtype="float32")
UP = np.array([0.0, 1.0, 0.0], dtype="float32")

# RECT PARAMS
RECT_SCALE = Vector3([15.0, 30.0, 1.0])
RECT_TRANSLATE = Vector3([0.0, 110.0, 15.0])

BG_TEXTURE_PATH = 'master_v2.jpg'


class GLProgram:
    def __init__(self, x = 50.0, y=0.0, z =-50, angle=1.5):
        self.gl_program = glCreateProgram()

        self.shaders()
        self.gl_buffers()
        self.mvp_matrix = self.compute_mvp(Vector3([x, y, z]), angle)
        self.gl_init()
        self.rendered = False

    def gl_init(self):
        #glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def compute_mvp(self, translation, rotation):
        # model matrix is correct
        identity_matrix = np.identity(4)
        scale_matrix = np.transpose(pyrr.matrix44.create_from_scale(RECT_SCALE))
        trans_matrix = np.transpose(pyrr.matrix44.create_from_translation(RECT_TRANSLATE))
        rot_matrix = np.transpose(pyrr.matrix44.create_from_y_rotation(np.radians(360.0 - rotation)))
        trans_matrix_cur = np.transpose(pyrr.matrix44.create_from_translation(translation))

        model_matrix = identity_matrix
        model_matrix = np.matmul(model_matrix, trans_matrix_cur)
        model_matrix = np.matmul(model_matrix, rot_matrix)
        model_matrix = np.matmul(model_matrix, trans_matrix)
        model_matrix = np.matmul(model_matrix, scale_matrix)


        view_matrix = np.transpose(
            pyrr.matrix44.create_look_at(
                ORIGIN,
                TARGET,
                UP
            )
        )

        proj_matrix = np.transpose(
            pyrr.matrix44.create_perspective_projection(
                FOV,
                WINDOW_WIDTH / WINDOW_HEIGHT,
                NEAR_CLIP,
                FAR_CLIP
            )
        )
        cam_matrix = np.matmul(proj_matrix, view_matrix)
        print(cam_matrix)
        m = np.matmul(cam_matrix, model_matrix)

        return np.transpose(m)

    def shaders(self):
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

        with open("VertexShader.vert", "r") as vert_file:
            vert_source = vert_file.read()
        with open("FragmentShader.frag", "r") as frag_file:
            frag_source = frag_file.read()

        glShaderSource(vertex_shader, vert_source)
        glShaderSource(fragment_shader, frag_source)

        glCompileShader(vertex_shader)
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(vertex_shader)
            print ("Compilation Failure for " + str(vertex_shader) + " shader:\n" + str(info_log))

        glCompileShader(fragment_shader)
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(fragment_shader)
            print ("Compilation Failure for " + str(fragment_shader) + " shader:\n" + str(info_log))

        glAttachShader(self.gl_program, vertex_shader)
        glAttachShader(self.gl_program, fragment_shader)

        glLinkProgram(self.gl_program)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def gl_texture(self, texture_path):
        image = Image.open(texture_path).transpose(Image.FLIP_TOP_BOTTOM)
        image_data =  np.array(list(image.getdata()), np.uint8)


        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)


        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            image.size[0],
            image.size[1],
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            image_data
        )


        image.close()

        return texture

    def gl_buffers(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.pos_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, VERT_DATA, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.text_coord_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.text_coord_vbo)
        glBufferData(GL_ARRAY_BUFFER, TEXTURE_COORD_DATA, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self.pos_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.pos_ebo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.pos_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, INDICES, GL_STATIC_DRAW)

        # not used
        self.background_texture = self.gl_texture(BG_TEXTURE_PATH)

    def ortho_start(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(-WINDOW_WIDTH / 2.0, WINDOW_WIDTH / 2.0, -WINDOW_HEIGHT / 2.0, WINDOW_HEIGHT / 2.0)
        glMatrixMode(GL_MODELVIEW)

    def ortho_end(self):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def background(self):
        glBindTexture( GL_TEXTURE_2D, self.background_texture )

        self.ortho_start()

        glPushMatrix()
        glTranslatef( -WINDOW_WIDTH / 2.0, - WINDOW_HEIGHT / 2.0, 0 )
        glBegin(GL_QUADS)
        glTexCoord2i(0, 0)
        glVertex2i(0, 0)
        glTexCoord2i(1, 0)
        glVertex2i(WINDOW_WIDTH, 0)
        glTexCoord2i(1, 1)
        glVertex2i(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2i(0, 1)
        glVertex2i(0, WINDOW_HEIGHT)
        glEnd()
        glPopMatrix()

        self.ortho_end()

    def display(self):
        self.gl_init()
        glLoadIdentity()
        glEnable( GL_TEXTURE_2D )

        # self.background()

        self.display_rect()

        # send buffer to window
        glutSwapBuffers()

    def display_rect(self):
        glUseProgram(self.gl_program)

        # glActiveTexture(GL_TEXTURE0)
        # glBindTexture(GL_TEXTURE_2D, self.background_texture)
        # texture_uniform = glGetUniformLocation(self.gl_program, "the_texture")
        # glUniform1i(texture_uniform, 0)

        trans_uniform = glGetUniformLocation(self.gl_program, "mvp")
        glUniformMatrix4fv(trans_uniform, 1, GL_FALSE, self.mvp_matrix)

        glBindVertexArray(self.vao)

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glUseProgram(0)

    def save_image_as_mask(self, file_name):
        self.display()

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (WINDOW_WIDTH, WINDOW_HEIGHT), data)
        image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
        image.save(file_name, 'PNG')
        print('Saved')

def main(file_name):
    if not os.path.isfile(file_name):
        print('File does not exist')
        return

    doc = xml.dom.minidom.parse(file_name)

    shape_node_rect = doc.getElementsByTagName('shape')[1]
    transform_node = shape_node_rect.getElementsByTagName('transform')[0]
    translation_node = transform_node.getElementsByTagName('translate')[1]
    rotation_node = transform_node.getElementsByTagName('rotate')[0]

    x = float(translation_node.getAttribute('x'))
    y = float(translation_node.getAttribute('y'))
    z = float(translation_node.getAttribute('z'))

    angle = float(rotation_node.getAttribute('angle'))
    print(f'using: x={x}, y={y}, z={z}, angle={angle}')

    glutInit()
    # Create a double-buffer RGBA window.   (Single-buffering is possible.
    # So is creating an index-mode window.)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutInitWindowPosition(0, 0)
    # titel is importanten so i3 shows it floating
    wind = glutCreateWindow("AX_FLOATING")


    plain_file_name, _ = os.path.splitext(file_name)
    gl = GLProgram(x, y, z, angle)
    gl.display()
    gl.save_image_as_mask(f"{plain_file_name}_mask.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute mask from scenefile')
    parser.add_argument('--filename', type=str, help='filename of the xml file', required=True)
    args = parser.parse_args()

    main(args.filename)