import traceback

import numpy as np

import bpy
import bmesh
import librosa

bl_info = {
    'name': 'Music Terrain',
    'author': 'Julius Flimmel',
    'version': (0, 2, 2),
    'category': 'Add Mesh',
    # 'location': 'View3D > UI panel > Music Terrain',
    'description': 'Takes a song and generates an appropriate terrain for it.'
}


def register():
    bpy.utils.register_class(MusicTo3D)
    bpy.types.INFO_MT_mesh_add.append(add_terrain_mesh_button)


def unregister():
    bpy.utils.unregister_class(MusicTo3D)
    bpy.types.INFO_MT_mesh_add.remove(add_terrain_mesh_button)


def add_terrain_mesh_button(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(MusicTo3D.bl_idname, text="Sound Terrain", icon="MOD_SUBSURF")
    # TODO: add custom icon


class MusicTo3D(bpy.types.Operator):
    """
    TODO: add documentation
    """

    bl_idname = 'music.terrain'
    bl_label = 'Music Terrain'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            generator = TerrainGenerator()
            generator.generate_terrain(context, 'song.mp3')
        except Exception:
            traceback.print_exc()

        return {'FINISHED'}


class TerrainGenerator:
    """
    TODO: documentation
    """

    _MESH_NAME = 'Spectrogram Mesh'
    _OBJECT_NAME = 'Music Terrain'

    def generate_terrain(self, context, song_path):
        spectrogram = SoundUtils.get_spectrogram(song_path)
        blender_object = Blender.create_blender_object_with_empty_mesh(TerrainGenerator._OBJECT_NAME, TerrainGenerator._MESH_NAME)
        material = MaterialFactory.create_material()

        self._initialize_blender_object(context, blender_object, material)
        self._create_terrain_mesh_for_object(blender_object, spectrogram)

    @staticmethod
    def _initialize_blender_object(context, blender_object, material):
        Blender.add_object_to_scene(context, blender_object)
        Blender.set_object_mode_edit()
        Blender.add_material_to_object(blender_object, material)

    @staticmethod
    def _create_terrain_mesh_for_object(blender_object, spectrogram):
        # TODO: refactor this method
        mesh = blender_object.data
        bm = bmesh.from_edit_mesh(mesh)

        vertices = []
        for wavelength in range(len(spectrogram)):
            row_vertices = []
            for time_step in range(len(spectrogram[wavelength])):
                amplitude = spectrogram[wavelength][time_step]
                logscale_wavelength = (np.log(wavelength) * 3) if wavelength > 0 else 0
                # TODO: parameterize if user want logscale, and parameterize the multiplier
                vertex = [logscale_wavelength, time_step, amplitude / 5]

                # TODO:: rotation around Y!
                angle = time_step
                rotation_matrix = np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                                            [0,             1, 0,              0],
                                            [np.sin(angle), 0, np.cos(angle),  0],
                                            [0,             0, 0,              1]])
                vertex_np = np.array([vertex[0], vertex[1], vertex[2], 0])
                vertex_np = vertex_np.dot(rotation_matrix)
                # vertex = [vertex_np[0], vertex_np[1], vertex_np[2]]

                row_vertices.append(bm.verts.new(vertex))

            vertices.append(row_vertices)

        for wavelength in range(len(vertices) - 1):
            for time_step in range(len(vertices[wavelength]) - 1):
                bm.faces.new((vertices[wavelength][time_step], vertices[wavelength][time_step + 1], vertices[wavelength + 1][time_step]))
                bm.faces.new((vertices[wavelength][time_step + 1], vertices[wavelength + 1][time_step + 1], vertices[wavelength + 1][time_step]))

        bmesh.update_edit_mesh(mesh)


class ColorRampElement:

    def __init__(self, position, color):
        self.position = position
        self.color = color


class MaterialFactory:

    _DEFAULT_MATERIAL = {
        'name': 'Terrain material',
        'elements': [
            ColorRampElement(0.001, (0.000, 0.000, 1.000, 1)),
            ColorRampElement(0.008, (0.007, 0.247, 0.625, 1)),
            ColorRampElement(0.014, (0.007, 0.247, 0.625, 1)),
            ColorRampElement(0.040, (0.875, 1.000, 0.539, 1)),
            ColorRampElement(0.452, (0.008, 0.381, 0.015, 1)),
            ColorRampElement(0.662, (0.008, 0.381, 0.015, 1)),
            ColorRampElement(0.770, (0.094, 0.041, 0.000, 1)),
        ]
    }

    @staticmethod
    def create_material():
        material_data = MaterialFactory._DEFAULT_MATERIAL
        material = Blender.create_node_material(material_data['name'])

        texture_coordinate_node = MaterialNodes.TextureCoordinate(material)
        separate_node = MaterialNodes.SeparateXYZ(material)
        divide_node = MaterialNodes.DivideBy(material, 9.0)  # TODO: parameterize this
        color_ramp_node = MaterialFactory._create_color_ramp_node(material, material_data['elements'])
        bsdf_node = MaterialNodes.BSDFDiffuse(material)
        material_output_node = MaterialNodes.MaterialOutput(material, Blender.find_node_in_material(material, MaterialNodes.MaterialOutput.IDENTIFIER))  # TODO: check if it is not None (if it is, the Cycles renderer is off)

        MaterialNodes.link(material, texture_coordinate_node.outputs.object, separate_node.inputs.vector)
        MaterialNodes.link(material, separate_node.outputs.z, divide_node.inputs.value)
        MaterialNodes.link(material, divide_node.outputs.value, color_ramp_node.inputs.factor)
        MaterialNodes.link(material, color_ramp_node.outputs.color, bsdf_node.inputs.color)
        MaterialNodes.link(material, bsdf_node.outputs.bsdf, material_output_node.inputs.surface)

        return material

    @staticmethod
    def _create_color_ramp_node(material, data):
        node = MaterialNodes.ColorRamp(material)
        color_ramp_elements = node.get_elements()

        # There are initially two elements that we need to get rid of (One at the beginning and one at the end). Another
        # problem is, that there needs to be at least one element present. So we delete one now, and second one later.

        color_ramp_elements.remove(color_ramp_elements[0])

        for element_data in data:
            # There are already two elements from the beginning
            element = color_ramp_elements.new(0)
            element.color = element_data.color
            element.position = element_data.position

        color_ramp_elements.remove(color_ramp_elements[len(data)])

        return node


class SoundUtils:
    """
    Utility functions for audio processing
    """

    @staticmethod
    def get_spectrogram(song_path):
        spectrogram = []
        try:
            waveform, sampling_rate = librosa.load(song_path)
            spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
        except Exception as e:
            print("ERROR LOADING SONG")
            print(str(e))
            traceback.print_exc()

        return spectrogram


class Blender:
    """
    Our wrapper for the calls on Blender's bpy' package because it is unreadable.
    """

    @staticmethod
    def create_empty_mesh(mesh_name):
        return bpy.data.meshes.new(mesh_name)

    @staticmethod
    def create_blender_object(object_name, mesh):
        return bpy.data.objects.new(object_name, mesh)

    @staticmethod
    def create_blender_object_with_empty_mesh(object_name, mesh_name):
        mesh = Blender.create_empty_mesh(mesh_name)
        return Blender.create_blender_object(object_name, mesh)

    @staticmethod
    def add_object_to_scene(context, blender_object):
        scene = context.scene
        scene.objects.link(blender_object)
        scene.objects.active = blender_object

    @staticmethod
    def set_object_mode_edit():
        bpy.ops.object.mode_set(mode='EDIT')

    @staticmethod
    def add_material_to_object(blender_object, material):
        blender_object.data.materials.append(material)

    @staticmethod
    def create_node_material(name):
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True

        return material

    @staticmethod
    def create_material_node(material, node_class_name):
        return material.node_tree.nodes.new(node_class_name)

    @staticmethod
    def find_node_in_material(material, node_identifier):
        return material.node_tree.nodes.get(node_identifier)


class NodeOutputs:
    def __init__(self, node):
        self._node = node

    def _get(self, index):
        return self._node.outputs[index]


class NodeInputs:
    def __init__(self, node):
        self._node = node

    def _get(self, index):
        return self._node.inputs[index]


class MaterialNodes:
    """
    Our wrapper for Blender's material nodes, because working directly with them is too inconvenient.
    """

    @staticmethod
    def link(material, from_output, to_input):
        material.node_tree.links.new(from_output, to_input)

    class MaterialNode:

        def __init__(self, material, node_class_name, node=None):
            self._material = material
            self._node = Blender.create_material_node(material, node_class_name) if node is None else node

    class TextureCoordinate(MaterialNode):

        class Outputs(NodeOutputs):
            @property
            def generated(self): return self._get(0)

            @property
            def normal(self): return self._get(1)

            @property
            def uv(self): return self._get(2)

            @property
            def object(self): return self._get(3)

            @property
            def camera(self): return self._get(4)

            @property
            def window(self): return self._get(5)

            @property
            def reflection(self): return self._get(6)

        IDENTIFIER = 'ShaderNodeTexCoord'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)
            self.outputs = MaterialNodes.TextureCoordinate.Outputs(self._node)

    class SeparateXYZ(MaterialNode):

        class Inputs(NodeInputs):
            @property
            def vector(self): return self._get(0)

        class Outputs(NodeOutputs):
            @property
            def x(self): return self._get(0)

            @property
            def y(self): return self._get(1)

            @property
            def z(self): return self._get(2)

        IDENTIFIER = 'ShaderNodeSeparateXYZ'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)
            self.outputs = MaterialNodes.SeparateXYZ.Outputs(self._node)
            self.inputs = MaterialNodes.SeparateXYZ.Inputs(self._node)

    class DivideBy(MaterialNode):

        class Inputs(NodeInputs):
            @property
            def value(self): return self._get(0)

        class Outputs(NodeOutputs):
            @property
            def value(self): return self._get(0)

        IDENTIFIER = 'ShaderNodeMath'

        def __init__(self, material, value):
            super().__init__(material, self.IDENTIFIER)
            self.outputs = MaterialNodes.DivideBy.Outputs(self._node)
            self.inputs = MaterialNodes.DivideBy.Inputs(self._node)

            self._node.operation = 'DIVIDE'
            self._node.inputs[1].default_value = value

    class ColorRamp(MaterialNode):

        class Inputs(NodeInputs):
            @property
            def factor(self): return self._get(0)

        class Outputs(NodeOutputs):
            @property
            def color(self): return self._get(0)

            @property
            def alpha(self): return self._get(1)

        IDENTIFIER = 'ShaderNodeValToRGB'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)
            self.outputs = MaterialNodes.ColorRamp.Outputs(self._node)
            self.inputs = MaterialNodes.ColorRamp.Inputs(self._node)

        def get_elements(self):
            return self._node.color_ramp.elements

    class BSDFDiffuse(MaterialNode):

        class Inputs(NodeInputs):
            @property
            def color(self): return self._get(0)

            @property
            def roughness(self): return self._get(1)

            @property
            def normal(self): return self._get(2)

        class Outputs(NodeOutputs):
            @property
            def bsdf(self): return self._get(0)

        IDENTIFIER = 'ShaderNodeBsdfDiffuse'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)
            self.outputs = MaterialNodes.BSDFDiffuse.Outputs(self._node)
            self.inputs = MaterialNodes.BSDFDiffuse.Inputs(self._node)

    class MaterialOutput(MaterialNode):

        class Inputs(NodeInputs):
            @property
            def surface(self): return self._get(0)

            @property
            def volume(self): return self._get(1)

            @property
            def displacement(self): return self._get(2)

        IDENTIFIER = 'Material Output'

        def __init__(self, material, node):
            super().__init__(material, self.IDENTIFIER, node)
            self.inputs = MaterialNodes.MaterialOutput.Inputs(self._node)
