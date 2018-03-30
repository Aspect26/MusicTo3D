import traceback

import numpy as np

import bpy
import bmesh
import librosa

bl_info = {
    'name': 'Music Terrain',
    'author': 'Julius Flimmel',
    'version': (0, 2, 0),
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


class MaterialFactory:

    _DEFAULT_MATERIAL = {
        'name': 'Terrain material'
    }

    @staticmethod
    def create_material():
        material_data = MaterialFactory._DEFAULT_MATERIAL
        material = Blender.create_node_material(material_data['name'])

        texture_coordinate_node = MaterialNodes.TextureCoordinate(material)
        separate_node = MaterialNodes.SeparateXYZ(material)
        divide_node = MaterialNodes.DivideBy(material, 9.0)  # TODO: parameterize this
        color_ramp_node = MaterialNodes.ColorRamp(material)
        bsdf_node = MaterialNodes.BSDFDiffuse(material)
        material_output_node = MaterialNodes.MaterialOutput(material, Blender.find_node_in_material(material, MaterialNodes.MaterialOutput.IDENTIFIER))  # TODO: check if it is not None (if it is, the Cycles renderer is off)

        MaterialNodes.link(material, texture_coordinate_node.output(MaterialNodes.TextureCoordinate.Outputs.Object), separate_node.input(MaterialNodes.SeparateXYZ.Inputs.Vector))
        MaterialNodes.link(material, separate_node.output(MaterialNodes.SeparateXYZ.Outputs.Z), divide_node.input(MaterialNodes.DivideBy.Inputs.VALUE))
        MaterialNodes.link(material, divide_node.output(MaterialNodes.DivideBy.Outputs.VALUE), color_ramp_node.input(MaterialNodes.ColorRamp.Inputs.Factor))
        MaterialNodes.link(material, color_ramp_node.output(MaterialNodes.ColorRamp.Outputs.Color), bsdf_node.input(MaterialNodes.BSDFDiffuse.Inputs.Color))
        MaterialNodes.link(material, bsdf_node.output(MaterialNodes.BSDFDiffuse.Outputs.BSDF), material_output_node.input(MaterialNodes.MaterialOutput.Inputs.Surface))


        color_ramp_elements = color_ramp_node.get_elements()

        blue_element = color_ramp_elements.new(2); blue_element.color = (0,0,1,1); blue_element.position = 0.000

        light_blue_element = color_ramp_elements.new(2); light_blue_element.color = (0.007, 0.247, 0.625, 1); light_blue_element.position = 0.008
        light_blue_element = color_ramp_elements.new(2); light_blue_element.color = (0.007, 0.247, 0.625, 1); light_blue_element.position = 0.014
        beach_yellow_element = color_ramp_elements.new(2); beach_yellow_element.color = (0.875, 1.0, 0.539, 1); beach_yellow_element.position = 0.04
        grass_green_element = color_ramp_elements.new(2); grass_green_element.color = (0.008, 0.381, 0.015, 1); grass_green_element.position = 0.452
        grass_green_element = color_ramp_elements.new(2); grass_green_element.color = (0.008, 0.381, 0.015, 1); grass_green_element.position = 0.662
        brown_hills_element = color_ramp_elements.new(2); brown_hills_element.color = (0.094, 0.041, 0.0, 1); brown_hills_element.position = 0.770

        color_ramp_elements.remove(color_ramp_elements[0])
        color_ramp_elements.remove(color_ramp_elements[0])

        return material


class SoundUtils:
    """
    TODO: docs
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
    TODO: docs
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


class MaterialNodes:
    """
    TODO: docs
    """

    @staticmethod
    def link(material, from_output, to_input):
        material.node_tree.links.new(from_output, to_input)

    class MaterialNode:

        def __init__(self, material, node_class_name, node=None):
            self._material = material
            self._node = Blender.create_material_node(material, node_class_name) if node is None else node

        def output(self, index):
            return self._node.outputs[index]

        def input(self, index):
            return self._node.inputs[index]

    class TextureCoordinate(MaterialNode):

        class Outputs:
            Generated = 0
            Normal = 1
            UV = 2
            Object = 3
            Camera = 4
            Window = 5
            Reflection = 6

        IDENTIFIER = 'ShaderNodeTexCoord'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)

    class SeparateXYZ(MaterialNode):

        class Inputs:
            Vector = 0

        class Outputs:
            X = 0
            Y = 1
            Z = 2

        IDENTIFIER = 'ShaderNodeSeparateXYZ'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)

    class DivideBy(MaterialNode):

        class Inputs:
            VALUE = 0

        class Outputs:
            VALUE = 0

        IDENTIFIER = 'ShaderNodeMath'

        def __init__(self, material, value):
            super().__init__(material, self.IDENTIFIER)
            self._node.operation = 'DIVIDE'
            self._node.inputs[1].default_value = value

    class ColorRamp(MaterialNode):

        class Inputs:
            Factor = 0

        class Outputs:
            Color = 0
            Alpha = 1

        IDENTIFIER = 'ShaderNodeValToRGB'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)

        def get_elements(self):
            return self._node.color_ramp.elements

    class BSDFDiffuse(MaterialNode):

        class Inputs:
            Color = 0
            Roughness = 1
            Normal = 2

        class Outputs:
            BSDF = 0

        IDENTIFIER = 'ShaderNodeBsdfDiffuse'

        def __init__(self, material):
            super().__init__(material, self.IDENTIFIER)

    class MaterialOutput(MaterialNode):

        class Inputs:
            Surface = 0
            Volume = 1
            Displacement = 2

        IDENTIFIER = 'Material Output'

        def __init__(self, material, node):
            super().__init__(material, self.IDENTIFIER, node)
