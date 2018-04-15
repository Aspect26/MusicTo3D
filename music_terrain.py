import traceback

import numpy as np

import bpy
import bmesh
import librosa

bl_info = {
    'name': 'Music Terrain',
    'author': 'Julius Flimmel',
    'version': (0, 3, 1),
    'category': 'Add Mesh',
    'description': 'Takes a music file and generates terrain for it based on its spectrogram.'
}


def register():
    bpy.utils.register_class(GenerationOperator)
    bpy.utils.register_class(PropertiesPanel)


def unregister():
    bpy.utils.unregister_class(GenerationOperator)
    bpy.utils.unregister_class(PropertiesPanel)


class Properties:
    """
    Specifies global properties that are used to parameterize the terrain generation
    """

    FILE_PATH = 'FilePath'
    OBJECT_NAME = 'ObjectName'
    MESH_NAME = 'MeshName'

    MATERIAL_COLOR_RAMP_SCALE = 'MaterialColorRampScale'

    TERRAIN_USE_LOG_SCALE = 'TerrainUseLogScale'
    TERRAIN_WIDTH_MULTIPLIER = 'TerrainSizeMultiplier'
    TERRAIN_HEIGHT_MULTIPLIER = 'TerrainHeightMultiplier'
    TERRAIN_STEP_MULTIPLIER = 'TerrainStepMultiplier'
    TERRAIN_STEPS = 'TerrainSteps'
    TERRAIN_STEPS_OFFSET = 'TerrainStepsOffset'

    EFFECT_ROTATE = 'EffectRotate'
    EFFECT_ROTATE_AMOUNT = 'EffectRotateAmount'

    @staticmethod
    def add_to_scene(scene):
        Properties._add_property_to_scene(scene, Properties.FILE_PATH,
                                          bpy.props.StringProperty(name="File path", description='Path to the music file', subtype='FILE_PATH'),
                                          './song.mp3')
        Properties._add_property_to_scene(scene, Properties.OBJECT_NAME,
                                          bpy.props.StringProperty(name="Object name", description='Name of the object that will be generated'),
                                          'Music Terrain')
        Properties._add_property_to_scene(scene, Properties.MESH_NAME,
                                          bpy.props.StringProperty(name="Mesh name", description='Name of the mesh component of the generated object'),
                                          'Spectrogram Mesh')
        Properties._add_property_to_scene(scene, Properties.MATERIAL_COLOR_RAMP_SCALE,
                                          bpy.props.FloatProperty(name="Material height scale", description='Multiplier for the material color change steps height'),
                                          9.0)
        Properties._add_property_to_scene(scene, Properties.TERRAIN_USE_LOG_SCALE,
                                          bpy.props.BoolProperty(name="LogScale", description='Use logscale for the wave axis?'),
                                          True)
        Properties._add_property_to_scene(scene, Properties.TERRAIN_WIDTH_MULTIPLIER,
                                          bpy.props.FloatProperty(name="Width multiplier"),
                                          3.0)
        Properties._add_property_to_scene(scene, Properties.TERRAIN_HEIGHT_MULTIPLIER,
                                          bpy.props.FloatProperty(name="Height multiplier"),
                                          0.2)
        Properties._add_property_to_scene(scene, Properties.TERRAIN_STEP_MULTIPLIER,
                                          bpy.props.FloatProperty(name='Step multiplier'),
                                          0.5)
        Properties._add_property_to_scene(scene, Properties.TERRAIN_STEPS,
                                          bpy.props.IntProperty(name='Steps', description='Number of the generated steps'),
                                          50)
        Properties._add_property_to_scene(scene, Properties.TERRAIN_STEPS_OFFSET,
                                          bpy.props.IntProperty(name='Steps offset', description='NOT IMPLEMENTED YET'),
                                          0)
        Properties._add_property_to_scene(scene, Properties.EFFECT_ROTATE,
                                          bpy.props.BoolProperty(name="Effect: Rotate", description='Add rotation effect. The terrain is rotated along the \'time\' axis'),
                                          False)
        Properties._add_property_to_scene(scene, Properties.EFFECT_ROTATE_AMOUNT,
                                          bpy.props.FloatProperty(name="Effect: rotate - amount", description='Degrees to rotate by in each step'),
                                          3)

    @staticmethod
    def get_all(scene):
        return TerrainGeneratorConfiguration(
            Properties._get(scene, Properties.FILE_PATH), Properties._get(scene, Properties.OBJECT_NAME),
            Properties._get(scene, Properties.MESH_NAME), Properties._get(scene, Properties.MATERIAL_COLOR_RAMP_SCALE),
            Properties._get(scene, Properties.TERRAIN_USE_LOG_SCALE), Properties._get(scene, Properties.TERRAIN_WIDTH_MULTIPLIER),
            Properties._get(scene, Properties.TERRAIN_HEIGHT_MULTIPLIER), Properties._get(scene, Properties.TERRAIN_STEPS),
            Properties._get(scene, Properties.TERRAIN_STEP_MULTIPLIER), Properties._get(scene, Properties.TERRAIN_STEPS_OFFSET),
            Properties._get(scene, Properties.EFFECT_ROTATE), Properties._get(scene, Properties.EFFECT_ROTATE_AMOUNT)
        )

    @staticmethod
    def _add_property_to_scene(scene, property_identifier, property_value, default_value):
        setattr(bpy.types.Scene, property_identifier, property_value)
        scene[property_identifier] = default_value

    @staticmethod
    def _get(scene, property_name):
        return getattr(scene, property_name)


class GenerationOperator(bpy.types.Operator):
    """
    The operator that generates terrain based on some music file and according to multiple parameters
    """

    bl_idname = 'music.terrain'
    bl_label = 'Music Terrain'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            generator = TerrainGenerator()
            generator.generate_terrain(context, Properties.get_all(context.scene))
        except Exception:
            traceback.print_exc()

        return {'FINISHED'}


class PropertiesPanel(bpy.types.Panel):
    """
    Panel used for user parameterization of the music to 3d operator
    """

    bl_idname = 'music.terrain.panel'
    bl_label = 'Music Terrain'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = 'Music Terrain'

    def draw(self, context):
        self.layout.row().prop(context.scene, Properties.FILE_PATH)
        self.layout.row().prop(context.scene, Properties.OBJECT_NAME)
        self.layout.row().prop(context.scene, Properties.MESH_NAME)
        self.layout.row().prop(context.scene, Properties.MATERIAL_COLOR_RAMP_SCALE)
        self.layout.row().prop(context.scene, Properties.TERRAIN_USE_LOG_SCALE)
        self.layout.row().prop(context.scene, Properties.TERRAIN_WIDTH_MULTIPLIER)
        self.layout.row().prop(context.scene, Properties.TERRAIN_HEIGHT_MULTIPLIER)
        self.layout.row().prop(context.scene, Properties.TERRAIN_STEPS)
        self.layout.row().prop(context.scene, Properties.TERRAIN_STEP_MULTIPLIER)
        self.layout.row().prop(context.scene, Properties.TERRAIN_STEPS_OFFSET)
        self.layout.row().prop(context.scene, Properties.EFFECT_ROTATE)
        self.layout.row().prop(context.scene, Properties.EFFECT_ROTATE_AMOUNT)

        self.layout.operator(GenerationOperator.bl_idname, text="Generate Terrain")


class TerrainGeneratorConfiguration:

    def __init__(self, file_path, object_name, mesh_name, material_scale, use_log_scale, width_multiplier,
                 height_multiplier, steps, step_multiplier, steps_offset, effect_rotate, effect_rotate_amount):
        self.file_path = file_path
        self.object_name = object_name
        self.mesh_name = mesh_name
        self.material_scale = material_scale
        self.use_log_scale = use_log_scale
        self.width_multiplier = width_multiplier
        self.height_multiplier = height_multiplier
        self.steps = steps
        self.step_multiplier = step_multiplier
        self.steps_offset = steps_offset
        self.effect_rotate = effect_rotate
        self.effect_rotate_amount = effect_rotate_amount


class TerrainGenerator:
    """
    Class which takes care of terrain generation
    """

    def generate_terrain(self, context, configuration: TerrainGeneratorConfiguration):
        spectrogram = SoundUtils.get_spectrogram(configuration.file_path)
        blender_object = Blender.create_blender_object_with_empty_mesh(configuration.object_name,
                                                                       configuration.mesh_name)
        material = MaterialFactory.create_material(configuration.material_scale)

        self._initialize_blender_object(context, blender_object, material)
        self._create_terrain_mesh_for_object(blender_object, spectrogram, configuration)

    @staticmethod
    def _initialize_blender_object(context, blender_object, material):
        Blender.add_object_to_scene(context, blender_object)
        Blender.set_object_mode_edit()
        Blender.add_material_to_object(blender_object, material)

    @staticmethod
    def _create_terrain_mesh_for_object(blender_object, spectrogram, configuration):
        mesh = blender_object.data
        bm = bmesh.from_edit_mesh(blender_object.data)
        vertices = TerrainGenerator._create_terrain_vertices(bm, spectrogram, configuration)
        TerrainGenerator._create_terrain_faces(bm, vertices)
        bmesh.update_edit_mesh(mesh)

    @staticmethod
    def _create_terrain_vertices(bm, spectrogram, configuration):
        vertices = []
        for wavelength in range(len(spectrogram)):
            row_vertices = []
            for time_step in range(configuration.steps if configuration.steps < len(spectrogram[wavelength]) else len(spectrogram[wavelength])):
                vertex = TerrainGenerator._create_vertex_from_spectrogram_point(wavelength, time_step,
                                                                                spectrogram[wavelength, time_step],
                                                                                configuration)
                row_vertices.append(bm.verts.new(vertex))

            vertices.append(row_vertices)

        return vertices

    @staticmethod
    def _create_vertex_from_spectrogram_point(x, y, z, configuration: TerrainGeneratorConfiguration):
        """
        :param x: wavelength
        :param y: time step
        :param z: amplitude
        :return:
        """
        if configuration.use_log_scale:
            x = (np.log(x) * configuration.width_multiplier) if x > 0 else 0
        else:
            x = x * configuration.width_multiplier

        z = z * configuration.height_multiplier
        y = y * configuration.step_multiplier

        vertex = [x, y, z]
        if configuration.effect_rotate:
            rotation_amount = np.deg2rad(configuration.effect_rotate_amount)
            vertex = TerrainGenerator._rotate_vertex_around_y(vertex, y * rotation_amount)

        return vertex

    @staticmethod
    def _rotate_vertex_around_y(vertex, angle):
        """
        Works only with logscale.
        """
        _X_DISTANCE_FROM_ORIGIN = 7
        cos = np.cos(angle)
        sin = np.sin(angle)
        rotation_matrix = np.array([[cos, 0, -sin, 0],
                                    [0, 1, 0, 0],
                                    [sin, 0, cos, 0],
                                    [0, 0, 0, 1]])
        np_vertex = np.array([vertex[0] - _X_DISTANCE_FROM_ORIGIN, vertex[1], vertex[2], 0])
        np_vertex = np_vertex.dot(rotation_matrix)
        return [np_vertex[0] + _X_DISTANCE_FROM_ORIGIN, np_vertex[1], np_vertex[2]]

    @staticmethod
    def _create_terrain_faces(bm, vertices):
        for wavelength in range(len(vertices) - 1):
            for time_step in range(len(vertices[wavelength]) - 1):
                bm.faces.new((vertices[wavelength][time_step], vertices[wavelength][time_step + 1],
                             vertices[wavelength + 1][time_step]))
                bm.faces.new((vertices[wavelength][time_step + 1], vertices[wavelength + 1][time_step + 1],
                             vertices[wavelength + 1][time_step]))


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
    def create_material(color_ramp_scale):
        material_data = MaterialFactory._DEFAULT_MATERIAL
        material = Blender.create_node_material(material_data['name'])

        texture_coordinate_node = MaterialNodes.TextureCoordinate(material)
        separate_node = MaterialNodes.SeparateXYZ(material)
        divide_node = MaterialNodes.DivideBy(material, color_ramp_scale)
        color_ramp_node = MaterialFactory._create_color_ramp_node(material, material_data['elements'])
        bsdf_node = MaterialNodes.BSDFDiffuse(material)
        material_output_node = MaterialNodes.MaterialOutput(material, Blender.find_node_in_material(material, MaterialNodes.MaterialOutput.IDENTIFIER))

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


Properties.add_to_scene(bpy.context.scene)
