import traceback
from typing import List

import numpy as np

import bpy
import bmesh
import librosa

bl_info = {
    'name': 'Music Terrain',
    'author': 'Julius Flimmel',
    'version': (0, 4, 1),
    'category': 'Add Mesh',
    'description': 'Takes a music file and generates terrain for it based on its spectrogram.'
}


def register():
    bpy.utils.register_class(GenerationOperator)
    bpy.utils.register_class(PropertiesPanel)
    Properties.create()


def unregister():
    bpy.utils.unregister_class(GenerationOperator)
    bpy.utils.unregister_class(PropertiesPanel)


class Properties:
    """
    Specifies global properties that are used to parameterize the terrain generation
    """

    class Property:

        def __init__(self, identifier, blender_property):
            self.identifier = identifier
            self.blender_property = blender_property

    FILE_PATH = 'FilePath'
    OBJECT_NAME = 'ObjectName'
    MESH_NAME = 'MeshName'

    MATERIAL_COLOR_RAMP_SCALE = 'MaterialColorRampScale'

    TERRAIN_USE_LOG_SCALE = 'TerrainUseLogScale'
    TERRAIN_WIDTH_MULTIPLIER = 'TerrainSizeMultiplier'
    TERRAIN_HEIGHT_MULTIPLIER = 'TerrainHeightMultiplier'
    TERRAIN_STEP_MULTIPLIER = 'TerrainStepMultiplier'
    SONG_DURATION = 'TerrainSteps'
    OFFSET = 'TerrainStepsOffset'

    EFFECT_ROTATE = 'EffectRotate'
    EFFECT_ROTATE_AMOUNT = 'EffectRotateAmount'

    EFFECT_SMOOTH = 'EffectSmooth'
    EFFECT_SMOOTH_AMOUNT = 'EffectSmoothAmount'

    EFFECT_DETAILED_SMOOTHING = 'EffectCombinedSmoothing'
    EFFECT_DETAILED_SMOOTHING_DEPTH = 'EffectCombinedSmoothingDepth'

    _PROPERTIES = [
        Property(FILE_PATH,
                 bpy.props.StringProperty(name="File path", description='Path to the music file', subtype='FILE_PATH', default='./song.mp3')),
        Property(OBJECT_NAME,
                 bpy.props.StringProperty(name="Object name", description='Name of the object that will be generated', default='Music Terrain')),
        Property(MESH_NAME,
                 bpy.props.StringProperty(name="Mesh name", description='Name of the mesh component of the generated object', default='Spectrogram Mesh')),
        Property(MATERIAL_COLOR_RAMP_SCALE,
                 bpy.props.FloatProperty(name="Material height scale", description='Multiplier for the material color change steps height', subtype='UNSIGNED', default=9.0)),
        Property(TERRAIN_USE_LOG_SCALE,
                 bpy.props.BoolProperty(name="LogScale", description='Use logscale for the wave axis?', default=True)),
        Property(TERRAIN_WIDTH_MULTIPLIER,
                 bpy.props.FloatProperty(name='Width multiplier', subtype='UNSIGNED', default=3.0)),
        Property(TERRAIN_HEIGHT_MULTIPLIER,
                 bpy.props.FloatProperty(name='Height multiplier', subtype='UNSIGNED', default=0.2)),
        Property(TERRAIN_STEP_MULTIPLIER,
                 bpy.props.FloatProperty(name='Step multiplier', subtype='UNSIGNED', default=0.5)),
        Property(SONG_DURATION,
                 bpy.props.FloatProperty(name='Duration', description='Duration of the sampled song (in seconds). Zero to load whole song', default=2.0)),
        Property(OFFSET,
                 bpy.props.FloatProperty(name='Offset', description='Offset the song (in seconds)', subtype='UNSIGNED', default=0)),
        Property(EFFECT_SMOOTH,
                 bpy.props.BoolProperty(name='Effect: Smoothing', description='Effect that turns on smoothing', default=False)),
        Property(EFFECT_SMOOTH_AMOUNT,
                 bpy.props.IntProperty(name='Effect: Smoothing amount', description='Amount of smoothing (higher number results in smoother terrain', subtype='UNSIGNED', default=3)),
        Property(EFFECT_ROTATE,
                 bpy.props.BoolProperty(name="Effect: Rotate", description='Add rotation effect. The terrain is rotated along the \'time\' axis', default=False)),
        Property(EFFECT_ROTATE_AMOUNT,
                 bpy.props.FloatProperty(name='Effect: Rotate amount', description='Degrees to rotate by in each step', default=3.0)),
        Property(EFFECT_DETAILED_SMOOTHING,
                 bpy.props.BoolProperty(name='Effect: Detailed smoothing', description='Smoothing which takes multiple smooth levels and averages them', default=True)),
        Property(EFFECT_DETAILED_SMOOTHING_DEPTH,
                 bpy.props.IntProperty(name='Effect: Detailed smoothing depth', description='Number of smoothing levels to compute', subtype='UNSIGNED', default=3))
    ]

    @staticmethod
    def create():
        for prop in Properties._PROPERTIES:
            Properties._create_property(prop)

    @staticmethod
    def get_all(scene):
        return TerrainGeneratorConfiguration(
            Properties._get(scene, Properties.FILE_PATH), Properties._get(scene, Properties.OBJECT_NAME),
            Properties._get(scene, Properties.MESH_NAME), Properties._get(scene, Properties.MATERIAL_COLOR_RAMP_SCALE),
            Properties._get(scene, Properties.TERRAIN_USE_LOG_SCALE), Properties._get(scene, Properties.TERRAIN_WIDTH_MULTIPLIER),
            Properties._get(scene, Properties.TERRAIN_HEIGHT_MULTIPLIER), Properties._get(scene, Properties.SONG_DURATION),
            Properties._get(scene, Properties.TERRAIN_STEP_MULTIPLIER), Properties._get(scene, Properties.OFFSET),
            Properties._get(scene, Properties.EFFECT_ROTATE), Properties._get(scene, Properties.EFFECT_ROTATE_AMOUNT),
            Properties._get(scene, Properties.EFFECT_SMOOTH), Properties._get(scene, Properties.EFFECT_SMOOTH_AMOUNT),
            Properties._get(scene, Properties.EFFECT_DETAILED_SMOOTHING), Properties._get(scene, Properties.EFFECT_DETAILED_SMOOTHING_DEPTH)
        )

    @staticmethod
    def _create_property(property: Property):
        setattr(bpy.types.Scene, property.identifier, property.blender_property)

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
            TerrainGenerator().generate(context, Properties.get_all(context.scene))
            player = PlayerGenerator().generate(context)
            Camera().update(player, Properties.get_all(context.scene).file_path)
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
        self.layout.row().prop(context.scene, Properties.TERRAIN_STEP_MULTIPLIER)
        self.layout.row().prop(context.scene, Properties.SONG_DURATION)
        self.layout.row().prop(context.scene, Properties.OFFSET)
        self.layout.row().prop(context.scene, Properties.EFFECT_ROTATE)
        self.layout.row().prop(context.scene, Properties.EFFECT_ROTATE_AMOUNT)
        self.layout.row().prop(context.scene, Properties.EFFECT_SMOOTH)
        self.layout.row().prop(context.scene, Properties.EFFECT_SMOOTH_AMOUNT)
        self.layout.row().prop(context.scene, Properties.EFFECT_DETAILED_SMOOTHING)
        self.layout.row().prop(context.scene, Properties.EFFECT_DETAILED_SMOOTHING_DEPTH)

        self.layout.operator(GenerationOperator.bl_idname, text="Generate Terrain")


class TerrainGeneratorConfiguration:

    def __init__(self, file_path, object_name, mesh_name, material_scale, use_log_scale, width_multiplier,
                 height_multiplier, duration, step_multiplier, offset, effect_rotate, effect_rotate_amount, smoothing,
                 smoothing_amount, detailed_smoothing, detailed_smoothing_depth):
        self.file_path = file_path
        self.object_name = object_name
        self.mesh_name = mesh_name
        self.material_scale = material_scale
        self.use_log_scale = use_log_scale
        self.width_multiplier = width_multiplier
        self.height_multiplier = height_multiplier
        self.duration = duration
        self.offset = offset
        self.step_multiplier = step_multiplier
        self.effect_rotate = effect_rotate
        self.effect_rotate_amount = effect_rotate_amount
        self.smoothing = smoothing
        self.smoothing_amount = smoothing_amount
        self.detailed_smoothing = detailed_smoothing
        self.detailed_smoothing_depth = detailed_smoothing_depth


class TerrainGenerator:
    """
    Class which takes care of terrain generation
    """

    def generate(self, context, configuration: TerrainGeneratorConfiguration):
        spectrogram = SoundUtils.get_spectrogram(configuration)
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
            for time_step in range(len(spectrogram[wavelength])):
                vertex = TerrainGenerator._create_vertex_from_spectrogram_point(wavelength, time_step,
                                                                                spectrogram[wavelength, time_step],
                                                                                configuration)
                row_vertices.append(bm.verts.new(vertex))

            vertices.append(row_vertices)

        if configuration.detailed_smoothing:
            vertices = TerrainGenerator._detailed_smooth_vertices(vertices, configuration.detailed_smoothing_depth)
        elif configuration.smoothing:
            vertices = TerrainGenerator._smooth_vertices(vertices, configuration.smoothing_amount)

        return vertices

    @staticmethod
    def _create_vertex_from_spectrogram_point(wavelength, time_step, amplitude, configuration: TerrainGeneratorConfiguration):
        if configuration.use_log_scale:
            x = (np.log(wavelength) * configuration.width_multiplier) if wavelength > 0 else 0
        else:
            x = wavelength * configuration.width_multiplier

        y = time_step * configuration.step_multiplier
        z = amplitude * configuration.height_multiplier

        vertex = [x, y, z]
        if configuration.effect_rotate:
            rotation_amount = np.deg2rad(configuration.effect_rotate_amount)
            vertex = TerrainGenerator._rotate_vertex_around_y(vertex, time_step * rotation_amount)

        return vertex

    @staticmethod
    def _detailed_smooth_vertices(vertices: List, levels: int):
        for x in range(len(vertices)):
            for y in range(len(vertices[0])):
                height_sum = 0
                for level in range(1, levels + 1):
                    # TODO: extract this to a function
                    neighbour_vertices = TerrainGenerator._get_neighbour_vertices(vertices, x, y, level)
                    height_sum += Utils.reduce(lambda vertex, acc: vertex.co[2] + acc, neighbour_vertices) / len(neighbour_vertices)

                smoothed_vertex = vertices[x][y]
                smoothed_vertex.co[2] = height_sum / levels

        # TODO: so... do we need to return it??
        return vertices

    @staticmethod
    def _smooth_vertices(vertices: List, smoothing_size: int):
        for x in range(len(vertices)):
            for y in range(len(vertices[0])):
                neighbour_vertices = TerrainGenerator._get_neighbour_vertices(vertices, x, y, smoothing_size)
                height_sum = Utils.reduce(lambda vertex, acc: vertex.co[2] + acc, neighbour_vertices)
                smoothed_vertex = vertices[x][y]
                smoothed_vertex.co[2] = height_sum / len(neighbour_vertices)

        # TODO: so... do we need to return it??
        return vertices

    @staticmethod
    def _get_neighbour_vertices(vertices: List, x: int, y: int, size: int) -> List:
        neighbours = []
        negative_half_size = round(size / 2)
        positive_half_size = round(size / 2) if size % 2 == 0 else round(size / 2) + 1

        lower_x = 0 if x - negative_half_size < 0 else x - negative_half_size
        upper_x = len(vertices) if x + positive_half_size > len(vertices) else x + positive_half_size
        lower_y = 0 if y - negative_half_size < 0 else y - negative_half_size
        upper_y = len(vertices[0]) if y + positive_half_size > len(vertices[0]) else y + positive_half_size

        for neighbour_x in range(lower_x, upper_x):
            for neighbour_y in range(lower_y, upper_y):
                neighbours.append(vertices[neighbour_x][neighbour_y])

        return neighbours

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
                bm.faces.new((vertices[wavelength][time_step], vertices[wavelength + 1][time_step],
                             vertices[wavelength][time_step + 1]))
                bm.faces.new((vertices[wavelength][time_step + 1], vertices[wavelength + 1][time_step],
                             vertices[wavelength + 1][time_step + 1]))


class PlayerGenerator:

    def __init__(self):
        self._player_object = None

    def generate(self, context):
        self._create_player()
        self._create_player_logic()

        return self._player_object

    def _create_player(self):
        scene = bpy.context.scene
        mesh = bpy.data.meshes.new('Basic_Sphere')
        self._player_object = bpy.data.objects.new("Player", mesh)
        scene.objects.link(self._player_object)
        self._player_object.location = (6.0, 3.0, 1.0)

        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=0.5)
        bm.to_mesh(mesh)
        bm.free()

    def _create_player_logic(self):
        sensor = self._create_always_sensor()
        controller = self._create_logic_controller()
        sensor.link(controller)

    def _create_always_sensor(self):
        sensor = Blender.create_always_sensor(self._player_object)
        sensor.use_pulse_true_level = True

        return sensor

    def _create_logic_controller(self):
        bpy.ops.logic.controller_add(type='PYTHON', name='logic_controller', object=self._player_object.name)
        controller = self._player_object.game.controllers[-1]
        controller.text = self._create_logic_controller_script()

        return controller

    def _create_logic_controller_script(self):
        script = bpy.data.texts.new('test.py')
        # TODO: consider adding this to a separate file...
        script.from_string(
"""
import bge

def playerLogic():
    
    controller = bge.logic.getCurrentController();
    player = controller.owner
    keyboard = bge.logic.keyboard
    
    forward_movement = 0.3
    left_movement = 0
    right_movement = 0
    
    aKey = bge.logic.KX_INPUT_ACTIVE == keyboard.events[bge.events.AKEY]
    dKey = bge.logic.KX_INPUT_ACTIVE == keyboard.events[bge.events.DKEY]
    spaceKey = bge.logic.KX_INPUT_JUST_ACTIVATED == keyboard.events[bge.events.SPACEKEY]
    
    side_movement = 0.0
    jump = 0.0
    if aKey:
        side_movement += -0.05
    if dKey:
        side_movement += 0.05
    if spaceKey:
        jump += 0.15
    
    player.applyMovement((side_movement, forward_movement, 0.0), True)
    # player.applyRotation((-0.05, 0.0, 0.0), True)
    
playerLogic()
"""
        )
        return script


class Camera:

    _HEIGHT = 5.0
    _MIN_DISTANCE = 10.0
    _MAX_DISTANCE = 15.0
    _DAMPING = 0.03

    def __init__(self):
        self._camera = None
        self._player_object = None
        self._sound_file_path = None

    def update(self, player_object, sound_file_path):
        self._player_object = player_object
        self._sound_file_path = sound_file_path

        self._get_camera()
        self._create_camera_logic()
        self._create_camera_sound()

    def _get_camera(self):
        self._camera = bpy.data.objects['Camera']
        if self._camera is None:
            self._create_new_camera()
        else:
            self._reset_camera()

    def _create_camera_logic(self):
        sensor = Blender.create_always_sensor(self._camera)
        controller = Blender.create_and_controller(self._camera)
        actuator = self._create_logic_actuator()

        controller.link(sensor, actuator)

    def _create_camera_sound(self):
        sensor = Blender.create_startup_sensor(self._camera)
        controller = Blender.create_and_controller(self._camera)
        actuator = self._create_sound_actuator()

        controller.link(sensor, actuator)

    def _create_logic_actuator(self):
        actuator = Blender.create_camera_actuator(self._camera)
        actuator.axis = 'POS_X'
        actuator.height = self._HEIGHT
        actuator.min = self._MIN_DISTANCE
        actuator.max = self._MAX_DISTANCE
        actuator.damping = self._DAMPING
        actuator.object = self._player_object

        return actuator

    def _create_sound_actuator(self):
        actuator = Blender.create_sound_actuator(self._camera)
        actuator.sound = Blender.create_sound(self._sound_file_path)

        return actuator

    def _create_new_camera(self):
        # TODO: implement me
        raise Exception("Creating a new camera object is not implemented, yet. Please do not remove the default camera for now")

    def _reset_camera(self):
        # TODO: these clears doesn't work
        self._camera.game.sensors.items().clear()
        self._camera.game.controllers.items().clear()
        self._camera.game.actuators.items().clear()


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
    def get_spectrogram(configuration):
        spectrogram = []
        try:
            duration = configuration.duration if configuration.duration> 0 else None
            waveform, sampling_rate = librosa.load(configuration.file_path,  duration=duration,
                                                   offset=configuration.offset)
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

    @staticmethod
    def create_always_sensor(obj, name='sensor'):
        bpy.ops.logic.sensor_add(type='ALWAYS', name=name, object=obj.name)
        return obj.game.sensors[-1]

    @staticmethod
    def create_startup_sensor(obj, name='sensor'):
        bpy.ops.logic.sensor_add(type='DELAY', name=name, object=obj.name)
        return obj.game.sensors[-1]

    @staticmethod
    def create_and_controller(obj, name='controller'):
        bpy.ops.logic.controller_add(type='LOGIC_AND', name=name, object=obj.name)
        return obj.game.controllers[-1]

    @staticmethod
    def create_camera_actuator(obj, name='actuator'):
        bpy.ops.logic.actuator_add(type='CAMERA', name=name, object=obj.name)
        return obj.game.actuators[-1]

    @staticmethod
    def create_sound_actuator(obj, name='actuator'):
        bpy.ops.logic.actuator_add(type='SOUND', name=name, object=obj.name)
        return obj.game.actuators[-1]

    @staticmethod
    def create_sound(file_path):
        return bpy.data.sounds.load(file_path, check_existing=True)


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


class Utils:

    @staticmethod
    def reduce(reduce_function, iterable, start=0):
        result = start
        for x in iterable:
            result = reduce_function(x, result)

        return result
