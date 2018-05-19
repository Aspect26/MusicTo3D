import numpy as np
import bpy
import bmesh
import librosa
import traceback

from typing import List
from mathutils import Euler


bl_info = {
    'name': 'Music Terrain',
    'author': 'Julius Flimmel',
    'version': (0, 5, 1),
    'category': 'Game Engine',
    'description': 'Add-on for creating a small interactive game, which generates terrain based on music'
}

# TODO: put the script codes to different files, OMFG


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

    TERRAIN_USE_LOG_SCALE = 'TerrainUseLogScale'
    TERRAIN_WIDTH_MULTIPLIER = 'TerrainSizeMultiplier'
    TERRAIN_HEIGHT_MULTIPLIER = 'TerrainHeightMultiplier'
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
        Property(TERRAIN_USE_LOG_SCALE,
                 bpy.props.BoolProperty(name="LogScale", description='Use logscale for the wave axis?', default=False)),
        Property(TERRAIN_WIDTH_MULTIPLIER,
                 bpy.props.FloatProperty(name='Width multiplier', subtype='UNSIGNED', default=0.2)),
        Property(TERRAIN_HEIGHT_MULTIPLIER,
                 bpy.props.FloatProperty(name='Height multiplier', subtype='UNSIGNED', default=1.0)),
        Property(SONG_DURATION,
                 bpy.props.FloatProperty(name='Duration', description='Duration of the sampled song (in seconds). Zero to load whole song', default=2.0)),
        Property(OFFSET,
                 bpy.props.FloatProperty(name='Offset', description='Offset the song (in seconds)', subtype='UNSIGNED', default=0)),
        Property(EFFECT_SMOOTH,
                 bpy.props.BoolProperty(name='Effect: Smoothing', description='Effect that turns on smoothing', default=True)),
        Property(EFFECT_SMOOTH_AMOUNT,
                 bpy.props.IntProperty(name='Effect: Smoothing amount', description='Amount of smoothing (higher number results in smoother terrain', subtype='UNSIGNED', default=5)),
        Property(EFFECT_ROTATE,
                 bpy.props.BoolProperty(name="Effect: Rotate", description='Add rotation effect. The terrain is rotated along the \'time\' axis', default=False)),
        Property(EFFECT_ROTATE_AMOUNT,
                 bpy.props.FloatProperty(name='Effect: Rotate amount', description='Degrees to rotate by in each step', default=3.0)),
        Property(EFFECT_DETAILED_SMOOTHING,
                 bpy.props.BoolProperty(name='Effect: Detailed smoothing', description='Smoothing which takes multiple smooth levels and averages them', default=False)),
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
            Properties._get(scene, Properties.MESH_NAME), Properties._get(scene, Properties.TERRAIN_USE_LOG_SCALE),
            Properties._get(scene, Properties.TERRAIN_WIDTH_MULTIPLIER),
            Properties._get(scene, Properties.TERRAIN_HEIGHT_MULTIPLIER),
            Properties._get(scene, Properties.SONG_DURATION), Properties._get(scene, Properties.OFFSET),
            Properties._get(scene, Properties.EFFECT_ROTATE), Properties._get(scene, Properties.EFFECT_ROTATE_AMOUNT),
            Properties._get(scene, Properties.EFFECT_SMOOTH), Properties._get(scene, Properties.EFFECT_SMOOTH_AMOUNT),
            Properties._get(scene, Properties.EFFECT_DETAILED_SMOOTHING),
            Properties._get(scene, Properties.EFFECT_DETAILED_SMOOTHING_DEPTH)
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
            config = Properties.get_all(context.scene)
            spectrogram, bpm, average_freqs = SoundUtils.get_song_data(config)

            Blender.clear_scene()
            TerrainGenerator().generate(context, spectrogram, bpm, config)
            player = PlayerGenerator().generate(bpm)
            SunGenerator().generate(average_freqs)
            CameraGenerator().generate(player, Properties.get_all(context.scene).file_path)
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
        self.layout.row().prop(context.scene, Properties.TERRAIN_USE_LOG_SCALE)
        self.layout.row().prop(context.scene, Properties.TERRAIN_WIDTH_MULTIPLIER)
        self.layout.row().prop(context.scene, Properties.TERRAIN_HEIGHT_MULTIPLIER)
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

    def __init__(self, file_path, object_name, mesh_name, use_log_scale, width_multiplier,
                 height_multiplier, duration, offset, effect_rotate, effect_rotate_amount, smoothing,
                 smoothing_amount, detailed_smoothing, detailed_smoothing_depth):
        self.file_path = file_path
        self.object_name = object_name
        self.mesh_name = mesh_name
        self.use_log_scale = use_log_scale
        self.width_multiplier = width_multiplier
        self.height_multiplier = height_multiplier
        self.duration = duration
        self.offset = offset
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

    def __init__(self):
        self._terrain = None
        self._context = None
        self._spectrogram = None
        self._bpm = None
        self._configuration = None

    # TODO: place logic in a separate class (also for all the other generated objects (camera, player, ...))
    def generate(self, context, spectrogram, bpm, configuration: TerrainGeneratorConfiguration):
        self._context = context
        self._configuration = configuration
        self._terrain = Blender.create_blender_object_with_empty_mesh(configuration.object_name,
                                                                      configuration.mesh_name)
        self._spectrogram = spectrogram
        self._bpm = bpm

        self._initialize_blender_object()
        self._create_terrain_mesh_for_object()
        self._add_material()
        self._create_terrain_logic()

    def _initialize_blender_object(self):
        Blender.add_object_to_scene(self._context, self._terrain)
        Blender.set_object_mode_edit()

    def _create_terrain_mesh_for_object(self):
        mesh = self._terrain.data
        bm = bmesh.from_edit_mesh(self._terrain.data)
        vertices = TerrainGenerator._create_terrain_vertices(bm, self._spectrogram, self._configuration)
        TerrainGenerator._create_terrain_faces(bm, vertices)
        bmesh.update_edit_mesh(mesh)

    def _add_material(self):
        """
        This is required because without a material on our terrain, the shaders are not working correctly (for some
        reason)
        """
        empty_material = bpy.data.materials.new(name="EmptyMaterial")
        self._terrain.data.materials.append(empty_material)

    def _create_terrain_logic(self):
        sensor = Blender.create_always_sensor(self._terrain)
        script = Blender.create_script('terrain_script.py',
'''
import bge

cont = bge.logic.getCurrentController()

VertexShader = """
    varying vec4 position;  
    varying vec4 light; 
    
    void main()
    {
        vec3 normalDirection = normalize(gl_NormalMatrix * gl_Normal);
        vec3 lightDirection;
        float attenuation;
 
        // directional light?
        if (0.0 == gl_LightSource[0].position.w) {
           attenuation = 1.0; // no attenuation
           lightDirection = normalize(vec3(gl_LightSource[0].position));
        } else {
           vec3 vertexToLightSource = 
              vec3(gl_LightSource[0].position 
              - gl_ModelViewMatrix * gl_Vertex);
           float distance = length(vertexToLightSource);
           attenuation = 1.0 / distance; // linear attenuation 
           lightDirection = normalize(vertexToLightSource);
 
           if (gl_LightSource[0].spotCutoff <= 90.0) // spotlight?
           {
              float clampedCosine = max(0.0, dot(-lightDirection, 
                 gl_LightSource[0].spotDirection));
              if (clampedCosine < gl_LightSource[0].spotCosCutoff) 
                 // outside of spotlight cone?
              {
                 attenuation = 0.0;
              }
              else
              {
                 attenuation = attenuation * pow(clampedCosine, 
                    gl_LightSource[0].spotExponent);
              }
           }
        }
        vec3 diffuseReflection = attenuation 
           * vec3(gl_LightSource[0].diffuse) 
           * max(0.0, dot(normalDirection, lightDirection));
 
        light = vec4(diffuseReflection, 1.0);
        position = gl_Vertex;
        
        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    }
"""

FragmentShader = """
    #version 120
    uniform float time;
    
    varying vec4 position; 
    varying vec4 light; 
    
    const float HEIGHT_MULTIPLIER = 1.0 / 9.0;
    const int COLOR_RAMPS_COUNT = 7;
    const vec4[COLOR_RAMPS_COUNT] COLOR_RAMPS = vec4[] (
        vec4(0.001, 0.000, 0.000, 1.000), 
        vec4(0.008, 0.007, 0.247, 0.625),
        vec4(0.014, 0.007, 0.247, 0.625),
        vec4(0.040, 0.875, 1.000, 0.539),
        vec4(0.452, 0.008, 0.381, 0.015),
        vec4(0.662, 0.008, 0.381, 0.015),
        vec4(0.770, 0.278, 0.196, 0.035)
    );
    
    vec4 getColorFromColorRamp(float position, int colorRampIndex) {
        vec4 start = vec4(0.0, COLOR_RAMPS[0].yzw);
        vec4 end = vec4(1.0, COLOR_RAMPS[COLOR_RAMPS_COUNT - 1].yzw);
        
        if (colorRampIndex > 0) {
            start = COLOR_RAMPS[colorRampIndex - 1];
        }
        if (colorRampIndex < COLOR_RAMPS_COUNT) {
            end = COLOR_RAMPS[colorRampIndex];
        }
        
        float relativePosition = (position - start.x) / (end.x - start.x);
        return vec4(mix(start.yzw, end.yzw, relativePosition), 1.0);
    }
    
    vec4 getColor() {
        float height = position.z * HEIGHT_MULTIPLIER;
    
        for (int i = 0; i < COLOR_RAMPS_COUNT; ++i) {
            vec4 colorRamp = COLOR_RAMPS[i];
            if (height < colorRamp.x) {
                return getColorFromColorRamp(height, i);
            }
        }
        
        return getColorFromColorRamp(height, COLOR_RAMPS_COUNT);
    }
    
    void main()
    {   
        gl_FragColor = light * getColor();
    }
"""

print('called')
mesh = cont.owner.meshes[0]
for mat in mesh.materials:
    shader = mat.getShader()
    if shader != None:
        if not shader.isValid():
            shader.setSource(VertexShader, FragmentShader, True)
'''
                                       )
        controller = Blender.create_python_controller(self._terrain)
        controller.text = script
        controller.link(sensor)

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
            TerrainGenerator._detailed_smooth_vertices(vertices, configuration.detailed_smoothing_depth)
        elif configuration.smoothing:
            TerrainGenerator._smooth_vertices(vertices, configuration.smoothing_amount)

        return vertices

    @staticmethod
    def _create_vertex_from_spectrogram_point(wavelength, time_step, amplitude, configuration: TerrainGeneratorConfiguration):
        if configuration.use_log_scale:
            x = (np.log(wavelength) * configuration.width_multiplier) if wavelength > 0 else 0
        else:
            x = wavelength * configuration.width_multiplier

        y = time_step * 0.5
        z = amplitude * configuration.height_multiplier
        z = z if z + 1.0 <= 0 else np.log(z + 1.0)

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
                    neighbour_vertices = TerrainGenerator._get_neighbour_vertices(vertices, x, y, level)
                    height_sum += Utils.reduce(lambda vertex, acc: vertex.co[2] + acc, neighbour_vertices) / len(neighbour_vertices)

                smoothed_vertex = vertices[x][y]
                smoothed_vertex.co[2] = height_sum / levels

    @staticmethod
    def _smooth_vertices(vertices: List, smoothing_size: int):
        for x in range(len(vertices)):
            for y in range(len(vertices[0])):
                neighbour_vertices = TerrainGenerator._get_neighbour_vertices(vertices, x, y, smoothing_size)
                height_sum = Utils.reduce(lambda vertex, acc: vertex.co[2] + acc, neighbour_vertices)
                smoothed_vertex = vertices[x][y]
                smoothed_vertex.co[2] = height_sum / len(neighbour_vertices)

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
                face = bm.faces.new((vertices[wavelength][time_step], vertices[wavelength + 1][time_step],
                                    vertices[wavelength][time_step + 1]))
                face.smooth = True
                face = bm.faces.new((vertices[wavelength][time_step + 1], vertices[wavelength + 1][time_step],
                                    vertices[wavelength + 1][time_step + 1]))
                face.smooth = True


class SunGenerator:

    def __init__(self):
        self._lamp = None
        self._lightarray = None

    def generate(self, lightarray):
        self._lightarray = lightarray
        self._create_camera()
        self._create_logic()

    def _create_camera(self):
        self._lamp = Blender.create_directional_light()
        self._lamp.location = (5.0, 5.0, 5.0)

    def _create_logic(self):
        sensor = Blender.create_always_sensor(self._lamp)
        sensor.use_pulse_true_level = True

        controller = Blender.create_python_controller(self._lamp, 'energy_update')
        controller.text = self._create_energy_update_controller_script()

        sensor.link(controller)

    def _create_energy_update_controller_script(self):
        return Blender.create_script('energy.py',
'''
import bge
import math

lightarray = ''' + str(self._lightarray) + '''

# TODO: move this preprocessing to addon
for i in range(len(lightarray)):
    x = lightarray[i]
    x = 1 / (1 + math.exp(-x))
    x = (x - 0.5) * 2
    x = x + 0.2
    if x > 1.0:
        x = 1.0
    lightarray[i] = x

scene = bge.logic.getCurrentScene()
cont = bge.logic.getCurrentController()
light = scene.lights['Directional Light']

offset = -49
time_to_index = 43.06
current_time = math.floor(bge.logic.getClockTime() * time_to_index)

light.energy = lightarray[current_time + offset]
'''
                                     )


class PlayerGenerator:

    def __init__(self):
        self._player_object = None
        self._bpm = None

    def generate(self, bpm):
        self._bpm = bpm
        self._create_player()
        self._create_player_logic()

        return self._player_object

    def _create_player(self):
        self._player_object = Blender.create_sphere('Player')
        self._player_object.location = (6.0, 0.0, 1.0)

    def _create_player_logic(self):
        self._create_logic()
        self._create_shaders()
        self._create_shader_update()

    def _create_logic(self):
        sensor = Blender.create_always_sensor(self._player_object)
        sensor.use_pulse_true_level = True

        controller = Blender.create_python_controller(self._player_object, 'logic')
        controller.text = self._create_logic_controller_script()

        sensor.link(controller)

    def _create_shaders(self):
        sensor = Blender.create_always_sensor(self._player_object)

        controller = Blender.create_python_controller(self._player_object, 'shaders')
        controller.text = self._create_shaders_controller_script()

        sensor.link(controller)

    def _create_shader_update(self):
        sensor = Blender.create_always_sensor(self._player_object)
        sensor.use_pulse_true_level = True

        controller = Blender.create_python_controller(self._player_object, 'shaders_update')
        controller.text = self._create_shaders_update_controller_script()

        sensor.link(controller)

    def _create_logic_controller_script(self):
        return Blender.create_script('player_logic.py',
"""
import bge

def playerLogic():
 
    controller = bge.logic.getCurrentController();
    player = controller.owner
    keyboard = bge.logic.keyboard

    forward_movement = 0.360
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
    def _create_shaders_controller_script(self):
        return Blender.create_script('player_shaders.py',
'''
 
import bge
 
cont = bge.logic.getCurrentController()
 
VertexShader = """
    varying vec4 position;  
    varying vec4 light; 
 
    void main()
    {
        vec3 normalDirection = normalize(gl_NormalMatrix * gl_Normal);
        vec3 lightDirection;
        float attenuation;

        // directional light?
        if (0.0 == gl_LightSource[0].position.w) {
           attenuation = 1.0; // no attenuation
           lightDirection = normalize(vec3(gl_LightSource[0].position));
        } else {
           vec3 vertexToLightSource = 
              vec3(gl_LightSource[0].position 
              - gl_ModelViewMatrix * gl_Vertex);
           float distance = length(vertexToLightSource);
           attenuation = 1.0 / distance; // linear attenuation 
           lightDirection = normalize(vertexToLightSource);

           if (gl_LightSource[0].spotCutoff <= 90.0) // spotlight?
           {
              float clampedCosine = max(0.0, dot(-lightDirection, 
                 gl_LightSource[0].spotDirection));
              if (clampedCosine < gl_LightSource[0].spotCosCutoff) 
                 // outside of spotlight cone?
              {
                 attenuation = 0.0;
              }
              else
              {
                 attenuation = attenuation * pow(clampedCosine, 
                    gl_LightSource[0].spotExponent);
              }
           }
        }
        vec3 diffuseReflection = attenuation 
           * vec3(gl_LightSource[0].diffuse) 
           * max(0.0, dot(normalDirection, lightDirection));

        light = vec4(diffuseReflection, 1.0);
        position = gl_Vertex;

        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    }
"""
 
FragmentShader = """
    #version 120
    uniform float time;

    varying vec4 position; 
    varying vec4 light; 

    void main()
    {   
        float bpm = ''' + str(self._bpm) + ''';
        float beatEvery = 60 / bpm;
        float beatNumber = floor(time / beatEvery);

        vec4 color = vec4(1, 1, 1, 0);
        if (mod(beatNumber, 2) == 1) {
            color = vec4(1, 0, 0, 0);
        } else {
            color = vec4(0, 1, 1, 0);
        }

        gl_FragColor = color * light;
    }
"""

print('called')
mesh = cont.owner.meshes[0]
for mat in mesh.materials:
    shader = mat.getShader()
    if shader != None:
        if not shader.isValid():
            shader.setSource(VertexShader, FragmentShader, True)

        shader.setUniform1f('time', bge.logic.getClockTime())
'''
                                     )
    def _create_shaders_update_controller_script(self):
        return Blender.create_script('player_shaders_update.py',
'''
import bge
cont = bge.logic.getCurrentController()
mesh = cont.owner.meshes[0]
for mat in mesh.materials:
    shader = mat.getShader()
    if shader != None:
        shader.setUniform1f('time', bge.logic.getClockTime())

'''
                                     )


class CameraGenerator:

    _HEIGHT = 5.0
    _MIN_DISTANCE = 10.0
    _MAX_DISTANCE = 15.0
    _DAMPING = 0.03

    def __init__(self):
        self._camera = None
        self._player_object = None
        self._sound_file_path = None

    def generate(self, player_object, sound_file_path):
        self._player_object = player_object
        self._sound_file_path = sound_file_path

        self._create_camera()
        self._create_camera_logic()
        self._create_camera_sound()

    def _create_camera(self):
        self._camera = Blender.create_camera()
        self._camera.location = (7.0, -7.0, 5.0)
        self._camera.rotation_euler = Euler((1.221, 0.0, 0.0), 'XYZ')

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


class SoundUtils:
    """
    Utility functions for audio processing
    """

    @staticmethod
    def get_song_data(configuration):
        spectrogram = []
        bpm = 0
        average_frequency_amplitudes = []
        try:
            duration = configuration.duration if configuration.duration > 0 else None
            waveform, sampling_rate = librosa.load(configuration.file_path,  duration=duration,
                                                   offset=configuration.offset)
            spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
            [bpm] = librosa.beat.tempo(waveform, sampling_rate)
            average_frequency_amplitudes = SoundUtils._compute_average_amplitudes(spectrogram)
        except Exception as e:
            print("ERROR LOADING SONG")
            print(str(e))
            traceback.print_exc()

        return spectrogram, bpm, average_frequency_amplitudes

    @staticmethod
    def _compute_average_amplitudes(spectrogram):
        averages = []
        for time_step in range(len(spectrogram[0])):
            sum = 0
            for wavelength in range(len(spectrogram)):
                sum += spectrogram[wavelength][time_step]

            averages.append(sum / len(spectrogram))

        return averages


class Blender:
    """
    Our wrapper for the calls on Blender's bpy' package because it is unreadable.
    """

    @staticmethod
    def clear_scene():
        for item in bpy.data.objects.values():
            bpy.data.objects.remove(item)

    @staticmethod
    def create_camera(name='Camera'):
        scene = bpy.context.scene
        camera_data = bpy.data.cameras.new(name=name)
        camera = bpy.data.objects.new(name=name, object_data=camera_data)
        scene.objects.link(camera)

        return camera

    @staticmethod
    def create_directional_light():
        scene = bpy.context.scene
        lamp_data = bpy.data.lamps.new(name="Directional light", type='SUN')
        lamp = bpy.data.objects.new(name="Directional Light", object_data=lamp_data)
        scene.objects.link(lamp)

        return lamp

    @staticmethod
    def create_sphere(name='Sphere'):
        scene = bpy.context.scene
        mesh = bpy.data.meshes.new('Basic_Sphere')
        sphere_object = bpy.data.objects.new(name, mesh)
        scene.objects.link(sphere_object)

        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=0.5)
        bm.to_mesh(mesh)
        bm.free()

        return sphere_object

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
    def smooth_shade():
        bpy.ops.mesh.normals_make_consistent()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.shade_smooth()

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
    def create_python_controller(obj, name='controller'):
        bpy.ops.logic.controller_add(type='PYTHON', name=name, object=obj.name)
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
    def create_script(script_name: str, script_text: str):
        script = bpy.data.texts.new(script_name)
        script.from_string(script_text)
        return script

    @staticmethod
    def create_sound(file_path):
        return bpy.data.sounds.load(file_path, check_existing=True)


class Utils:

    @staticmethod
    def reduce(reduce_function, iterable, start=0):
        result = start
        for x in iterable:
            result = reduce_function(x, result)

        return result
