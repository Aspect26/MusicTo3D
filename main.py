import traceback
import numpy as np

import bpy
import bmesh
import librosa

bl_info = {
    'name': 'Music to 3D',
    'category': 'Mesh'
}


class MusicTo3D(bpy.types.Operator):
    """
    Add docs
    """

    bl_idname = 'mesh.music_to_3d'
    bl_label = 'Music To 3D'
    bl_options = {'REGISTER', 'UNDO'}

    terrain_object = None

    def execute(self, context):
        if self.terrain_object is None:
            self.recreate_terrain(context)

        return {'FINISHED'}

    def recreate_terrain(self, context):
        print("SELF: {0}".format(str(self)))
        spectrogram = self.get_spectrogram()
        mesh = self.create_new_mesh()
        self.create_new_object(mesh)
        self.add_object_to_scene(context, self.terrain_object)
        self.add_material_to_object(self.terrain_object)
        self.create_terrain_object(context, spectrogram)

    file_path = bpy.props.StringProperty(name='Song file path', description='desc', subtype='FILE_PATH',
                                         update=recreate_terrain)

    def get_spectrogram(self):
        try:
            self.file_path = "song.mp3"
            print("file_path: {0}".format(self.file_path))
            waveform, sampling_rate = librosa.load(self.file_path)
            print("WAVEFORM: {0}".format(str(waveform)))
            print("WAVEFORM TYPE: {0}".format(type(waveform)))
            print("WAVEFORM LENGTH: {0}".format(len(waveform)))
            print("SAMPLING_RATE: {0}".format(sampling_rate))
            mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
            print("MEL_SPECTROGRAM: {0}".format(str(mel_spectrogram)))
            print("MEL_SPECTROGRAM TYPE: {0}".format(type(mel_spectrogram)))
            print("MEL_SPECTROGRAM LENGTH: {0}".format(len(mel_spectrogram)))
            print("LENGTH: {0}".format(len(mel_spectrogram[0])))
        except Exception as e:
            print("ERROR LOADING SONG")
            print(str(e))
            traceback.print_exc()
            return []

        return mel_spectrogram

    def create_new_mesh(self):
        # TODO: do something about the name, there can be multiple of these mehses
        return bpy.data.meshes.new("m3d")

    def create_new_object(self, mesh):
        # TODO: the name...
        self.terrain_object = bpy.data.objects.new("M3D Object", mesh)

    def add_object_to_scene(self, context, obj):
        scene = context.scene
        scene.objects.link(obj)
        scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

    def add_material_to_object(self, obj):
        material = bpy.data.materials.new(name="Terrain material")
        material.use_nodes = True

        texture_coordinate_node = material.node_tree.nodes.new('ShaderNodeTexCoord')
        separate_node = material.node_tree.nodes.new('ShaderNodeSeparateXYZ')
        material.node_tree.links.new(separate_node.inputs[0], texture_coordinate_node.outputs[3])

        divide_node = material.node_tree.nodes.new('ShaderNodeMath')
        divide_node.inputs[1].default_value = 9.0
        divide_node.operation = 'DIVIDE'
        material.node_tree.links.new(divide_node.inputs[0], separate_node.outputs[2])

        # TODO: set up color ramp...
        color_ramp_node = material.node_tree.nodes.new('ShaderNodeValToRGB')
        color_ramp_elements = color_ramp_node.color_ramp.elements
        # TODO: remove initial elements
        blue_element = color_ramp_elements.new(2); blue_element.color = (0,0,1,1); blue_element.position = 0.000

        color_ramp_elements.remove(color_ramp_elements[0])
        color_ramp_elements.remove(color_ramp_elements[0])

        light_blue_element = color_ramp_elements.new(0); light_blue_element.color = (0.007, 0.247, 0.625, 1); light_blue_element.position = 0.008
        light_blue_element = color_ramp_elements.new(0); light_blue_element.color = (0.007, 0.247, 0.625, 1); light_blue_element.position = 0.014
        beach_yellow_element = color_ramp_elements.new(0); beach_yellow_element.color = (0.875, 1.0, 0.539, 1); beach_yellow_element.position = 0.04
        grass_green_element = color_ramp_elements.new(0); grass_green_element.color = (0.008, 0.381, 0.015, 1); grass_green_element.position = 0.452
        grass_green_element = color_ramp_elements.new(0); grass_green_element.color = (0.008, 0.381, 0.015, 1); grass_green_element.position = 0.662
        brown_hills_element = color_ramp_elements.new(0); brown_hills_element.color = (0.094, 0.041, 0.0, 1); brown_hills_element.position = 0.770


        material.node_tree.links.new(color_ramp_node.inputs[0], divide_node.outputs[0])

        bsdf_node = material.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        material.node_tree.links.new(bsdf_node.inputs[0], color_ramp_node.outputs[0])

        material_output = material.node_tree.nodes.get('Material Output')  # Should be there implicitly
        material.node_tree.links.new(material_output.inputs[0], bsdf_node.outputs[0])

        obj.data.materials.append(material)

    def create_terrain_object(self, context, spectrogram):
        mesh = bpy.context.object.data
        bm = bmesh.from_edit_mesh(mesh)

        vertices = []
        for wavelength in range(len(spectrogram)):
            row_vertices = []
            for time_step in range(len(spectrogram[wavelength])):
                amplitude = spectrogram[wavelength][time_step]
                logscale_wavelength = (np.log(wavelength) * 3) if wavelength > 0 else 0
                # TODO: parameterize if user want logscale, and parameterize the multiplier
                vertex = [logscale_wavelength, time_step, amplitude / 80]

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


def add_terrain_mesh_button(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(MusicTo3D.bl_idname, text="Sound Terrain", icon="MOD_SUBSURF")  # TODO: add custom icon


def register():
    bpy.utils.register_class(MusicTo3D)
    bpy.types.INFO_MT_mesh_add.append(add_terrain_mesh_button)


def unregister():
    bpy.utils.unregister_class(MusicTo3D)


if __name__ == '__main__':
    register()
