import random
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

        return mel_spectrogram

        width = 20
        height = 50
        data = []

        for x in range(width):
            row = []
            for y in range(height):
                row.append(random.uniform(0.0, 1.2))
            data.append(row)

        return data

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
        material = bpy.data.materials.new(name="Greeny material")
        material.diffuse_color = (0.2, 0.7, 0.5)
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
                row_vertices.append(bm.verts.new((logscale_wavelength, time_step, amplitude / 80)))

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
