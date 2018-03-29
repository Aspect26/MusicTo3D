import random
import traceback

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

    def create_terrain_object(self, context, spectrogram):
        mesh = bpy.context.object.data
        bm = bmesh.from_edit_mesh(mesh)

        vertices = []
        for row in range(len(spectrogram)):
            row_vertices = []
            for column in range(len(spectrogram[row])):
                row_vertices.append(bm.verts.new((row, column, spectrogram[row][column])))

            vertices.append(row_vertices)

        for row in range(len(vertices) - 1):
            for column in range(len(vertices[row]) - 1):
                bm.faces.new((vertices[row][column], vertices[row][column + 1], vertices[row + 1][column]))
                bm.faces.new((vertices[row][column + 1], vertices[row + 1][column + 1], vertices[row + 1][column]))

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
