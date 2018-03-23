import random

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

    song_path = bpy.props.StringProperty(name="Song path", description="Path to the song specifying the terrain",
                                         subtype='FILE_PATH')

    def execute(self, context):
        spectrogram = self.get_spectrogram()
        mesh = self.create_new_mesh()
        obj = self.create_new_object(mesh)
        self.add_object_to_scene(context, obj)
        self.create_terrain_object(context, spectrogram)

        return {'FINISHED'}

    def get_spectrogram(self):
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
        return bpy.data.objects.new("M3D Object", mesh)

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
