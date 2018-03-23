import bpy
import librosa

bl_info = {
    'name': 'Music to 3D',
    'category': 'Mesh'
}


class MusicTo3D(bpy.types.Operator):
    """
    Add docs
    """

    bl_idname = 'mesh.music_to_3d' # TODO: constant...
    bl_label = 'Music To 3D'
    bl_options = {'REGISTER', 'UNDO'}

    song_path = bpy.props.StringProperty(name="Song path", description="Path to the song specifying the terrain")

    def execute(self, context):
        print("The M3D operator was executed")


def add_terrain_mesh(self, context):
    layout = self.layout
    layout.separator()
    layout.operator("mesh.music_to_3d", text="Sound Terrain", icon="MOD_SUBSURF")


def register():
    bpy.utils.register_class(MusicTo3D)
    bpy.types.INFO_MT_mesh_add.append(add_terrain_mesh)


def unregister():
    bpy.utils.unregister_class(MusicTo3D)


if __name__ == '__main__':
    register()
