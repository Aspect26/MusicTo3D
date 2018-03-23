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
        vertices = self.get_terrain_vertices()
        mesh = self.create_new_mesh()
        obj = self.create_new_object(mesh)
        self.add_object_to_scene(context, obj)
        self.create_terrain_object(context, vertices)

        return {'FINISHED'}

    def get_terrain_vertices(self):
        return [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0)]

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

    def create_terrain_object(self, context, vertex_locations):
        mesh = bpy.context.object.data
        bm = bmesh.from_edit_mesh(mesh)

        vertices = []
        for vertes_location in vertex_locations:
            vertices.append(bm.verts.new(vertes_location))

        for vertex_index in range(len(vertices) - 2):
            bm.faces.new((vertices[vertex_index], vertices[vertex_index + 1], vertices[vertex_index + 2]))

        bmesh.update_edit_mesh(mesh)


def add_terrain_mesh_button(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(MusicTo3D.bl_idname, text="Sound Terrain", icon="MOD_SUBSURF")


def register():
    bpy.utils.register_class(MusicTo3D)
    bpy.types.INFO_MT_mesh_add.append(add_terrain_mesh_button)


def unregister():
    bpy.utils.unregister_class(MusicTo3D)


if __name__ == '__main__':
    register()
