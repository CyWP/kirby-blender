import bpy

class MESH_OT_Transfer(bpy.types.Operator):
    bl_idname = "mesh.transfer"
    bl_label = "Transfer"
    bl_description = "An example operator for mesh processing"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.report({'INFO'}, "Operation executed!")
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        # Disable the operator if any required property is missing
        scene = context.scene
        return bool(scene.mesh_1 and scene.mesh_2 and "Create" not in scene.vertex_group_1 and "Create" not in scene.vertex_group_2 and scene.folder_path and "Load" not in scene.folder_path)

def register():
    bpy.utils.register_class(MESH_OT_Transfer)

def unregister():
    bpy.utils.unregister_class(MESH_OT_Transfer)
