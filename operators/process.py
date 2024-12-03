import bpy

class MESH_OT_Process_1(bpy.types.Operator):
    bl_idname = "mesh.process1"
    bl_label = "Process"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.mesh_1 is not None

    def execute(self, context):
        # Placeholder processing code
        self.report({'INFO'}, f"Processing mesh_1")
        return {'FINISHED'}

class MESH_OT_Process_2(bpy.types.Operator):
    bl_idname = "mesh.process2"
    bl_label = "Process"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.mesh_2 is not None

    def execute(self, context):
        # Placeholder processing code
        self.report({'INFO'}, f"Processing mesh_2")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(MESH_OT_Process_1)
    bpy.utils.register_class(MESH_OT_Process_2)

def unregister():
    bpy.utils.unregister_class(MESH_OT_Process_1)   
    bpy.utils.unregister_class(MESH_OT_Process_2) 