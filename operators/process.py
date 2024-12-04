import bpy
from ..utils import ModelManager

class MESH_OT_Process_1(bpy.types.Operator):
    bl_idname = "mesh.process1"
    bl_label = "Process"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.mesh_1 and ModelManager.is_loaded()

    def execute(self, context):
        ModelManager.set_mesh1(context.scene.mesh_1)
        ModelManager.process1(context)
        self.report({'INFO'}, f"Processing mesh_1")
        return {'FINISHED'}

class MESH_OT_Process_2(bpy.types.Operator):
    bl_idname = "mesh.process2"
    bl_label = "Process"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.mesh_2 and ModelManager.is_loaded()

    def execute(self, context):
        ModelManager.set_mesh2(context.scene.mesh_2)
        ModelManager.process2(context)
        self.report({'INFO'}, f"Processing mesh_2")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(MESH_OT_Process_1)
    bpy.utils.register_class(MESH_OT_Process_2)

def unregister():
    bpy.utils.unregister_class(MESH_OT_Process_1)   
    bpy.utils.unregister_class(MESH_OT_Process_2) 