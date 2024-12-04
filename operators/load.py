import bpy
from os.path import isfile, abspath

from ..utils import ModelManager

class MESH_OT_Load_Model(bpy.types.Operator):
    bl_idname = "model.load"
    bl_label = "Load"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return isfile(abspath(scene.model_path)) and scene.model_path[-4:]==".pth"

    def execute(self, context):
        scene = context.scene
        try:
            ModelManager.load_model(abspath(scene.model_path), scene.device)
        except Exception as e:
            raise e
            self.report({'ERROR'}, f"Failed to load model {context.scene.model_path}.\n{str(e)}")
            return {'FINISHED'}
        self.report({'INFO'}, f"Loaded model {context.scene.model_path}")
        return {'FINISHED'}

class MESH_OT_Release_Model(bpy.types.Operator):
    bl_idname = "model.release"
    bl_label = "Release"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return ModelManager.is_loaded()

    def execute(self, context):
        try:
            ModelManager.release_model()
        except:
            self.report({'ERROR'}, f"Failed to release model.\n{str(e)}")
            return {'FINISHED'}
        self.report({'INFO'}, f"Released model.")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(MESH_OT_Load_Model)
    bpy.utils.register_class(MESH_OT_Release_Model)

def unregister():
    bpy.utils.unregister_class(MESH_OT_Load_Model)
    bpy.utils.unregister_class(MESH_OT_Release_Model)