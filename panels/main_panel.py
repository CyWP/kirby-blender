import bpy
from ..utils import ModelManager

class VIEW3D_PT_MainPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_main_panel"
    bl_label = "Kirby"
    bl_category = "Kirby"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw_header(self, context):
        scene = context.scene

        layout = self.layout
        layout.label(icon="MATSHADERBALL")

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        box = row.box()
        row = box.row()
        row.prop(scene, "model_path", text="")
        row = box.row()
        row.operator("model.load", text="Load")
        row.operator("model.release", text="Release")
        for line in ModelManager.get_model_info():
            row = box.row()
            row.label(text=line)
        if ModelManager.is_loaded():
            row = layout.row() 
            box = row.box()
            row = box.row()
            boxL = row.box()
            boxR = row.box()
            # Mesh Selection
            boxL.prop(scene, "mesh_1", text='')
            boxR.prop(scene, "mesh_2", text='')

            # Vertex Group Selection
            if scene.mesh_1 and scene.mesh_1.type == 'MESH':
                boxL.prop(scene, "vertex_group_1", icon="SNAP_VERTEX")
                boxL.operator("mesh.process1", text="Process")
            if scene.mesh_2 and scene.mesh_2.type == 'MESH':
                boxR.prop(scene, "vertex_group_2", icon="SNAP_VERTEX")
                boxR.operator("mesh.process2", text="Process")

            row = layout.row()
            box = row.box()
            if len(ModelManager.get_available_devices())>1:
                row = box.row()
                row.prop(scene, "device", expand=True)
            if ModelManager.is_loaded():
                row = box.row()
                row.prop(scene, "level", expand=True)
            # Text Direction Selection (Enforcing single-line)
            row = box.row()
            box = row.box()
            row = box.row()
            # Operator Button
            row.operator("mesh.transfer")
            row = box.row()
            row.prop(scene, "direction", expand=True)

def register():
    bpy.utils.register_class(VIEW3D_PT_MainPanel)

def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_MainPanel)
