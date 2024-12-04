import bpy
from .utils import ModelManager

def mesh_poll_1(self, obj):
    """Filter mesh_1 to ensure it is not the same as mesh_2."""
    scene = bpy.context.scene
    return obj.type == 'MESH' and obj != scene.mesh_2

def mesh_poll_2(self, obj):
    """Filter mesh_2 to ensure it is not the same as mesh_1."""
    scene = bpy.context.scene
    return obj.type == 'MESH' and obj != scene.mesh_1

def mesh_load_1(self, context):
    ModelManager.set_mesh1(context.scene.mesh_1)

def mesh_load_2(self, context):
    ModelManager.set_mesh2(context.scene.mesh_2)

def get_vertex_groups_1(self, context):
    """Return a list of vertex group names for the selected mesh."""
    mesh = context.scene.mesh_1
    ret = []
    if mesh and mesh.type == 'MESH':
        ret = [(vg.name, vg.name, "") for vg in mesh.vertex_groups]
    if len(ret) == 0:
        ret = [("Create vertex group", "Create vertex group", "")]
    return ret

def get_vertex_groups_2(self, context):
    """Return a list of vertex group names for the selected mesh."""
    mesh = context.scene.mesh_2
    ret = []
    if mesh and mesh.type == 'MESH':
        ret = [(vg.name, vg.name, "") for vg in mesh.vertex_groups]
    if len(ret) == 0:
        ret = [("Create vertex group", "Create vertex group", "Create vertex group")]
    return ret

def get_pool_resolutions(self, context):
    if ModelManager.is_loaded():
        return [(str(res), str(res), "") for res in ModelManager.get_pool_resolutions()]
    else:
        return [("", "", "")]

def load_model(self, context):
    print("Model loaded")

def change_device(self, contex):
    if ModelManager.is_loaded():
        ModelManager.set_device(context.scene.device)

def register():
    bpy.types.Scene.mesh_1 = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Mesh 1",
        description="First mesh selection",
        poll=mesh_poll_1,
        update=mesh_load_1
    )
    bpy.types.Scene.mesh_2 = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Mesh 2",
        description="Second mesh selection",
        poll=mesh_poll_2,
        update=mesh_load_2
    )
    
    bpy.types.Scene.vertex_group_1 = bpy.props.EnumProperty(
        name="",
        description="First vertex group",
        items=get_vertex_groups_1
    )
    
    bpy.types.Scene.vertex_group_2 = bpy.props.EnumProperty(
        name="",
        description="Second vertex group",
        items=get_vertex_groups_2
    )

    # New EnumProperties
    bpy.types.Scene.device = bpy.props.EnumProperty(
        name="Processor",
        description="Select the processor type",
        items=[(device, device.upper(), "") for device in ModelManager.get_available_devices()],
        default="cpu",
        update=change_device
    )
    
    bpy.types.Scene.direction = bpy.props.EnumProperty(
    name="Direction",
    description="Style transfer direction",
    items=[
        ('L2R', "L2R", "Transfer detail to second mesh.", 'FORWARD', 1),
        ('LR', "BOTH", "Transfer detail to first mesh.", 'ARROW_LEFTRIGHT', 2),
        ('R2L', "R2L", "Exchange mesh detail.", 'BACK', 3)
    ],
    default='LR'
    )
    
    bpy.types.Scene.model_path = bpy.props.StringProperty(
        name="Model",
        subtype='FILE_PATH',
        description="Select a .pth model file.",
        default="Select a model file (.pth)",
        update=load_model
    )

    bpy.types.Scene.level = bpy.props.EnumProperty(
        name="Level",
        items=get_pool_resolutions,
        description="Number of edges to which meshes will be pooled"
    )

def unregister():
    del bpy.types.Scene.mesh_1
    del bpy.types.Scene.mesh_2
    del bpy.types.Scene.vertex_group_1
    del bpy.types.Scene.vertex_group_2
    del bpy.types.Scene.device
    del bpy.types.Scene.direction
    del bpy.types.Scene.model_path
    del bpy.types.Scene.level
