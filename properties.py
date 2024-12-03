import bpy

def mesh_poll_1(self, obj):
    """Filter mesh_1 to ensure it is not the same as mesh_2."""
    scene = bpy.context.scene
    return obj.type == 'MESH' and obj != scene.mesh_2

def mesh_poll_2(self, obj):
    """Filter mesh_2 to ensure it is not the same as mesh_1."""
    scene = bpy.context.scene
    return obj.type == 'MESH' and obj != scene.mesh_1

def get_vertex_groups(self, context):
    """Return a list of vertex group names for the selected mesh."""
    mesh = context.scene.mesh_1 if "1" in self.name else context.scene.mesh_2
    ret = []
    if mesh and mesh.type == 'MESH':
        ret = [(vg.name, vg.name, vg.name) for vg in mesh.vertex_groups]
    if len(ret) == 0:
        ret = [("Create vertex group", "Create vertex group", "Create vertex group")]
    return ret

def load_model(self, context):
    print("Model loaded")

def register():
    bpy.types.Scene.mesh_1 = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Mesh 1",
        description="First mesh selection",
        poll=mesh_poll_1
    )
    bpy.types.Scene.mesh_2 = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Mesh 2",
        description="Second mesh selection",
        poll=mesh_poll_2
    )
    
    bpy.types.Scene.vertex_group_1 = bpy.props.EnumProperty(
        name="",
        description="First vertex group",
        items=get_vertex_groups
    )
    
    bpy.types.Scene.vertex_group_2 = bpy.props.EnumProperty(
        name="",
        description="Second vertex group",
        items=get_vertex_groups
    )

    # New EnumProperties
    bpy.types.Scene.processor_type = bpy.props.EnumProperty(
        name="Processor",
        description="Select the processor type",
        items=[('CPU', "CPU", "Use the CPU computation"),
               ('GPU', "GPU", "Use the GPU computation")],
        default='CPU',
        update=load_model
    )
    
    bpy.types.Scene.direction = bpy.props.EnumProperty(
    name="Direction",
    description="Style transfer direction",
    items=[
        ('L2R', " ", "Transfer detail to second mesh.", 'FORWARD', 1),
        ('R2L', " ", "Exchange mesh detail.", 'ARROW_LEFTRIGHT', 2),
        ('LR', " ", "Transfer detail to first mesh.", 'BACK', 3)
    ],
    default='LR'
    )
    
    bpy.types.Scene.folder_path = bpy.props.StringProperty(
        name="Model",
        subtype='DIR_PATH',
        description="Path to model folder",
        default="Load a model",
        update=load_model
    )

    bpy.types.Scene.level = bpy.props.EnumProperty(
        name="Level",
        items=[("1", "1500", ""),
               ("2", "900", ""),
               ("3", "500", ""),
               ("4", "350", "")],
        description="Number of edges to which meshes will be pooled",
        default="4",
    )

def unregister():
    del bpy.types.Scene.mesh_1
    del bpy.types.Scene.mesh_2
    del bpy.types.Scene.vertex_group_1
    del bpy.types.Scene.vertex_group_2
    del bpy.types.Scene.processor_type
    del bpy.types.Scene.direction
    del bpy.types.Scene.folder_path
    del bpy.types.Scene.level
