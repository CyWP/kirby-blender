import bpy
import bmesh
import numpy as np

def np2mesh(context, V:np.ndarray, F:np.ndarray, name:str):

    if not isinstance(V, np.ndarray) or not isinstance(F, np.ndarray):
        raise ValueError("V and F must be NumPy arrays.")
    if V.shape[1] != 3:
        raise ValueError("V must have shape (n, 3).")
    if F.shape[1] not in (3, 4):
        raise ValueError("F must have shape (m, 3) or (m, 4).")
    
    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    # Link object to the active collection
    collection = context.collection
    collection.objects.link(obj)
    # Create the mesh from the vertex and face data
    mesh.from_pydata(V.tolist(), [], F.tolist()) 
    # Update mesh with calculated normals and topology
    mesh.update()
    
    return obj


def mesh2np(mesh:bpy.types.Object):

    if not isinstance(mesh, bpy.types.Object):
        raise TypeError("Input must be a Blender mesh object.")

    data = mesh.data.copy()
    temp_mesh = bmesh.new()
    temp_mesh.from_mesh(data)
    bmesh.ops.triangulate(temp_mesh, faces=temp_mesh.faces[:])

    # Extract vertices
    V = np.array([v.co for v in temp_mesh.verts], dtype=np.float32)
    # Extract triangulated faces
    F = np.array([[vert.index for vert in face.verts] for face in temp_mesh.faces], dtype=np.int32)
    # Extract edges
    E = np.array([(edge.verts[0].index, edge.verts[1].index) for edge in temp_mesh.edges], dtype=np.int32)

    return V, E, F


def vg2np(mesh:bpy.types.Object, name:str):

    if not mesh or mesh.type != 'MESH':
        raise ValueError("The provided object is not a valid mesh.")
    
    vertex_group = mesh.vertex_groups.get(name)
    if not vertex_group:
        raise ValueError(f"Vertex group '{name}' not found on the object.")
    
    group_index = vertex_group.index
    indices = [v.index for v in mesh.data.vertices if any(g.group == group_index for g in v.groups)]
    
    return np.array(indices, dtype=np.int32)
