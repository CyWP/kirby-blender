import numpy as np

world_x = np.array([1, 0, 0], dtype=np.float32)
world_y = np.array([0, 1, 0], dtype=np.float32)
world_z = np.array([0, 0, 1], dtype=np.float32)
world_identity = np.array([world_x, world_y, world_z])

class MeshReconstruction():

    def __init__(self, mesh):
        self.mesh = mesh
        self.pool_layers = []
        self.active_layer = -1

    def new_layer(self):
        self.pool_layers.append([])
        self.active_layer += 1

    def add_pool(self, disp, src_idx, new_idx, normal):
        z = normal
        if np.sum(z-world_z)<10e-3:
            trans = world_identity
        else:
            x = closest_orthogonal(z, world_x)
            y = positive_orthogonal(x, z, world_y)
            basis = np.array([x, y, z])
            trans = np.linalg.inv(basis)
        trans_disp = trans @ disp
        self.pool_layers[self.active_layer].append(EdgeRestore(src_idx, new_idx, trans, trans_disp))

class EdgeRestore():

    def __init__(self, src_idx, new_idx, inv_basis, disp):
        self.src_idx = src_idx
        self.new_idx = new_idx
        self.inv_basis = inv_basis
        self.disp = disp

    def apply(self, mesh, vtx_swap:int=None):
        src_idx = self.src_idx if vtx_swap is None else vtx_swap
        z = mesh.vertex_normals[src_idx]
        if np.sum(z-world_z)<10e-3:
            trans = world_identity
        else:
            x = closest_orthogonal(z, world_x)
            y = positive_orthogonal(x, z, world_y)
            basis = np.array([x, y, z])
        new_basis = basis_transformation(self.inv_basis, basis)
        disp = new_basis @ self.disp

        oldv = mesh.vs[src_idx]
        v1 = oldv + disp
        v2 = oldv - disp

        V = [np.array(vec) for vec in mesh.vs.tolist()]
        F = mesh.faces.tolist()
        V.append(v2)
        v2_idx = len(V)
        #Figure out how to connect these new points
        for f_idx in mesh.vf[src_idx]:
            f = F[f_idx]
            idx = f.index(src_idx)
            other_idx = [i for i in f if i!=idx]
            other_1 = V[other_idx[1]]
            

def closest_orthogonal(a, b):
    # Ensure the vectors are numpy arrays
    v1 = np.array(a, dtype=float)
    v2 = np.array(b, dtype=float)
    # Compute the projection of v2 onto v1
    projection = np.dot(b, a) / np.dot(a, a) * a
    # Find the orthogonal component
    v_perp = b - projection 
    return v_perp

def positive_orthogonal(a, b, c):
    # Compute the cross product of a and b
    orthogonal_vector = np.cross(a, b)
    # Check if the dot product with c is negative
    if np.dot(orthogonal_vector, c) < 0:
        # Flip the vector to ensure a positive dot product
        orthogonal_vector = -orthogonal_vector   
    return orthogonal_vector

def basis_transformation(inv_S, T):
    return np.dot(T, inv_s)