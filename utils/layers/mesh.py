from tempfile import mkstemp
from shutil import move
from scipy.sparse import lil_matrix, csr_matrix
import torch
import numpy as np
import os

from .mesh_union import MeshUnion
from .mesh_prepare import fill_mesh
from .mesh_reconstruction import MeshReconstruction


class Mesh:

    def __init__(self, V:np.array, E:np.array, F:np.array):

        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = E
        self.gemm_edges = self.sides = None
        self.pool_count = 0
        fill_mesh(self, V, E, F)
        self.history_data = None
        self.reconstruction = MeshReconstruction(self)
        self.init_history()

    def extract_features(self):
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        iv_a = edge[0]
        iv_b = edge[1]
        v_a = self.vs[iv_a]
        v_b = self.vs[iv_b]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[iv_b] = False  # masking point since it no longer is used
        mask = self.edges == iv_b  # Gets all indices at which point was used
        self.ve[edge[0]].extend(self.ve[iv_b])
        self.edges[mask] = iv_a  # Essentially replacing all connection of since delted point with the new averaged point.
        #Update face representations
        for f_idx in self.vf[iv_b]:
            f = self.faces[f_idx]
            if iv_a in f:
                self.faces_mask[f_idx] = False
                self.vf[iv_a].remove(f_idx)
                self.faces[f_idx, :] = np.array([-1, -1, -1])
            else:
                self.faces[f_idx, np.where(f==iv_b)] = iv_a
                self.vf[iv_a].append(f_idx)

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        #only edits self.ve
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1


    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
            "groups": [],
            "gemm_edges": [self.gemm_edges.copy()],
            "occurrences": [],
            "old2current": np.arange(self.edges_count, dtype=np.int32),
            "current2old": np.arange(self.edges_count, dtype=np.int32),
            "edges_mask": [torch.ones(self.edges_count, dtype=torch.bool)],
            "edges_count": [self.edges_count],
        }


    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data["edges_mask"][-1][
                self.history_data["current2old"][index]
            ] = 0
            self.history_data["old2current"][
                self.history_data["current2old"][index]
            ] = -1

    def get_groups(self):
        return self.history_data["groups"].pop()

    
    def union_groups(self, source, target):
        return
        if self.history_data:
            self.history_data["collapses"].union(
                self.history_data["current2old"][source],
                self.history_data["current2old"][target],
            )
        return

    def get_occurrences(self):
        return self.history_data["occurrences"].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data["old2current"] != -1
            self.history_data["old2current"][mask] = np.arange(
                self.edges_count, dtype=np.int32
            )
            self.history_data["current2old"][0 : self.edges_count] = np.ma.where(mask)[
                0
            ]
            self.history_data["occurrences"].append(groups.get_occurrences())
            self.history_data["groups"].append(groups.get_groups(pool_mask))
            self.history_data["gemm_edges"].append(self.gemm_edges.copy())
            self.history_data["edges_count"].append(self.edges_count)

    def unroll_gemm(self):
        self.history_data["gemm_edges"].pop()
        self.gemm_edges = self.history_data["gemm_edges"][-1]
        self.history_data["edges_count"].pop()
        self.edges_count = self.history_data["edges_count"][-1]

    def get_edge_areas(self):
        return self.edge_areas

    def mean_curvature_energy(self):
        V = self.vs[self.v_mask, :]
        L = csr_matrix(self.compute_laplacian())
        return np.mean(np.abs(L @ V))

    def compute_laplacian(self):
        faces = self.faces[self.faces_mask]
        vertices = self.vs
        n_vertices = vertices.shape[0]
        # Initialize a sparse matrix for the Laplacian
        L = lil_matrix((n_vertices, n_vertices))
        # Loop over all triangles (faces)
        for face in faces:
            i, j, k = face
            # Vertices of the triangle
            vi, vj, vk = vertices[i], vertices[j], vertices[k]
            # Cotangent weights for the edges
            cot_jk = 0.5 * cotangent(vi, vj, vk)  # Cotangent at vertex vi (opposite edge vj-vk)
            cot_ik = 0.5 * cotangent(vj, vk, vi)  # Cotangent at vertex vj (opposite edge vi-vk)
            cot_ij = 0.5 * cotangent(vk, vi, vj)  # Cotangent at vertex vk (opposite edge vi-vj)
            # Update the Laplacian matrix using cotangent weights
            L[i, i] -= cot_ij + cot_ik
            L[j, j] -= cot_jk + cot_ij
            L[k, k] -= cot_jk + cot_ik
            L[i, j] += cot_ij
            L[j, i] += cot_ij
            L[i, k] += cot_ik
            L[k, i] += cot_ik
            L[j, k] += cot_jk
            L[k, j] += cot_jk
        valid_idx = np.where(self.v_mask)[0]
        return L[valid_idx, :][:, valid_idx]

def cotangent(v1, v2, v3):
    # Vectors along the triangle edges
    u = v2 - v1
    v = v3 - v1
    # Compute cotangent using dot product and cross product
    cos_angle = np.dot(u, v)
    sin_angle = np.linalg.norm(np.cross(u, v))
    return cos_angle / (sin_angle + 10e-6)