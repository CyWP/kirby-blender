import numpy as np
import os
import ntpath
import pickle

def fill_mesh(mesh2fill:'Mesh', V:np.array, E:np.array, F:np.array):

    mesh_data = from_scratch(V, E, F)

    mesh2fill.vs = mesh_data["vs"]
    mesh2fill.edges = mesh_data["edges"]
    mesh2fill.faces = mesh_data["faces"]
    mesh2fill.gemm_edges = mesh_data["gemm_edges"]
    mesh2fill.edges_count = int(mesh_data["edges_count"])
    mesh2fill.ve = mesh_data["ve"]
    mesh2fill.v_mask = mesh_data["v_mask"]
    mesh2fill.filename = str(mesh_data["filename"])
    mesh2fill.edge_lengths = mesh_data["edge_lengths"]
    mesh2fill.edge_areas = mesh_data["edge_areas"]
    mesh2fill.features = mesh_data["features"]
    mesh2fill.sides = mesh_data["sides"]
    mesh2fill.vf = mesh_data["vf"]
    mesh2fill.faces_mask = mesh_data["faces_mask"]


class MeshPrep:
    def __getitem__(self, item):
        return eval("self." + item)


def from_scratch(V:np.array, E:np.array, F:np.array):

    mesh_data = MeshPrep()
    mesh_data.gemm_edges = mesh_data.edges = mesh_data.edges_count = mesh_data.sides = mesh_data.ve = mesh_data.edge_lengths = None
    mesh_data.vs = V
    mesh_data.filename = "unknown"
    mesh_data.edge_areas = []
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(mesh_data, F)
    mesh_data.faces = faces
    mesh_data.face_areas = face_areas
    mesh_data.vf = get_vf(faces, mesh_data.vs.shape[0])
    mesh_data.faces_mask = np.ones((faces.shape[0],), dtype=bool)
    build_gemm(mesh_data, mesh_data.faces, mesh_data.face_areas)
    mesh_data.features = extract_features(mesh_data)
    return mesh_data


def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]


def build_gemm(mesh, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
            # Easy to retriweve edges in face, just grouped by 3 adjacent idx
            # Also, this is a halfedge datastructure
        #Loop below builds ve, list of edges attached to a given vertex
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge))) 
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count  
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(
                    edges_count
                )  # ve seems to use a vertex index as key
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3

        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[
                faces_edges[(idx + 2) % 3]
            ]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = (
                nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            )
            sides[edge_key][nb_count[edge_key] - 1] = (
                nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
            )
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(
        face_areas
    )  # todo whats the difference between edge_areas and edge_lenghts?


def get_vf(faces, num_v):
    vf = [[] for _ in range(num_v)]
    for f_idx, face in enumerate(faces):
        for v in face:
            vf[v].append(f_idx)
    return vf


def compute_face_normals_and_areas(mesh, faces):
    try:
        face_normals = np.cross(
            mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]],
        )
        face_areas = np.sqrt((face_normals**2).sum(axis=1))
        face_normals /= face_areas[:, np.newaxis]
        assert not np.any(face_areas[:, np.newaxis] == 0), (
            "has zero area face: %s" % mesh.filename
        )
        face_areas *= 0.5
    except Exception as e:
        raise Exception(f"Problem with faces in mesh {mesh.filename}: {[f for f in faces]}.\n Exception: {str(e)}")
    return face_normals, face_areas


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = (
            mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        )
        edge_b = (
            mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        )
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face


def check_area(mesh, faces):
    face_normals = np.cross(
        mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
        mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]],
    )
    face_areas = np.sqrt((face_normals**2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(
        mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1
    )
    mesh.edge_lengths = edge_lengths


def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide="raise"):
        try:
            for extractor in [
                dihedral_angle,
                symmetric_opposite_angles,
                symmetric_ratios
            ]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, "bad features")


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """computes two angles: one for each face shared between the edge
    the angle is in each face opposite the edge
    sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate(
        (np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0
    )
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """computes two ratios: one for each face shared between the edge
    the ratio is between the height / base (edge) of each triangle
    sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate(
        (np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0
    )
    return np.sort(ratios, axis=0)


def dot_skew(mesh, edge_points):
    """computes the dot product between the edge vector and the vector
    of opposite vertices, normalized by the product of their norms.
    Essentially noting if opposing faces skew in the same or opposing sides.
    """
    edge_vec = mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]]
    diag_vec = mesh.vs[edge_points[:, 2]] - mesh.vs[edge_points[:, 3]]
    max_dot = np.linalg.norm(edge_vec, axis=1) * np.linalg.norm(diag_vec, axis=1)
    dot = np.abs(np.sum(edge_vec * diag_vec, axis=1)).reshape((1, -1))
    return dot / max_dot


def get_edge_points(mesh):
    """returns: edge_points (#E x 4) tensor, with four vertex ids per edge
    for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id
    each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [
        edge_a[first_vertex],
        edge_a[1 - first_vertex],
        edge_b[second_vertex],
        edge_d[third_vertex],
    ]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals


def get_opposite_angles(mesh, edge_points, side):
    edges_a = (
        mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    )
    edges_b = (
        mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    )

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[
        :, np.newaxis
    ]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[
        :, np.newaxis
    ]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(
        mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
        ord=2,
        axis=1,
    )
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1
    )
    closest_point = (
        point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    )
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
