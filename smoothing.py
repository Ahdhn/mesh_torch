import torch
import os
import igl
import scipy as sp
import numpy as np
import sys
import time

from torchviz import make_dot

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(f"Working on {device} device")

use_dense = False
benchmark = False

def create_sparse_adjacency_matrix(faces, n_vertices):
    """
    Create the Laplacian graph i.e., the vertex adjacency graph as a sparse 
    CSR tensor 
    """
    per_face_half_edges = torch.stack(
        [faces[:, (1, 2)], faces[:, (2, 0)], faces[:, (0, 1)]], -2
    )
    half_edges = torch.flatten(per_face_half_edges, -3, -2)

    idx = half_edges
    values = torch.ones(half_edges.shape[0], device=device)
    diag_idx = idx[:, 0]

    idx = torch.cat((idx.mT, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-values, values))

    adj_matrix = torch.sparse_coo_tensor(
        idx, values, (n_vertices, n_vertices)).coalesce().to_sparse_csr()
    adj_matrix.requires_grad_(True)
    return adj_matrix


def sparse_laplacian_smoothing_energy(vert, adj_matrix):
    """
    Compute the Laplacian energy, i.e., the sum of squared edge length using 
    the sparse graph Laplacian. 
    """
    res = adj_matrix @ vert
    energy = torch.sum(res ** 2)
    return energy


def create_dense_adjacency_matrix(faces, n_vertices):
    """
    Create the Laplacian graph i.e., the vertex adjacency graph as a dense tensor 
    """
    adj_matrix = torch.zeros((n_vertices, n_vertices), device=device)
    for f in faces:
        for i in range(3):
            adj_matrix[f[i], f[(i + 1) % 3]] = -1
            adj_matrix[f[i], f[i]] += 1
    return adj_matrix


def dense_laplacian_smoothing_energy(vertices, adj_matrix):
    """
    Compute the Laplacian energy, i.e., the sum of squared edge length using 
    the dense graph Laplacian. 
    """
    res = torch.matmul(adj_matrix, vertices)
    energy = torch.sum(res ** 2)
    return energy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing.py <path_to_obj_file>")
    else:
        obj_file = sys.argv[1]
        V, F = igl.read_triangle_mesh(obj_file)

        if not benchmark:
            plot(V, F, filename="mesh.html", shading={"wireframe": True})

        vert = torch.tensor(V, dtype=torch.float32,
                            requires_grad=True, device=device)
        faces = torch.tensor(F, device=device)

        adj_matrix = create_dense_adjacency_matrix(faces, len(
            V)) if use_dense else create_sparse_adjacency_matrix(faces, len(V))

        learning_rate = 0.005
        num_iterations = 100

        start_time = time.time()

        for iter in range(num_iterations):
            energy = dense_laplacian_smoothing_energy(
                vert, adj_matrix) if use_dense else sparse_laplacian_smoothing_energy(vert, adj_matrix)
            
            if not benchmark:
                if iter == 0:
                    dot = make_dot(energy, params={"vert": vert, "adj_matrix": adj_matrix})
                    dot.save("comp_graph.dot")
            
            energy.backward()

            with torch.no_grad():
                vert -= learning_rate * vert.grad
                vert.grad.zero_()

            if not benchmark:
                if iter % 10 == 0:
                    V = vert.detach().cpu().numpy()
                    plot(V, F, filename="mesh.html",
                         shading={"wireframe": True})
                    print(f"Iteration {iter}: Energy = {energy.item()}")

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(
            f"Smoothing PyTorch: {elapsed_time_ms:.3f} ms, {elapsed_time_ms/num_iterations:.3f} ms per iteration")

        # igl.write_triangle_mesh(os.path.join(os.getcwd() + "\out", "iter_" + str(iter) + "_.obj"), V, F)
