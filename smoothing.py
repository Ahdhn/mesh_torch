import torch
import os
import igl
import scipy as sp
import numpy as np

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()

root_folder = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(f"Working on {device} device")

use_dense = False


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


V, F = igl.read_triangle_mesh(os.path.join(root_folder, "bunnyhead.obj"))
plot(V, F, filename="mesh.html", shading={"wireframe": True})

vert = torch.tensor(V, dtype=torch.float32, requires_grad=True, device=device)
faces = torch.tensor(F, device=device)

adj_matrix = create_dense_adjacency_matrix(faces, len(
    V)) if use_dense else create_sparse_adjacency_matrix(faces, len(V))

print(adj_matrix)

learning_rate = 0.005
num_iterations = 100

for iter in range(num_iterations):
    energy = dense_laplacian_smoothing_energy(
        vert, adj_matrix) if use_dense else sparse_laplacian_smoothing_energy(vert, adj_matrix)
    energy.backward()
    
    with torch.no_grad():
        vert -= learning_rate * vert.grad
        vert.grad.zero_()

    if iter % 10 == 0:
        V = vert.detach().cpu().numpy()
        plot(V, F, filename="mesh.html", shading={"wireframe": True})
        print(f"Iteration {iter}: Energy = {energy.item()}")


# igl.write_triangle_mesh(os.path.join(root_folder + "\out", "iter_" + str(iter) + "_.obj"), V, F)
