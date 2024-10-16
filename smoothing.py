import torch
import os
import igl
import scipy as sp
import numpy as np

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(f"Working on {device} device")


def laplacian_smoothing_energy(vertices, adj_matrix):
    """
    Compute the Laplacian smoothing energy using the sparse adjacency matrix 
    For each vertex, compute the average position of its neighbors 
    """
    local_energies = torch.zeros(
        vertices.shape[0], device=vertices.device)

    for v in range(vertices.shape[0]):
        # get the indices of neighbors from the adjacency matrix (row i)
        start = adj_matrix.crow_indices()[v].item()
        end = adj_matrix.crow_indices()[v+1].item()
        neighbor_indices = adj_matrix.col_indices()[start:end]
        neighbor_positions = vertices[neighbor_indices]
        average_position = neighbor_positions.mean(dim=0)
        local_energies[v] = torch.sum((vertices[v] - average_position) ** 2)

    total_energy = local_energies.sum()
    return total_energy


def create_adjacency_matrix(V, F):
    """
    Create an adjacency matrix in a sparse format
    """
    num_vertices = len(V)
    row_id = []
    col_id = []
    for face in F:
        for i in range(3):
            for j in range(3):
                if i != j:
                    row_id.append(face[i])
                    col_id.append(face[j])

    # Create adjacency matrix indices as tensor
    indices = torch.tensor([row_id, col_id], dtype=torch.long)

    # Create a sparse adjacency matrix with values of 1 at each (i, j)
    coo_adj_matrix = torch.sparse_coo_tensor(indices, torch.ones(
        len(row_id)), (num_vertices, num_vertices), device=device)

    coo_adj_matrix = coo_adj_matrix.coalesce()

    csr_adj_matrix = coo_adj_matrix.to_sparse_csr()

    return csr_adj_matrix


root_folder = os.getcwd()

V, F = igl.read_triangle_mesh(os.path.join(root_folder, "bunnyhead.obj"))
plot(V, F, filename="mesh.html", shading={"wireframe": True})


vert = torch.tensor(V,  requires_grad=True, device=device)
adj_matrix = create_adjacency_matrix(V, F)

# print(adj_matrix)

learning_rate = 0.005
num_iterations = 100

for iter in range(num_iterations):
    # loss function
    energy = laplacian_smoothing_energy(vert, adj_matrix)
    # compute gradients
    energy.backward()
    print(f"energy = {energy}")

    with torch.no_grad():
        # take a step
        vert -= learning_rate * vert.grad
        # rest the gradients
        vert.grad.zero_()
    
    
    #if iter % 10 == 0:
    #    V = vert.detach().cpu().numpy()
    #    plot(V, F, filename="mesh.html", shading={"wireframe": True})
    #    igl.write_triangle_mesh(os.path.join(
    #        root_folder + "\out", "iter_" + str(iter) + "_.obj"), V, F)
    #    print(f"Iteration {iter}: Energy = {energy.item()}")
    
