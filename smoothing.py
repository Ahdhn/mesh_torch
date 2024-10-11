import torch
import os
import igl
import scipy as sp
import numpy as np

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()


def laplacian_smoothing_energy(vertices, neighbors):
    energy = 0.0
    for i, neighbor_indices in enumerate(neighbors):
        neighbor_positions = vertices[neighbor_indices]
        average_position = neighbor_positions.mean(dim=0)
        energy += torch.sum((vertices[i] - average_position) ** 2)
    return energy


def create_vertex_adj(V, F):
    vertex_adj = {i: set() for i in range(len(V))}
    for face in F:
        for i in range(3):
            for j in range(3):
                if i != j:
                    vertex_adj[face[i]].add(face[j])
    vertex_adj_list = {k: list(v) for k, v in vertex_adj.items()}
    return vertex_adj_list


root_folder = os.getcwd()

V, F = igl.read_triangle_mesh(os.path.join(root_folder, "sphere1.obj"))
plot(V, F, filename="mesh.html")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Working on {device} device")

vert = torch.tensor(V,  requires_grad=True, device=device)
vertex_adj = create_vertex_adj(V, F)
print(vertex_adj)

learning_rate = 0.01
num_iterations = 100

for iter in range(num_iterations):
    # loss function
    energy = laplacian_smoothing_energy(vert, vertex_adj)
    # compute gradients
    energy.backward()

    with torch.no_grad():
        # take a step
        vert -= learning_rate * vert.grad
        # rest the gradients
        vert.grad.zero_()

    if iter % 10 == 0:
        V = vert.detach().cpu().numpy()
        plot(V, F, filename="mesh.html")
        print(f"Iteration {iter}: Energy = {energy.item()}")

igl.write_triangle_mesh(os.path.join(root_folder, "out.obj"), V, F)
