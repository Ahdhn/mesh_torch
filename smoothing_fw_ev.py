import torch
import os
import igl
import scipy as sp
import numpy as np
import sys
import time

import torch.autograd.forward_ad as fwAD
from torch.func import jacfwd

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(f"Working on {device} device")

benchmark = True


def laplacian_smoothing_energy(edges, vert):
    v0 = vert[edges[:, 0]]
    v1 = vert[edges[:, 1]]

    edge_lengths = torch.norm(v1 - v0, dim=1)**2
    energy = torch.sum(edge_lengths)

    return energy


def energy_fn(vert):    
    return laplacian_smoothing_energy(edges, vert)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing_fw_ev.py <path_to_obj_file>")
    else:
        obj_file = sys.argv[1]
        V, F = igl.read_triangle_mesh(obj_file)

        if not benchmark:
            plot(V, F, filename="mesh.html", shading={"wireframe": True})

        vert = torch.tensor(V, dtype=torch.float32,
                            requires_grad=True, device=device)
        faces = torch.tensor(F, device=device)

        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ], dim=0)

        edges = torch.sort(edges, dim=1).values

        edges = torch.unique(edges, dim=0)

        learning_rate = 0.01
        num_iterations = 100

        start_time = time.time()

        for iter in range(num_iterations):

            grad = jacfwd(energy_fn)(vert)
            
            # grad = torch.zeros_like(vert)
            # with fwAD.dual_level():
            #     for i in range(vert.shape[0]):
            #         for j in range(vert.shape[1]):
            #             tangent = torch.zeros_like(vert)
            #             tangent[i, j] = 1.0
            #             dual_input = fwAD.make_dual(vert, tangent)
            #             energy = laplacian_smoothing_energy(edges, dual_input)
            #             grad[i, j] = fwAD.unpack_dual(energy).tangent

            with torch.no_grad():
                vert -= learning_rate * grad

            if not benchmark:
                if iter % 10 == 0:
                    V = vert.detach().cpu().numpy()
                    plot(V, F, filename="mesh.html",
                         shading={"wireframe": True})
                    print(
                        f"Iteration {iter}: Energy = {energy_fn(vert).item()}")

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(
            f"Smoothing PyTorch: {elapsed_time_ms:.3f} ms, {elapsed_time_ms/num_iterations:.3f} ms per iteration")
