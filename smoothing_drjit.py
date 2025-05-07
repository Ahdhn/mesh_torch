import drjit as dr
from drjit.cuda import Array1i
from drjit.cuda.ad import Array3f, Array2i

import igl
import numpy as np
import time
import sys
import os
import json


def laplacian_smoothing_energy(edges0, edges1, vert):
    v0 = dr.gather(Array3f, vert, edges0)
    v1 = dr.gather(Array3f, vert, edges1)
    edge_lengths = dr.squared_norm(v1 - v0)
    return dr.sum(edge_lengths)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing_drjit.py <path_to_obj_file>")
        sys.exit(1)

    obj_file = sys.argv[1]
    V, F = igl.read_triangle_mesh(obj_file)

    V = Array3f(np.array(V.T, dtype=np.float32))

    # Extract unique edges
    edges = np.vstack([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    edges0 = dr.cuda.UInt(edges[:, 0])
    edges1 = dr.cuda.UInt(edges[:, 1])

    learning_rate = 0.01
    num_iterations = 100

    start_time = time.time()
    # with dr.scoped_set_flag(dr.JitFlag.KernelHistory):

    for i in range(num_iterations):
        dr.enable_grad(V)
        energy = laplacian_smoothing_energy(edges0, edges1, V)
        dr.backward(energy)
        grad = dr.grad(V)
        # print(f"Iteration {i}: Energy = {energy}")

        V = dr.detach(V) - learning_rate * grad
        V = type(V)(V)
        dr.eval(V)

    # hist = dr.kernel_history()
    # print(hist)

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    entry = {
        "num_faces": int(F.shape[0]),
        "total_time_ms": round(elapsed_time_ms, 3),
        "num_iter": num_iterations
    }

    print(f'"{os.path.basename(obj_file)}": {json.dumps(entry, indent=2)}')
    print(",")
