import drjit as dr
from drjit.cuda.ad import Array3f, Array2i, Array3i, TensorXf, Float

import igl
import numpy as np
import time
import sys
import os
import json

is_area = True

def _gather_vec3_from_tensor(V: TensorXf, idx):
    # V has shape (nV,3) but storage is linear V.array of length nV*3
    base = idx * 3
    x = dr.gather(Float, V.array, base + 0)
    y = dr.gather(Float, V.array, base + 1)
    z = dr.gather(Float, V.array, base + 2)
    return Array3f(x, y, z)

def triangle_area_energy(f0, f1, f2, vert, eps=1e-20):    
    x0 = _gather_vec3_from_tensor(vert, f0)
    x1 = _gather_vec3_from_tensor(vert, f1)
    x2 = _gather_vec3_from_tensor(vert, f2)

    e01 = x1 - x0
    e02 = x2 - x0
    n = dr.cross(e01, e02)                     
    nn = dr.squared_norm(n)                    
    area = 0.5 * dr.sqrt(dr.maximum(nn, eps))
    return dr.sum(area)

def laplacian_smoothing_energy(edges0, edges1, vert):
    v0 = vert[edges0, :]          
    v1 = vert[edges1, :]
    d = v1 - v0
    edge_len2 = dr.sum(d * d, axis=1)
    return dr.sum(edge_len2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing_drjit.py <path_to_obj_file>")
        #sys.exit(1)
        obj_file = "sphere1.obj"
    else:
        obj_file = sys.argv[1]

    V_np, F = igl.read_triangle_mesh(obj_file)
        
    V_np = np.asarray(V_np, dtype=np.float32, order='C')   
    V = TensorXf(V_np.ravel(), shape=V_np.shape) 
    
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
    
    F_np = np.asarray(F, dtype=np.uint32, order='C')
    f0 = dr.cuda.UInt(F_np[:, 0])
    f1 = dr.cuda.UInt(F_np[:, 1])
    f2 = dr.cuda.UInt(F_np[:, 2])

    learning_rate = 0.02
    num_iterations = 100

    start_time = time.time()
    
    for i in range(num_iterations):
        dr.enable_grad(V)
        dr.clear_grad(V)
        if is_area:
            energy = triangle_area_energy(f0, f1, f2, V)
        else:
            energy = laplacian_smoothing_energy(edges0, edges1, V)
        dr.backward(energy)
        grad = dr.grad(V)
        #print(f"Iteration {i}: Energy = {energy}")

        V = dr.detach(V) - learning_rate * grad
        V = type(V)(V)
        dr.eval(V)      
        dr.sync_thread()    

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    entry = {
        "num_faces": int(F.shape[0]),
        "total_time_ms": round(elapsed_time_ms, 3),
        "num_iter": num_iterations
    }

    print(f'"{os.path.basename(obj_file)}": {json.dumps(entry, indent=2)}')
    print(",")
    
    # igl.write_triangle_mesh("out.obj", V.numpy(), F)
