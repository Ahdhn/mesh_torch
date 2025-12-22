import torch
import os
import igl
import scipy as sp
import numpy as np
import sys
import time
import json

#import meshplot
#from meshplot import plot, subplot, interact

#meshplot.offline()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
# print(f"Working on {device} device")

benchmark = True
is_area = True

def laplacian_smoothing_energy_edges(edges, vert):
    v0 = vert[edges[:, 0]]
    v1 = vert[edges[:, 1]]

    edge_lengths = torch.norm(v1 - v0, dim=1)**2
    energy = torch.sum(edge_lengths)

    return energy

def laplacian_smoothing_energy_area(faces, vert):
    x0 = vert[faces[:, 0]]
    x1 = vert[faces[:, 1]]
    x2 = vert[faces[:, 2]]

    n = torch.cross(x1 - x0, x2 - x0, dim=1) 
    nn = torch.linalg.norm(n, dim=1)
    
    area = 0.5 * nn
    
    energy = area.sum()

    return energy

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing_ev.py <path_to_obj_file>")
    else:
        obj_file = sys.argv[1]
        V, F = igl.read_triangle_mesh(obj_file)

        # if not benchmark:
        #     plot(V, F, filename="mesh.html", shading={"wireframe": True})

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

        learning_rate = 0.02
        num_iterations = 100

        start_time = time.time()

        for iter in range(num_iterations):
            if is_area:
                energy = laplacian_smoothing_energy_area(faces, vert)
            else:
                energy = laplacian_smoothing_energy_edges(edges, vert)
            energy.backward()
            
            #print(f"Iteration {iter}: Energy = {energy.item()}")        

            with torch.no_grad():
                vert -= learning_rate * vert.grad
                vert.grad.zero_()

            if not benchmark:
                if iter % 10 == 0:
                    V = vert.detach().cpu().numpy()
                    # plot(V, F, filename="mesh.html",
                    #      shading={"wireframe": True})
                    print(f"Iteration {iter}: Energy = {energy.item()}")

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000

        entry = {
            "num_faces": int(faces.shape[0]),
            "total_time_ms": round(elapsed_time_ms, 3),
            "num_iter": int(num_iterations)
            }

        print(f'"{os.path.basename(obj_file)}": {json.dumps(entry, indent=2)}')    
        print(",")   

        # print(f"#Faces = {faces.shape[0]}")
        # print(
        #     f"Smoothing PyTorch: {elapsed_time_ms:.3f} ms, {elapsed_time_ms/num_iterations:.3f} ms per iteration")
        
        #V = vert.detach().cpu().numpy()        
        #igl.write_triangle_mesh("out.obj", V, F)
        
        #igl.write_triangle_mesh(os.path.join(os.getcwd() + "\out", "iter_" + str(iter) + "_.obj"), V, F)
