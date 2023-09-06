import torch
import mano
from tkinter import *
from mano.utils import Mesh
import numpy as np
import threading
from psbody.mesh import MeshViewers, MeshViewer, Mesh
from psbody.mesh.colors import name_to_rgb
from scipy import io

model_path = './hand_model/MANO_RIGHT.pkl'
n_comps = 45

batch_size = 1

rh_model = mano.load(model_path=model_path,
                     is_right= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=True)

if __name__ == "__main__":

    mvs = MeshViewers(window_width=800, window_height=800, shape=[1, 1])

    betas = torch.tensor([[0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0]], dtype=torch.float32)
    data_pose = []
    for i in range(45):
        data_pose.append(0)
    pose = torch.tensor(np.tile(data_pose, [batch_size, 1]))* np.pi / 180





    transl = torch.rand(batch_size, 3)
    output = rh_model(betas=betas,
                      hand_pose=pose,
                      transl=transl,
                      return_verts=True,
                      return_tips=True,
                      flat_hand_mean = True)
    mvs[0][0].set_dynamic_meshes([Mesh(v=output.vertices.cpu().detach().numpy().squeeze(), f=rh_model.faces, vc=name_to_rgb['pink'])],blocking=True)

    v = rh_model.hand_meshes(output)[0].vertices.view(np.ndarray) * 1000
    f = rh_model.hand_meshes(output)[0].faces.view(np.ndarray)
    io.savemat('./Output/mano_vert.mat', {'v': v})
    io.savemat('./Output/mano_face.mat', {'f': f})
    print('The mesh has been successfully saved')




