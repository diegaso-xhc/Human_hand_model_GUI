import torch
import mano
from tkinter import *
from mano.utils import Mesh
import numpy as np
import threading
from psbody.mesh import MeshViewers, MeshViewer, Mesh
from psbody.mesh.colors import name_to_rgb
from scipy import io

model_path = './hand_model/MANO_LEFT.pkl'
n_comps = 45

root = Tk()
root.geometry("550x950")
root.title("Hand controllers")

# Variables definition
global value_beta_1
global value_beta_2
global value_beta_3
global value_beta_4
global value_beta_5
global value_beta_6
global value_beta_7
global value_beta_8
global value_beta_9
global value_beta_10
global pose_index_J1
global pose_index_J2
global pose_index_J3
global pose_middle_J1
global pose_middle_J2
global pose_middle_J3
global pose_pinky_J1
global pose_pinky_J2
global pose_pinky_J3
global pose_ring_J1
global pose_ring_J2
global pose_ring_J3
global pose_daumen_J1
global pose_daumen_J2
global pose_daumen_J3
global h_meshes
global batch_size
global global_frame_1
global global_frame_2
global global_frame_3

batch_size = 1

rh_model = mano.load(model_path=model_path,
                     is_right= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=True)

def hand_show(values=None):
    print("Motion in progress...")
    value_beta_1 = vertical1.get()
    value_beta_2 = vertical2.get()
    value_beta_3 = vertical3.get()
    value_beta_4 = vertical4.get()
    value_beta_5 = vertical5.get()
    value_beta_6 = vertical6.get()
    value_beta_7 = vertical7.get()
    value_beta_8 = vertical8.get()
    value_beta_9 = vertical9.get()
    value_beta_10 = vertical10.get()
    global_frame_1 = global_frame_1_scale.get()
    global_frame_2 = global_frame_2_scale.get()
    global_frame_3 = global_frame_3_scale.get()


    data_pose = [[int(pose_index_J1[0].get()),int(pose_index_J1[1].get()),int(pose_index_J1[2].get()), # Index J1
             int(pose_index_J2[0].get()),int(pose_index_J2[1].get()),int(pose_index_J2[2].get()),  # Index J2
             int(pose_index_J3[0].get()),int(pose_index_J3[1].get()),int(pose_index_J3[2].get()),  # Index J3
             int(pose_middle_J1[0].get()),int(pose_middle_J1[1].get()),int(pose_middle_J1[2].get()),  # Middle J1
             int(pose_middle_J2[0].get()),int(pose_middle_J2[1].get()),int(pose_middle_J2[2].get()),  # Middle J2
             int(pose_middle_J3[0].get()),int(pose_middle_J3[1].get()),int(pose_middle_J3[2].get()),  # Middle J3
             int(pose_pinky_J1[0].get()),int(pose_pinky_J1[1].get()),int(pose_pinky_J1[2].get()),  # Pinky J1
             int(pose_pinky_J2[0].get()),int(pose_pinky_J2[1].get()),int(pose_pinky_J2[2].get()),  # Pinky J2
             int(pose_pinky_J3[0].get()),int(pose_pinky_J3[1].get()),int(pose_pinky_J3[2].get()),  # Pinky J3
             int(pose_ring_J1[0].get()),int(pose_ring_J1[1].get()),int(pose_ring_J1[2].get()),  # Ring J1
             int(pose_ring_J2[0].get()),int(pose_ring_J2[1].get()),int(pose_ring_J2[2].get()),  # Ring J2
             int(pose_ring_J3[0].get()),int(pose_ring_J3[1].get()),int(pose_ring_J3[2].get()),  # Ring J3
             int(pose_daumen_J1[0].get()),int(pose_daumen_J1[1].get()),int(pose_daumen_J1[2].get()),  # Daumen J1
             int(pose_daumen_J2[0].get()),int(pose_daumen_J2[1].get()),int(pose_daumen_J2[2].get()),  # Daumen J2
             int(pose_daumen_J3[0].get()),int(pose_daumen_J3[1].get()),int(pose_daumen_J3[2].get())]]  # Daumen J3

    betas = torch.tensor([[value_beta_1, value_beta_2, value_beta_3, value_beta_4,
                           value_beta_5, value_beta_6, value_beta_7, value_beta_8,
                           value_beta_9, value_beta_10]], dtype=torch.float32)

    pose = torch.tensor(np.tile(data_pose, [batch_size, 1]))* np.pi / 180
    #global_orient = torch.zeros(batch_size, 3)
    global_orient = torch.tensor([[int(global_frame_1),int(global_frame_2),int(global_frame_3)]], dtype=torch.float32)
    global_orient = global_orient*np.pi/180
    #global_orient = torch.tensor([[0.,0.,0.]])
    print("Global Orientation is: ", global_frame_1,global_frame_2,global_frame_3)

    transl = torch.rand(batch_size, 3)
    output = rh_model(betas=betas,
                      global_orient=global_orient,
                      hand_pose=pose,
                      transl=transl,
                      return_verts=True,
                      return_tips=True)
    mvs[0][0].set_dynamic_meshes([Mesh(v=output.vertices.cpu().detach().numpy().squeeze(), f=rh_model.faces, vc=name_to_rgb['pink'])],blocking=True)

    v = rh_model.hand_meshes(output)[0].vertices.view(np.ndarray) * 1000
    f = rh_model.hand_meshes(output)[0].faces.view(np.ndarray)
    io.savemat('./Output/mano_vert.mat', {'v': v})
    io.savemat('./Output/mano_face.mat', {'f': f})
    print('The mesh has been successfully saved')





#GUI elements declare
vertical1 = Scale(root,label='Shape parameter 1',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical2 = Scale(root,label='Shape parameter 2',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical3 = Scale(root,label='Shape parameter 3',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical4 = Scale(root,label='Shape parameter 4',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical5 = Scale(root,label='Shape parameter 5',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical6 = Scale(root,label='Shape parameter 6',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical7 = Scale(root,label='Shape parameter 7',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical8 = Scale(root,label='Shape parameter 8',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical9 = Scale(root,label='Shape parameter 9',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
vertical10 = Scale(root,label='Shape parameter 10',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)
BatchSize = Scale(root,label='Batch_Size',from_=-7,to=7,orient=HORIZONTAL,command=hand_show)


pose_index_J1 = [Scale(root,label='Index_J1_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Index_J1_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Index_J1_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_index_J2 = [Scale(root,label='Index_J2_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Index_J2_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Index_J2_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_index_J3 = [Scale(root,label='Index_J3_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Index_J3_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Index_J3_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_middle_J1 = [Scale(root,label='Middle_J1_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Middle_J1_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Middle_J1_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_middle_J2 = [Scale(root,label='Middle_J2_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Middle_J2_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Middle_J2_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_middle_J3 = [Scale(root,label='Middle_J3_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Middle_J3_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Middle_J3_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_pinky_J1 = [Scale(root,label='Pinky_J1_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Pinky_J1_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Pinky_J1_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_pinky_J2 = [Scale(root,label='Pinky_J2_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Pinky_J2_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Pinky_J2_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_pinky_J3 = [Scale(root,label='Pinky_J3_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Pinky_J3_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Pinky_J3_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_ring_J1 = [Scale(root,label='Ring_J1_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Ring_J1_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Ring_J1_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_ring_J2 = [Scale(root,label='Ring_J2_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Ring_J2_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Ring_J2_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_ring_J3 = [Scale(root,label='Ring_J3_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Ring_J3_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Ring_J3_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_daumen_J1 = [Scale(root,label='Thumb_J1_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Thumb_J1_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Thumb_J1_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_daumen_J2 = [Scale(root,label='Thumb_J2_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Thumb_J2_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Thumb_J2_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]
pose_daumen_J3 = [Scale(root,label='Thumb_J3_x',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Thumb_J3_y',from_=-45,to=45,orient=HORIZONTAL,command=hand_show),
                 Scale(root,label='Thumb_J3_z',from_=-45,to=45,orient=HORIZONTAL,command=hand_show)]

global_frame_1_scale = Scale(root,label='Rotation base frame X',from_=-90,to=90,orient=HORIZONTAL,command=hand_show)
global_frame_2_scale = Scale(root,label='Rotation base frame Y',from_=-90,to=90,orient=HORIZONTAL,command=hand_show)
global_frame_3_scale = Scale(root,label='Rotation base frame Z',from_=-90,to=90,orient=HORIZONTAL,command=hand_show)

#GUI elements show on grid
vertical1.grid(row=0,column=0)
vertical2.grid(row=1,column=0)
vertical3.grid(row=2,column=0)
vertical4.grid(row=3,column=0)
vertical5.grid(row=4,column=0)
vertical6.grid(row=5,column=0)
vertical7.grid(row=6,column=0)
vertical8.grid(row=7,column=0)
vertical9.grid(row=8,column=0)
vertical10.grid(row=9,column=0)
BatchSize.grid(row=10,column=0)
pose_index_J1[0].grid(row=0,column=1)
pose_index_J1[1].grid(row=0,column=2)
pose_index_J1[2].grid(row=0,column=3)
pose_index_J2[0].grid(row=1,column=1)
pose_index_J2[1].grid(row=1,column=2)
pose_index_J2[2].grid(row=1,column=3)
pose_index_J3[0].grid(row=2,column=1)
pose_index_J3[1].grid(row=2,column=2)
pose_index_J3[2].grid(row=2,column=3)
pose_middle_J1[0].grid(row=3,column=1)
pose_middle_J1[1].grid(row=3,column=2)
pose_middle_J1[2].grid(row=3,column=3)
pose_middle_J2[0].grid(row=4,column=1)
pose_middle_J2[1].grid(row=4,column=2)
pose_middle_J2[2].grid(row=4,column=3)
pose_middle_J3[0].grid(row=5,column=1)
pose_middle_J3[1].grid(row=5,column=2)
pose_middle_J3[2].grid(row=5,column=3)
pose_pinky_J1[0].grid(row=6,column=1)
pose_pinky_J1[1].grid(row=6,column=2)
pose_pinky_J1[2].grid(row=6,column=3)
pose_pinky_J2[0].grid(row=7,column=1)
pose_pinky_J2[1].grid(row=7,column=2)
pose_pinky_J2[2].grid(row=7,column=3)
pose_pinky_J3[0].grid(row=8,column=1)
pose_pinky_J3[1].grid(row=8,column=2)
pose_pinky_J3[2].grid(row=8,column=3)
pose_ring_J1[0].grid(row=9,column=1)
pose_ring_J1[1].grid(row=9,column=2)
pose_ring_J1[2].grid(row=9,column=3)
pose_ring_J2[0].grid(row=10,column=1)
pose_ring_J2[1].grid(row=10,column=2)
pose_ring_J2[2].grid(row=10,column=3)
pose_ring_J3[0].grid(row=11,column=1)
pose_ring_J3[1].grid(row=11,column=2)
pose_ring_J3[2].grid(row=11,column=3)
pose_daumen_J1[0].grid(row=12,column=1)
pose_daumen_J1[1].grid(row=12,column=2)
pose_daumen_J1[2].grid(row=12,column=3)
pose_daumen_J2[0].grid(row=13,column=1)
pose_daumen_J2[1].grid(row=13,column=2)
pose_daumen_J2[2].grid(row=13,column=3)
pose_daumen_J3[0].grid(row=14,column=1)
pose_daumen_J3[1].grid(row=14,column=2)
pose_daumen_J3[2].grid(row=14,column=3)

global_frame_1_scale.grid(row=0,column=4)
global_frame_2_scale.grid(row=1,column=4)
global_frame_3_scale.grid(row=2,column=4)

if __name__ == "__main__":
    mvs = MeshViewers(window_width=800, window_height=1000, shape=[1, 1])
    threading.Thread(target=hand_show).start()
    root.mainloop()
