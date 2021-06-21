import numpy as np
import torch
import tomosipo as ts
import tomosipo.torch_support
from tomosipo.qt import animate
import ts_algorithms as tsa
from matplotlib import pyplot as plt

num_proj = 500

# Setup 2D volume and fan beam projection geometry
vg = ts.volume(shape=(1, 256, 256), size=(0.1, 0.5, 0.5))
pg = ts.cone(shape=(1, 384), size=(0.1, 5), src_orig_dist=2.25, src_det_dist=3)

# Setup the a rotation and translation transform
tra = ts.translate(axis=np.array((0, 0, 1)), alpha=np.linspace(-2.5, 2.5, num_proj))
rot = ts.rotate(pos=0, axis=np.array((1, 0, 0)), angles=np.linspace(0, 3*np.pi, num_proj))

# Apply transformations to the volume geometry
vg = tra * rot * vg.to_vec()
pg = pg.to_vec() 

# Create an operator from the transformed geometries
A = ts.operator(vg, pg)

# Make an animation of the geometries and save it
s = ts.scale(1.8)
animation = animate(s * vg, s * pg)
animation.save("geometry_video.mp4")

# Create hollow cube phantom and copy it to the GPU
x = torch.zeros(A.domain_shape)
x[:, 10:-10, 10:-10] = 1.0
x[:, 40:-40, 40:-40] = 0.0
x = x.cuda()

# Project the volume data to obtain the projection data
y = A(x)

# Reconstruct using SIRT and copy everything back to RAM
recon = tsa.sirt(A, y, num_iterations=100)
recon = recon.cpu(); x = x.cpu(); y = y.cpu()

plt.figure(figsize=(9, 3))
plt.subplot(131); plt.imshow(x[0, ...]); plt.title("Volume data")
plt.subplot(132); plt.imshow(y[0, ...]); plt.title("Projection data")
plt.subplot(133); plt.imshow(recon[0, ...]); plt.title("Reconstruction")
plt.show()
