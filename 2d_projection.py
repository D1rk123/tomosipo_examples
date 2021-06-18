import numpy as np
import tomosipo as ts
from matplotlib import pyplot as plt

# Setup 2D volume and parallel projection geometry
vg = ts.volume(shape=(1, 256, 256))
pg = ts.parallel(angles=384, shape=(1, 384))

# Create an operator from the geometries
A = ts.operator(vg, pg)

# Create hollow cube phantom
x = np.zeros(A.domain_shape, dtype=np.float32)
x[:, 10:-10, 10:-10] = 1.0
x[:, 40:-40, 40:-40] = 0.0

# Project the volume data to obtain the projection data and backproject it again
y = A(x)
b = A.T(y)

plt.figure(figsize=(9, 3))
plt.subplot(131); plt.imshow(x[0, ...]); plt.title("Volume data")
plt.subplot(132); plt.imshow(y[0, ...]); plt.title("Projection data")
plt.subplot(133); plt.imshow(b[0, ...]); plt.title("Backprojection")
plt.show()

