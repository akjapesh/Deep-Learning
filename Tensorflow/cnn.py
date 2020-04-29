import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X,pad):
    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
    return X_pad

def conv_single_step(a_slice_prev,W,b):
    s=a_slice_prev*W
    Z=np.sum(s,dtype=np.float32)
    Z=Z+float(b)
    return Z
def conv_forward(A_prev,W,b,hparameters):
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (f,f,n_C_prev,n_C)=W.shape
    stride=hparameters["stride"]
    pad=hparameters["pad"]
    n_H=int((n_H_prev+2*pad-f)/stride+1)
    n_W=int((n_W_prev + 2 * pad - f) / stride + 1)
    Z=np.zeros((m,n_H,n_W,n_C))
    A_prev_pad=zero_pad(A_prev,pad)
    for i in range(m):
        a_prev_pad=A_prev_pad[i]
        for h in range(n_H):
            vert_start=h*stride
            vert_end=h*stride+f
            for w in range(n_W):
                horiz_start=w*stride
                horiz_end=w*stride+f
                for c in range(n_C):
                    a_slice_prev=a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    weights=W[:,:,:,c]
                    biases=b[:,:,:,c]
                    Z[i,h,w,c]=conv_single_step(a_slice_prev,weights,biases)
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache

def pool_forward(A_prev,hparameters,mode="max"):
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    f=hparameters["f"]
    stride=hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            vert_start=h*stride
            vert_end=vert_start+f
            for w in range(n_W):
                horiz_start=w*stride
                horiz_end=horiz_start+f
                for c in range(n_C):
                    a_prev_slide=A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode=="max":
                        A[i,h,w,c]=np.max(a_prev_slide)
                    elif mode=="average":
                        A[i,h,w,c]=np.mean(a_prev_slide)
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1,1])
print ("x_pad[1,1] =\n", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])