import numpy as np


def forward_prop(X, W, b, activation, padding='same', stride=(1, 1)):

    m, h_in, w_in, c_in = X.shape
    kh, kw, kc_in, kc_out = W.shape
    sh, sw = stride
    if padding=='same':
        padh = int(((h_in - 1) * sh - h_in + kh ) / 2)
        padw = int(((w_in - 1) * sw -w_in + kw) / 2)
    else:
        padh = padw = 0
    h_out = int(((h_in + 2 * padh - kh )/ sh) + 1)
    w_out = int(((w_in + 2 *padw - kw) / sw) + 1)
    z = np.zeros([m, h_out, w_out, kc_out])
    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    x_pad  = np.pad(X, pad_width=pad, mode='constant', constant_values=0)
    for h in range(h_out):
        for w in range(h_in):
            for c in range(kc_out):
                x = w * sw
                y = h * sh
                x_slice = x_pad[:, y:y+kh, x:x+kw, :]
                z[:, h, w, c] = np.multiply(x_slice, W[:, :, :, c]).sum(axis=(1, 2, 3))
            z = z + b 
            return activation(z)


    
