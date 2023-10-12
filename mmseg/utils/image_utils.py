# image utils

def split_images(x):
    if x.ndim == 4:
        channels = x.shape[1]
        channels //= 2
        x1 = x[:, 0:channels, :, :]
        x2 = x[:, channels:, :, :]
    elif x.ndim == 3:
        channels = x.shape[-1]
        channels //= 2
        x1 = x[..., 0:channels]
        x2 = x[..., channels:]
    else:
        raise ValueError(f'dimension of x should be 3 or 4, but got {x.ndim}')
    return x1, x2