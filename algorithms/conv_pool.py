import numpy as np

class Conv2d:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride)
        self.padding = self._to_tuple(padding)
        self.dilation = self._to_tuple(dilation)
        self.groups = groups
        self.bias_flag = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        
        # Ядро: (out_channels, in_channels, kH, kW)
        self.weight = np.random.randn(out_channels, in_channels,
                                      self.kernel_size[0], self.kernel_size[1])
        self.bias = np.random.randn(out_channels) if self.bias_flag else np.zeros(out_channels)


        # Проверка padding_mode
        assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular'], \
            f"Unsupported padding_mode: {padding_mode}"

    def _to_tuple(self, val):
        if isinstance(val, int):
            return (val, val)
        elif isinstance(val, tuple) and len(val) == 2:
            return val
        else:
            raise ValueError(f"Expected int or tuple of length 2, got {val}")

    def __call__(self, x: np.ndarray):
        C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        # Паддинг
        if self.padding_mode == 'zeros':
            x = np.pad(x, pad_width=((0, 0), (pH, pH), (pW, pW)),
                       mode='constant', constant_values=0)
        elif self.padding_mode == 'reflect':
            x = np.pad(x, pad_width=((0, 0), (pH, pH), (pW, pW)),
                       mode='reflect')
        elif self.padding_mode == 'replicate':
            x = np.pad(x, pad_width=((0, 0), (pH, pH), (pW, pW)),
                       mode='edge')
        elif self.padding_mode == 'circular':
            x = np.pad(x, pad_width=((0, 0), (pH, pH), (pW, pW)),
                       mode='wrap')

        # Вычисление размеров выходной карты признаков
        H_out = ((H + 2*p - d*(kH - 1) - 1) // s) + 1
        W_out = ((W + 2*p - d*(kW - 1) - 1) // s) + 1
        out = np.zeros((self.out_channels, H_out, W_out))

        # Свёртка
        for oc in range(self.out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * s
                    w_start = j * s
                    sum_val = 0
                    for ic in range(self.in_channels):
                        for ki in range(kH):
                            for kj in range(kW):
                                h_index = h_start + ki * d
                                w_index = w_start + kj * d
                                sum_val += x[ic, h_index, w_index] * self.weight[oc, ic, ki, kj]
                    out[oc, i, j] = sum_val + self.bias[oc]

        return out

class MaxPool2d:
    def __init__(self, kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):

        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride if stride is not None else kernel_size)
        self.padding = self._to_tuple(padding)
        self.dilation = self._to_tuple(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def _to_tuple(self, val):
        if isinstance(val, int):
            return (val, val)
        elif isinstance(val, tuple) and len(val) == 2:
            return val
        else:
            raise ValueError(f"Expected int or tuple of length 2, got {val}")

    def __call__(self, x):
        # x: (C, H, W)
        C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        # Calculate output shape
        H_out = ((H + 2 * pH - dH * (kH - 1) - 1) / sH) + 1
        W_out = ((W + 2 * pW - dW * (kW - 1) - 1) / sW) + 1
        if self.ceil_mode:
            H_out = int(np.ceil(H_out))
            W_out = int(np.ceil(W_out))
        else:
            H_out = int(np.floor(H_out))
            W_out = int(np.floor(W_out))

        # Padding with -inf (like implicit padding in PyTorch)
        x_padded = np.full((C, H + 2 * pH, W + 2 * pW), -np.inf, dtype=x.dtype)
        x_padded[:, pH:pH+H, pW:pW+W] = x

        out = np.full((C, H_out, W_out), -np.inf, dtype=x.dtype)
        indices = np.zeros((C, H_out, W_out), dtype=np.int32) if self.return_indices else None

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * sH
                    w_start = j * sW
                    window = []
                    index_map = []

                    for ki in range(kH):
                        for kj in range(kW):
                            h_idx = h_start + ki * dH
                            w_idx = w_start + kj * dW
                            if 0 <= h_idx < x_padded.shape[1] and 0 <= w_idx < x_padded.shape[2]:
                                val = x_padded[c, h_idx, w_idx]
                                window.append(val)
                                index_map.append((h_idx, w_idx))

                    if window:
                        max_val = max(window)
                        out[c, i, j] = max_val
                        if self.return_indices:
                            max_idx = window.index(max_val)
                            h_max, w_max = index_map[max_idx]
                            # flatten index into original x
                            indices[c, i, j] = (h_max - pH) * W + (w_max - pW)

        return (out, indices) if self.return_indices else out