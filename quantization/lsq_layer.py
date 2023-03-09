"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import paddle
import paddle.nn.functional as F
import math
import numpy as np
from ._quan_base import _Conv2dQ, Qmodes, _LinearQ, _ActQ, _MultiHeadActQ, _MultiHeadLinearQ
# paddle.disable_static()
#import ipdb

__all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ']


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def bit_pass(x: paddle.Tensor) -> paddle.Tensor:
    x.clip(2, 8)
    y = x.numpy()
    y = paddle.to_tensor(y)
    return round_pass(y)
    
def clamp(x, minv, maxv):
    x = paddle.minimum(x, maxv)
    x = paddle.maximum(x, minv)
    return x

class QuantConv2d(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=-1,
                 mode=Qmodes.layer_wise, learned=True, mixpre=True):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode, learned=learned, mixpre=mixpre)
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
    def initialize_scale(self, device):
            # Qn = -2 ** (self.nbits - 1)
            # Qp = 2 ** (self.nbits - 1) - 1
            # self.alpha.data.copy_(quantize_by_mse(self.weight))
            quantize_by_mse(self.weight, self.alpha)
            self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # print(utils.get_rank(), self.alpha.data)
        nbits = bit_pass(self.nbits)
        Qn = -2 ** (nbits - 1)
        Qp = 2 ** (nbits - 1) - 1
        n = int(nbits)
        # if self.init_state == 0:
            # print(f"initialize weight scale for int{self.nbits} quantization")
            # self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(quantize_by_mse(self.weight, Qn, Qp))
            # self.init_state.fill_(1)
            
        assert self.init_state == 1
        with paddle.no_grad():
            g = 1.0 / math.sqrt(float(self.weight.numel()) * Qp)
            # g = 1.0 / math.sqrt(self.weight.numel()) / Qp
        # g = 1.0 / math.sqrt(self.weight.numel()) / 4
        
        aaa = paddle.to_tensor(self.alpha.numpy())
        aaa.clip(min=1e-4)
        self.alpha.set_value(aaa)

        # self.alpha = paddle.to_tensor(self.alpha)
        # self.alpha.clip(min=1e-4)
        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha[n-2], g)
        # w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = clamp(round_pass(self.weight / alpha), Qn, Qp) * alpha
        w_q = round_pass(clamp(self.weight / alpha, Qn, Qp)) * alpha
        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # return F.conv2d(x, w_q, self.bias)

class QuantLinear(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=-1, learned=True, mixpre=True, **kwargs):
        super(QuantLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits, learned=learned, mixpre=mixpre)

    def initialize_scale(self, device):
        # Qn = -2 ** (self.nbits - 1)
        # Qp = 2 ** (self.nbits - 1) - 1
        # self.alpha.data.copy_(quantize_by_mse(self.weight))
        quantize_by_mse(self.weight, self.alpha)
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        nbits = bit_pass(self.nbits)
        Qn = -2 ** (nbits - 1)
        Qp = 2 ** (nbits - 1) - 1
        n = int(nbits)
        # print(utils.get_rank(), self.alpha.data)
        # if self.init_state == 0:
            # self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # lsq+ init
            # m, v = self.weight.abs().mean(), self.weight.abs().std()
            # self.alpha.data.copy_(paddle.max(paddle.abs(m - 3*v), paddle.abs(m + 3*v)) / 2 ** (self.nbits - 1) )
        assert self.init_state == 1
        with paddle.no_grad():
            g = 1.0 / math.sqrt(float(self.weight.numel()) * Qp)
            # g = 1.0 / math.sqrt(self.weight.numel()) / Qp
        # g = 1.0 / math.sqrt(self.weight.numel()) / 4
        aaa = paddle.to_tensor(self.alpha.numpy())
        aaa.clip(min=1e-4)
        self.alpha.set_value(aaa)

        # self.alpha = paddle.to_tensor(self.alpha)
        # self.alpha.clip(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha[n-2], g)
        # w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = clamp(round_pass(self.weight / alpha), Qn, Qp) * alpha
        w_q = round_pass(clamp(self.weight / alpha, Qn, Qp)) * alpha
        

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)

class QuantAct(_ActQ):
    def __init__(self, nbits=-1, signed=True, offset=False, dim=1, learned=True, mixpre=True, **kwargs):
        super(QuantAct, self).__init__(nbits=nbits, signed=signed, offset=offset, dim=dim, learned=learned, mixpre=mixpre)
        self.act_samples = np.zeros(1)
        print('Using regular quant')

    def initialize_scale_offset(self, device):
        # if self.signed:
            # Qn = -2 ** (self.nbits - 1)
            # Qp = 2 ** (self.nbits - 1) - 1
        # else:
            # Qn = 0 * self.nbits
            # Qp = 2 ** self.nbits - 1
        assert self.act_samples.any(), "no activation samples!"
        act_samples = paddle.to_tensor(self.act_samples)
        if self.offset:
            quantize_by_mse_with_offset(act_samples, self.alpha, self.beta, signed=self.signed)
            # alpha, beta = quantize_by_mse_with_offset(act_samples, signed=self.signed)

                # self.beta.data.copy_(x.mean())
                # self.beta.data.copy_((x.max() + x.min()) / 2)
            # self.beta.data.copy_(beta)
            # self.alpha.data.copy_(alpha)
        else:
            # self.alpha.data.copy_(quantize_by_mse(act_samples, signed=self.signed))
            quantize_by_mse(act_samples, self.alpha, signed=self.signed)
        self.init_state.fill_(1)
        del act_samples
        self.act_samples = None

    def forward(self, x):
        if self.alpha is None:
            return x
        # print(utils.get_rank(), self.alpha.data)

        if self.init_state == 0:
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.init_state.fill_(1)
            sample = x.cpu().numpy()
            if not self.act_samples.any():
                self.act_samples = sample
            else:
                self.act_samples = np.concatenate((self.act_samples, sample), axis=0)
            return x
            
        assert self.init_state == 1
        nbits = bit_pass(self.nbits)
        n = int(nbits)
        if self.signed:
            Qn = -2 ** (nbits - 1)
            Qp = 2 ** (nbits - 1) - 1
        else:
            Qn = 0 * self.nbits
            Qp = 2 ** nbits - 1
        with paddle.no_grad():
            g = 1.0 / math.sqrt(float(x.numel()) * Qp)
            # g = 1.0 / math.sqrt(x.numel()) / Qp
        # g_o = 1.0 / math.sqrt(x.numel())
        # g_o = 1.0 / x.numel()
        # g = 1.0 / math.sqrt(Qp)
        # g = 1.0 / math.sqrt(x.numel()) / 4
        aaa = paddle.to_tensor(self.alpha.numpy())
        aaa.clip(min=1e-4)
        self.alpha.set_value(aaa)

        # self.alpha = paddle.to_tensor(self.alpha)
        # self.alpha.clip(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha[n-2], g)
        beta = 0
        if self.offset:
            # x = x.clamp(max=1)
            beta = self.beta[n-2]
            # beta = grad_scale(self.beta, g_o)
        # x_q = round_pass(((x - beta) / alpha).clamp(Qn, Qp))
        # x_q = clamp(round_pass((x - beta) / alpha), Qn, Qp)
        x_q = round_pass(clamp((x-beta) / alpha, Qn, Qp))
        x_out = x_q * alpha + beta
        # if self.offset:
            # self.beta.data.copy_((x - alpha * x_q).mean())

        # Method2:
        # x = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x_out

class QuantMultiHeadAct(_MultiHeadActQ):
    def __init__(self, nbits=-1, signed=True, offset=False, dim=1, learned=True, mixpre=True, num_head=1):
        super(QuantMultiHeadAct, self).__init__(nbits=nbits, signed=signed, offset=offset, dim=dim, learned=learned, mixpre=mixpre, num_head=num_head)
        self.act_samples = np.zeros(1)
        print('Using multihead quant')

    def initialize_scale_offset(self, device):
        # if self.signed:
            # Qn = -2 ** (self.nbits - 1)
            # Qp = 2 ** (self.nbits - 1) - 1
        # else:
            # Qn = 0 * self.nbits
            # Qp = 2 ** self.nbits - 1
        assert self.act_samples.any(), "no activation samples!"
        act_samples = paddle.from_numpy(self.act_samples).to(device)
        if self.offset:
            quantize_by_mse_with_offset(act_samples, self.alpha, self.beta, signed=self.signed)
            # alpha, beta = quantize_by_mse_with_offset(act_samples, signed=self.signed)

                # self.beta.data.copy_(x.mean())
                # self.beta.data.copy_((x.max() + x.min()) / 2)
            # self.beta.data.copy_(beta)
            # self.alpha.data.copy_(alpha)
        else:
            # self.alpha.data.copy_(quantize_by_mse(act_samples, signed=self.signed))
            quantize_by_mse(act_samples, self.alpha, signed=self.signed)
        self.init_state.fill_(1)
        del act_samples
        self.act_samples = None

    def forward(self, x):
        if self.alpha is None:
            return x
        # print(utils.get_rank(), self.alpha.data)

        if self.init_state == 0:
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.init_state.fill_(1)
            sample = x.cpu().numpy()
            if not self.act_samples.any():
                self.act_samples = sample
            else:
                self.act_samples = np.concatenate((self.act_samples, sample), axis=0)
            return x
            
        assert self.init_state == 1
        nbits = bit_pass(self.nbits)

        if self.signed:
            Qn = -2 ** (nbits - 1)
            Qp = 2 ** (nbits - 1) - 1
        else:
            Qn = 0 * self.nbits
            Qp = 2 ** nbits - 1
        with paddle.no_grad():
            g = 1.0 / paddle.sqrt(float(x.numel()) / self.num_head * Qp)
            # g = 1.0 / math.sqrt(x.numel()) / Qp
        # g_o = 1.0 / math.sqrt(x.numel())
        # g_o = 1.0 / x.numel()
        # g = 1.0 / math.sqrt(Qp)
        # g = 1.0 / math.sqrt(x.numel()) / 4
        aaa = paddle.to_tensor(self.alpha.numpy())
        aaa.clip(min=1e-4)
        self.alpha.set_value(aaa)

        # self.alpha = paddle.to_tensor(self.alpha)
        # self.alpha.clip(min=1e-4)
        # x.shape : B H N D
        Qp = Qp.reshape(1, -1, 1, 1)
        Qn = Qn.reshape(1, -1, 1, 1)
        n = nbits.to(paddle.long).detach()
        # Method1:
        alpha = grad_scale(self.alpha[n-2], g)
        # print(n)
        # print(alpha)
        alpha = alpha.reshape(1, -1, 1, 1)
        # x_q = clamp(round_pass(x / alpha), Qn, Qp)
        x_q = round_pass(clamp(x / alpha, Qn, Qp))
        x_out = x_q * alpha

        # Method2:
        # x = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x_out

class QuantMuitiHeadLinear(_MultiHeadLinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=-1, learned=True, mixpre=True, num_head=1):
        super(QuantMuitiHeadLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits, learned=learned, mixpre=mixpre, num_head=num_head)

    def initialize_scale(self, device):
        # Qn = -2 ** (self.nbits - 1)
        # Qp = 2 ** (self.nbits - 1) - 1
        # self.alpha.data.copy_(quantize_by_mse(self.weight))
        quantize_by_mse(self.weight, self.alpha)
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        nbits = bit_pass(self.nbits)
        Qn = -2 ** (nbits - 1)
        Qp = 2 ** (nbits - 1) - 1
        assert self.init_state == 1
        # self.weight.shape = (Cin, Cout)
        Cin, Cout = self.weight.shape
        # weight.shape = (Cin, H, Cout/H)
        weight = self.weight.reshape(Cin, self.num_head, Cout // self.num_head)
        with paddle.no_grad():
            g = 1.0 / paddle.sqrt(float(weight.numel()) / self.num_head * Qp)
        Qp = Qp.reshape(1, -1, 1)
        Qn = Qn.reshape(1, -1, 1)
        n = nbits.to(paddle.long).detach()
        
        aaa = paddle.to_tensor(self.alpha.numpy())
        aaa.clip(min=1e-4)
        self.alpha.set_value(aaa)

        # self.alpha = paddle.to_tensor(self.alpha)
        # self.alpha.clip(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha[n-2], g)
        alpha = alpha.reshape(1, -1, 1)

        w_q = round_pass(clamp(weight / alpha, Qn, Qp)) * alpha
        w_q = w_q.reshape(Cin, Cout)

        return F.linear(x, w_q, self.bias)

class QuantMuitiHeadLinear_in(_MultiHeadLinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=-1, learned=True, mixpre=True, num_head=1):
        super(QuantMuitiHeadLinear_in, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits, learned=learned, mixpre=mixpre, num_head=num_head)

    def initialize_scale(self, device):
        # Qn = -2 ** (self.nbits - 1)
        # Qp = 2 ** (self.nbits - 1) - 1
        # self.alpha.data.copy_(quantize_by_mse(self.weight))
        quantize_by_mse(self.weight, self.alpha)
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        nbits = bit_pass(self.nbits)
        Qn = -2 ** (nbits - 1)
        Qp = 2 ** (nbits - 1) - 1
        assert self.init_state == 1
        # self.weight.shape = (Cin, Cout)
        Cin, Cout = self.weight.shape
        # weight.shape = (Cin, H, Cout/H)
        weight = self.weight.reshape(self.num_head, Cin // self.num_head , Cout)
        with paddle.no_grad():
            g = 1.0 / paddle.sqrt(float(weight.numel())/ self.num_head * Qp)
        Qp = Qp.reshape(-1, 1, 1)
        Qn = Qn.reshape(-1, 1, 1)
        n = nbits.to(paddle.long).detach()
        
        aaa = paddle.to_tensor(self.alpha.numpy())
        aaa.clip(min=1e-4)
        self.alpha.set_value(aaa)

        # self.alpha = paddle.to_tensor(self.alpha)
        # self.alpha.clip(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha[n-2], g)
        alpha = alpha.reshape(-1, 1, 1)

        w_q = round_pass(clamp(weight / alpha, Qn, Qp)) * alpha
        w_q = w_q.reshape(Cin, Cout)

        return F.linear(x, w_q, self.bias)

def quantize_by_mse(tensor, p_alpha, signed=True):

    # size = tensor.size()
    a = tensor.numpy()
    a = paddle.to_tensor(a)
    flatten_tensor = paddle.reshape(a,[-1])
    # print(flatten_tensor)

    for n in range(8, 9):
        if signed:
            Qn, Qp = -2 ** (n - 1), 2 ** (n - 1) - 1
        else:
            Qn, Qp = 0, 2 ** n - 1
        alpha = flatten_tensor.abs().max() / Qp
        alpha_old = -1.0
        eps = 1e-5
        step = 0
        while paddle.abs((alpha - alpha_old) / alpha) >= eps:
            flatten_tensor_q = (flatten_tensor / alpha).clip(Qn, Qp).round()
            alpha_old = alpha
            alpha = flatten_tensor.dot(flatten_tensor_q) / flatten_tensor_q.dot(flatten_tensor_q)
            if not math.isfinite(alpha):
                raise ValueError("alpha is infinite!")
            step += 1
        print(f"scale initial value for {n}-bit: {alpha}, iter steps: {step}")
        aaa = p_alpha.numpy()
        aaa[n-2] = alpha
        aaa = paddle.to_tensor(aaa)
        p_alpha.set_value(aaa)

def quantize_by_mse_with_offset(tensor, p_alpha, p_beta, signed=True):

    # size = tensor.size()
    # tensor = tensor.clamp(max=1)
    a = tensor.numpy()
    a = paddle.to_tensor(a)
    flatten_tensor = paddle.reshape(a,[-1])
    # print(flatten_tensor)
    for n in range(8, 9):
        if signed:
            Qn, Qp = -2 ** (n - 1), 2 ** (n - 1) - 1
        else:
            Qn, Qp = 0, 2 ** n - 1
        beta = flatten_tensor.min()
        # beta = flatten_tensor.mean()
        # beta = paddle.tensor([-0.17]).to(tensor.device)
        flatten_tensor_nomin = flatten_tensor - beta
        alpha = flatten_tensor_nomin.abs().max() / Qp
        alpha_old = -1.0
        eps = 1e-5
        step = 0
        while paddle.abs((alpha - alpha_old) / alpha) >= eps:
            flatten_tensor_q = (flatten_tensor_nomin / alpha).clip(Qn, Qp).round()
            alpha_old = alpha
            beta = (flatten_tensor - alpha * flatten_tensor_q).mean()
            alpha = flatten_tensor_nomin.dot(flatten_tensor_q) / flatten_tensor_q.dot(flatten_tensor_q)
            flatten_tensor_nomin = flatten_tensor - beta
            if not math.isfinite(alpha):
                raise ValueError("alpha is infinite!")
            step += 1
        print(f"scale initial value for {n}-bit: alpha:{alpha}, beta:{beta}, iter steps: {step}")
        aaa = p_alpha.numpy()
        aaa[n-2] = alpha
        aaa = paddle.to_tensor(aaa)
        p_alpha.set_value(aaa)

        
        bbb = p_beta.numpy()
        bbb[n-2] = beta
        bbb = paddle.to_tensor(bbb)
        p_beta.set_value(bbb)