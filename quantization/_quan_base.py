"""
    Quantized modules: the base class
"""
import paddle
import paddle.nn as nn
from enum import Enum
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid import unique_name
# paddle.disable_static()
__all__ = ['Qmodes', '_Conv2dQ', '_LinearQ', '_ActQ']


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


class _Conv2dQ(nn.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias_attr=None, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias_attr=bias_attr)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        # self.nbits = self.create_parameter(shape=paddle.to_tensor([float(kwargs_q['nbits'])]).shape,
        #                                     dtype=str(paddle.to_tensor([float(kwargs_q['nbits'])]).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([float(kwargs_q['nbits'])])))
        # self.add_parameter("nbits", self.nbits)
        # self.nbits.stop_gradient = not kwargs_q['mixpre']
        nbits_attr = ParamAttr(
            name=unique_name.generate("nbits"),
            initializer=paddle.nn.initializer.Assign(paddle.to_tensor([float(kwargs_q['nbits'])])),
            trainable=False)
        self.nbits = self.create_parameter(shape=paddle.to_tensor([float(kwargs_q['nbits'])]).shape,
                                            dtype=str(paddle.to_tensor([float(kwargs_q['nbits'])]).numpy().dtype),
                                            attr = nbits_attr)
        self.nbits.stop_gradient = not kwargs_q['mixpre']
        # self.nbits = Parameter(paddle.tensor([4.]))
        if kwargs_q['nbits'] < 0:
            self.add_parameter("alpha", None)
            return
        self.q_mode = kwargs_q['mode']
        self.learned = kwargs_q['learned']
        if self.q_mode == Qmodes.kernel_wise:
            # self.alpha = self.create_parameter(shape=paddle.to_tensor(out_channels).shape,
            #                                 dtype=str(paddle.to_tensor(out_channels).numpy().dtype),
            #                                 default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(out_channels)))
            # self.add_parameter("alpha", self.alpha)

            alpha_attr = ParamAttr(
                name=unique_name.generate("alpha"),
                initializer=paddle.nn.initializer.Assign(paddle.to_tensor(out_channels)),
                trainable=False)
            self.alpha = self.create_parameter(shape=paddle.to_tensor(out_channels).shape,
                                                dtype=str(paddle.to_tensor(out_channels).numpy().dtype),
                                                attr = alpha_attr)

            self.alpha.stop_gradient = not self.learned
        else:  # layer-wise quantization
            # self.alpha = self.create_parameter(shape=paddle.zeros([7]).shape,
            #                                 dtype=str(paddle.zeros([7]).numpy().dtype),
            #                                 default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))
            # self.add_parameter("alpha", self.alpha)

            alpha_attr = ParamAttr(
                name=unique_name.generate("alpha"),
                initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
                trainable=False)
            self.alpha = self.create_parameter(shape=paddle.zeros([7]).shape,
                                                dtype=str(paddle.zeros([7]).numpy().dtype),
                                                attr = alpha_attr)

            self.alpha.stop_gradient = not self.learned
        self.register_buffer('init_state', paddle.zeros([1]))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias_attr=None, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias_attr=bias_attr)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        # self.nbits = self.create_parameter(shape=paddle.to_tensor([float(kwargs_q['nbits'])]).shape,
        #                                     dtype=str(paddle.to_tensor([float(kwargs_q['nbits'])]).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([float(kwargs_q['nbits'])])))
        # self.add_parameter("nbits", self.nbits)
        # self.nbits.stop_gradient = not kwargs_q['mixpre']

        nbits_attr = ParamAttr(
            name=unique_name.generate("nbits"),
            initializer=paddle.nn.initializer.Assign(paddle.to_tensor([float(kwargs_q['nbits'])])),
            trainable=False)
        self.nbits = self.create_parameter(shape=paddle.to_tensor([float(kwargs_q['nbits'])]).shape,
                                            dtype=str(paddle.to_tensor([float(kwargs_q['nbits'])]).numpy().dtype),
                                            attr = nbits_attr)
        self.nbits.stop_gradient = not kwargs_q['mixpre']

        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.add_parameter("alpha", None)
            return
        # self.alpha = self.create_parameter(shape=paddle.zeros([7]).shape,
        #                                     dtype=str(paddle.zeros([7]).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))
        # self.add_parameter("alpha", self.alpha)

        alpha_attr = ParamAttr(
            name=unique_name.generate("alpha"),
            initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
            trainable=False)
        self.alpha = self.create_parameter(shape=paddle.zeros([7]).numpy().shape,
                                            dtype=str(paddle.zeros([7]).numpy().dtype),
                                            attr = alpha_attr)

        self.alpha.stop_gradient = not self.learned

        self.register_buffer('init_state', paddle.zeros([1]))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)



class _ActQ(nn.Layer):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)

        # self.nbits = kwargs_q['nbits']
        # self.nbits = self.create_parameter(shape=paddle.to_tensor([float(kwargs_q['nbits'])]).shape,
        #                                     dtype=str(paddle.to_tensor([float(kwargs_q['nbits'])]).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([float(kwargs_q['nbits'])])))
        # self.add_parameter("nbits", self.nbits)
        # self.nbits.stop_gradient = not kwargs_q['mixpre']

        nbits_attr = ParamAttr(
            name=unique_name.generate("nbits"),
            initializer=paddle.nn.initializer.Assign(paddle.to_tensor([float(kwargs_q['nbits'])])),
            trainable=False)
        self.nbits = self.create_parameter(shape=paddle.to_tensor([float(kwargs_q['nbits'])]).shape,
                                            dtype=str(paddle.to_tensor([float(kwargs_q['nbits'])]).numpy().dtype),
                                            attr = nbits_attr)
        self.nbits.stop_gradient = not kwargs_q['mixpre']

        self.learned = kwargs_q['learned']
        # if kwargs_q['nbits'] < 0:
        #     paddle.create_parameter(name = 'alpha', attr = 'None')
        #     return
        self.signed = kwargs_q['signed']
        self.offset = kwargs_q['offset']
        self.dim = kwargs_q['dim']

        
        # self.alpha = self.create_parameter(shape=[7],
        #                                 dtype=str(paddle.zeros([7]).numpy().dtype),
        #                                 default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))
        # self.add_parameter("alpha", self.alpha)

        alpha_attr = ParamAttr(
            name=unique_name.generate("alpha"),
            initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
            trainable=False)
        self.alpha = self.create_parameter(shape=paddle.zeros([7]).shape,
                                            dtype=str(paddle.zeros([7]).numpy().dtype),
                                            attr = alpha_attr)
        self.alpha.stop_gradient = not self.learned
        if self.offset:
            # self.beta = self.create_parameter(shape=paddle.zeros([7]).shape,
            #                                 dtype=str(paddle.zeros([7]).numpy().dtype),
            #                                 default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))                             
            # self.add_parameter("beta", self.beta)

            beta_attr = ParamAttr(
                name=unique_name.generate("beta"),
                initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
                trainable=False)
            self.beta = self.create_parameter(shape=paddle.zeros([7]).shape,
                                                dtype=str(paddle.zeros([7]).numpy().dtype),
                                                attr = beta_attr)

            self.beta.stop_gradient = not self.learned
        self.register_buffer('init_state', paddle.zeros([1]))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)

class _MultiHeadActQ(nn.Layer):
    def __init__(self, **kwargs_q):
        super(_MultiHeadActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.num_head = kwargs_q['num_head']
        # self.nbits = self.create_parameter(shape=(paddle.ones([self.num_head]) * kwargs_q['nbits']).shape,
        #                                     dtype=str((paddle.ones([self.num_head]) * kwargs_q['nbits']).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign((paddle.ones([self.num_head]) * kwargs_q['nbits'])))
        # self.add_parameter("nbits", self.nbits)

        nbits_attr = ParamAttr(
            name=unique_name.generate("nbits"),
            initializer=paddle.nn.initializer.Assign((paddle.ones(self.num_head) * kwargs_q['nbits'])),
            trainable=False)
        self.nbits = self.create_parameter(shape=(paddle.ones(self.num_head) * kwargs_q['nbits']).shape,
                                            dtype=str((paddle.ones(self.num_head) * kwargs_q['nbits']).numpy().dtype),
                                            attr = nbits_attr) 

        # self.nbits = Parameter(paddle.tensor([float(kwargs_q['nbits'])]))
        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.add_parameter("alpha", None)
            return
        self.signed = kwargs_q['signed']
        self.offset = kwargs_q['offset']
        self.dim = kwargs_q['dim']
        # self.alpha = self.create_parameter(shape=paddle.zeros([7]).shape,
        #                                     dtype=str(paddle.zeros([7]).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))
        # self.add_parameter("alpha", self.alpha)

        alpha_attr = ParamAttr(
            name=unique_name.generate("alpha"),
            initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
            trainable=False)
        self.alpha = self.create_parameter(shape=paddle.zeros([7]).numpy().shape,
                                            dtype=str(paddle.zeros([7]).numpy().dtype),
                                            attr = alpha_attr)

        self.alpha.stop_gradient = not self.learned
        if self.offset:
            # self.beta = self.create_parameter(shape=paddle.zeros([7]).shape,
            #                                 dtype=str(paddle.zeros([7]).numpy().dtype),
            #                                 default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))
            # self.add_parameter("beta", self.beta)

            beta_attr = ParamAttr(
                name=unique_name.generate("beta"),
                initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
                trainable=False)
            self.beta = self.create_parameter(shape=paddle.zeros([7]).shape,
                                                dtype=str(paddle.zeros([7]).numpy().dtype),
                                                attr = beta_attr)

            self.beta.stop_gradient = not self.learned
        self.register_buffer('init_state', paddle.zeros([1]))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)

class _MultiHeadLinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias_attr=None, **kwargs_q):
        super(_MultiHeadLinearQ, self).__init__(in_features=in_features, out_features=out_features, bias_attr=bias_attr)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        self.num_head = kwargs_q['num_head']
        # self.nbits = self.create_parameter(shape=(paddle.ones(self.num_head) * kwargs_q['nbits']).shape,
        #                                     dtype=str((paddle.ones(self.num_head) * kwargs_q['nbits']).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign((paddle.ones(self.num_head) * kwargs_q['nbits'])))
        # # self.nbits = Parameter(paddle.tensor([float(kwargs_q['nbits'])]))
        # self.add_parameter("nbits", self.nbits)

        nbits_attr = ParamAttr(
            name=unique_name.generate("nbits"),
            initializer=paddle.nn.initializer.Assign((paddle.ones(self.num_head) * kwargs_q['nbits'])),
            trainable=False)
        self.nbits = self.create_parameter(shape=(paddle.ones(self.num_head) * kwargs_q['nbits']).shape,
                                            dtype=str((paddle.ones(self.num_head) * kwargs_q['nbits']).numpy().dtype),
                                            attr = nbits_attr) 

        self.learned = kwargs_q['learned']
        if kwargs_q['nbits'] < 0:
            self.add_parameter("alpha", None)
            return
        # self.alpha = self.create_parameter(shape=paddle.zeros([7]).shape,
        #                                     dtype=str(paddle.zeros([7]).numpy().dtype),
        #                                     default_initializer=paddle.nn.initializer.Assign(paddle.zeros([7])))
        # self.add_parameter("alpha", self.alpha)
        
        alpha_attr = ParamAttr(
            name=unique_name.generate("alpha"),
            initializer=paddle.nn.initializer.Assign(paddle.zeros([7])),
            trainable=False)
        self.alpha = self.create_parameter(shape=paddle.zeros([7]).numpy().shape,
                                            dtype=str(paddle.zeros([7]).numpy().dtype),
                                            attr = alpha_attr)
        self.alpha.stop_gradient = not self.learned


        self.register_buffer('init_state', paddle.zeros([1]))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_MultiHeadLinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 8
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, _LinearQ):
        pass
    elif isinstance(layer_type, _ActQ):
        default.update({
            'signed': True})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q