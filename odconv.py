import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.autograd
import numpy as np
from paddle import ParamAttr
from paddle.nn.initializer import  Constant

# class Attention(nn.Layer):
#     def __init__(self, in_channels, out_channels, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
#         super(Attention, self).__init__()
#         attention_channel = max(int(in_channels * reduction), min_channel)
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num
#         self.temperature = 10.0
#
#         self.avgpool = nn.AdaptiveAvgPool2D(1)
#         # 因为其后面就是BN层所以不需要使用偏置项
#         self.fc = nn.Conv2D(in_channels, attention_channel, 1, bias_attr=False,
#                             weight_attr=nn.initializer.KaimingNormal())
#         self.bn = nn.BatchNorm2D(attention_channel)
#         self.relu = nn.ReLU()  # inplace=True
#
#         self.channel_fc = nn.Conv2D(attention_channel, in_channels, 1, bias_attr=True,
#                                     weight_attr=nn.initializer.KaimingNormal())
#         self.func_channel = self.get_channel_attention
#
#         if in_channels == groups and in_channels == out_channels:  # depth-wise convolution
#             self.func_filter = self.skip
#         else:
#             self.filter_fc = nn.Conv2D(attention_channel, out_channels, 1, bias_attr=True,
#                                        weight_attr=nn.initializer.KaimingNormal())
#             self.func_filter = self.get_filter_attention
#
#         if kernel_size == 1:  # point-wise convolution
#             self.func_spatial = self.skip
#         else:
#             self.spatial_fc = nn.Conv2D(attention_channel, kernel_size * kernel_size, 1, bias_attr=True,
#                                         weight_attr=nn.initializer.KaimingNormal())
#             self.func_spatial = self.get_spatial_attention
#
#         if kernel_num == 1:
#             self.func_kernel = self.skip
#         else:
#             self.kernel_fc = nn.Conv2D(attention_channel, kernel_num, 1, bias_attr=True,
#                                        weight_attr=nn.initializer.KaimingNormal())
#             self.func_kernel = self.get_kernel_attention
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.sublayers():
#             if isinstance(m, nn.Conv2D):
#                 # m.weight.set_value(nn.initializer.KaimingNormal(nonlinearity='relu')._value)
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     # nn.init.constant_(m.bias, 0)
#                     m.bias.set_value(np.zeros(m.bias.shape, dtype=np.float32))
#             if isinstance(m, nn.BatchNorm2D):
#                 # nn.init.constant_(m.weight, 1)
#                 # nn.init.constant_(m.bias, 0)
#                 m.weight.set_value(np.ones(m.weight.shape, dtype=np.float32))
#                 m.bias.set_value(np.zeros(m.bias.shape, dtype=np.float32))
#
#     def update_temperature(self, temperature):
#         self.temperature = temperature
#
#     @staticmethod
#     def skip(_):
#         return 1.0
#
#     def get_channel_attention(self, x):
#         channel_attention = paddle.nn.functional.sigmoid(
#             self.channel_fc(x).reshape((x.shape[0], -1, 1, 1)) / self.temperature)
#         return channel_attention
#
#     def get_filter_attention(self, x):
#         filter_attention = paddle.nn.functional.sigmoid(
#             self.filter_fc(x).reshape((x.shape[0], -1, 1, 1)) / self.temperature)
#         return filter_attention
#
#     def get_spatial_attention(self, x):
#         spatial_attention = self.spatial_fc(x).reshape(
#             (x.shape[0], 1, 1, 1, self.kernel_size, self.kernel_size))  # x.size(0) view
#         spatial_attention = paddle.nn.functional.sigmoid(spatial_attention / self.temperature)
#         return spatial_attention
#
#     def get_kernel_attention(self, x):
#         kernel_attention = self.kernel_fc(x).reshape((x.shape[0], -1, 1, 1, 1, 1))
#         kernel_attention = F.softmax(kernel_attention / self.temperature, axis=1)
#         return kernel_attention
#
#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
#
#
# class ODConv2D(nn.Layer):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                  reduction=0.0625, kernel_num=4):
#         super(ODConv2D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.kernel_num = kernel_num
#         self.attention = Attention(in_channels, out_channels, kernel_size, groups=groups,
#                                    reduction=reduction, kernel_num=kernel_num)
#         # self.weight = nn.Parameter(torch.randn(kernel_num, out_channels, in_channels//groups, kernel_size, kernel_size),requires_grad=True)
#         self.weight = paddle.create_parameter((kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size),
#                                               np.float32, default_initializer=nn.initializer.KaimingNormal())
#         # self.weight在paddle.create_parameter时已经进行了KaimingNormal初始化，故不需要调用_initialize_weights
#         # self._initialize_weights()
#
#         if self.kernel_size == 1 and self.kernel_num == 1:
#             self._forward_impl = self._forward_impl_pw1x
#         else:
#             self._forward_impl = self._forward_impl_common
#
#     def _initialize_weights(self):
#         for i in range(self.kernel_num):
#             # nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
#             pass
#
#     def update_temperature(self, temperature):
#         self.attention.update_temperature(temperature)
#
#     def _forward_impl_common(self, x):
#         # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
#         # while we observe that when using the latter method the models will run faster with less gpu memory cost.
#         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
#         batch_size, in_channels, height, width = x.shape
#         x = x * channel_attention
#         x = x.reshape([1, -1, height, width])
#         aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(axis=0)
#         aggregate_weight = paddle.sum(aggregate_weight, axis=1).reshape(
#             [-1, self.in_channels // self.groups, self.kernel_size, self.kernel_size])
#         output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups * batch_size)
#         output = output.reshape((batch_size, self.out_channels, output.shape[-2], output.shape[-1]))
#         output = output * filter_attention
#         return output
#
#     # 对于conv1x1只进行chanel_attention
#     def _forward_impl_pw1x(self, x):
#         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
#         x = x * channel_attention
#         output = F.conv2d(x, weight=self.weight.squeeze(axis=0), bias=None, stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups)
#         output = output * filter_attention
#         return output
#
#     def forward(self, x):
#         return self._forward_impl(x)


# 核心修改
# reshape操作的静态化，对所有的reshape都不使用包含-1的通道项。需新增self.in_planes=in_planes、self.out_planes=out_planes
class Attention_n2(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, min_channel=16):
        super(Attention_n2, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.temperature = 10.0

        self.avgpool = nn.AdaptiveAvgPool2D(1)
        # 因为其后面就是BN层所以不需要使用偏置项
        self.fc = nn.Conv2D(in_planes, attention_channel, 1, bias_attr=False,
                            weight_attr=nn.initializer.KaimingNormal())
        self.bn = nn.BatchNorm2D(attention_channel)
        self.relu = nn.ReLU()  # inplace=True

        self.channel_fc = nn.Conv2D(attention_channel, in_planes, 1, bias_attr=True,
                                    weight_attr=nn.initializer.KaimingNormal())
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2D(attention_channel, out_planes, 1, bias_attr=True,
                                       weight_attr=nn.initializer.KaimingNormal())
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            # change shape to 1,kernel_size * kernel_size
            self.spatial_fc = nn.Conv2D(attention_channel, kernel_size * kernel_size, 1, bias_attr=True,
                                        weight_attr=nn.initializer.KaimingNormal())
            self.func_spatial = self.get_spatial_attention
            # final shape 1,1,kernel_size , kernel_size

        self.kernel_fc = nn.Conv2D(attention_channel, in_planes * out_planes, 1, bias_attr=True,
                                   weight_attr=nn.initializer.KaimingNormal())
        self.func_kernel = self.get_kernel_attention
        # final shape in_planes,out_planes,1 , 1

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = paddle.nn.functional.sigmoid(
            self.channel_fc(x).reshape((paddle.shape(x)[0], self.in_planes, 1, 1)) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = paddle.nn.functional.sigmoid(
            self.filter_fc(x).reshape((paddle.shape(x)[0], self.out_planes, 1, 1)) / self.temperature)
        return filter_attention

    # attention.shape 1, 1, self.kernel_size, self.kernel_size
    def get_spatial_attention(self, x):
        x = paddle.mean(self.spatial_fc(x), axis=[0])
        spatial_attention = x.reshape((1, 1, self.kernel_size, self.kernel_size))  # x.size(0) view
        spatial_attention = paddle.nn.functional.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    # attention.shape self.out_planes,self.in_planes, 1, 1
    def get_kernel_attention(self, x):
        x = paddle.mean(self.kernel_fc(x), axis=[0])
        kernel_attention = x.reshape((self.out_planes, self.in_planes, 1, 1))
        kernel_attention = F.softmax(kernel_attention / self.temperature, axis=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2D(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention_n2(in_planes, out_planes, kernel_size, groups=groups,
                                      reduction=reduction)
        # self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),requires_grad=True)
        self.weight = paddle.create_parameter((out_planes, in_planes // groups, kernel_size, kernel_size), np.float32,
                                              default_initializer=nn.initializer.KaimingNormal())
        # self.weight在paddle.create_parameter时已经进行了KaimingNormal初始化，故不需要调用_initialize_weights
        # self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        # batch_size, in_planes, height, width = x.shape

        x = x * channel_attention
        # x = x.reshape([1, -1, height, width])

        aggregate_weight = spatial_attention * kernel_attention * self.weight
        # print('aggregate_weight:',aggregate_weight.shape)
        # print('x_shape',x.shape)
        # aggregate_weight = paddle.sum(aggregate_weight, axis=[0,1]).reshape(
        #    [self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        # output = output.reshape((batch_size, self.out_planes, output.shape[-2], output.shape[-1]))
        output = output * filter_attention
        # print('aggregate_weight-x.shape:',aggregate_weight.shape,x.shape)
        return output

    # 对于conv1x1只进行chanel_attention
    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(axis=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


class LearnableAffineBlock(nn.Layer):
    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 lr_mult=1.0,
                 lab_lr=0.01):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias