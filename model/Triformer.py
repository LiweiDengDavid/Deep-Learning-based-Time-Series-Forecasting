
import math

import torch
import torch.nn as nn
from torch.nn import init


class Triformer(nn.Module):
    def __init__(self,configs):
        super(Triformer, self).__init__()

        self.d_feature = configs.d_feature  # 后续图结构  图节点等于维度数
        self.channels = configs.d_dimension
        self.start_fc = nn.Linear(in_features=1, out_features=self.channels)
        self.layers = nn.ModuleList()
        self.skip_generators = nn.ModuleList()
        self.pred_len = configs.pred_len
        self.patch_sizes = configs.patch_sizes
        self.mem_dim = configs.d_dimension
        self.lag = get_multiply(self.patch_sizes)
        self.device = configs.device
        self.d_dimension = configs.d_dimension
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff

        cuts = self.lag
        for patch_size in self.patch_sizes:
            if cuts % patch_size != 0:
                a = cuts % patch_size
                raise Exception('Lag not divisible by patch size')

            cuts = int(cuts / patch_size)
            self.layers.append(Layer(device=self.device, input_dim=self.channels ,
                                     num_nodes=self.d_feature, cuts=cuts,
                                     cut_size=patch_size, factorized=True))
            self.skip_generators.append(WeightGenerator(in_dim=cuts * self.channels, out_dim=self.d_dimension, number_of_weights=1,
                                                        mem_dim=self.mem_dim, num_nodes=self.d_feature,
                                                        device=self.device,factorized=False))

        self.custom_linear = CustomLinear(factorized=False)
        self.projections = nn.Sequential(*[
            nn.Linear(self.d_dimension, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)])

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):

        x = self.start_fc(batch_x.unsqueeze(-1))
        batch_size = x.size(0)
        skip = 0

        for layer, skip_generator in zip(self.layers, self.skip_generators):
            x = layer(x)  #shape的变化：bs,seqlen,d_feature,hidden_dimension  -> bs, num ,d_feature,hidden_dimension    num为每次patch后patch的个数 比如第一次由seqlen变成  seqlen / patch_sizes[0]
            weights, biases = skip_generator()
            skip_inp = x.transpose(2, 1).reshape(batch_size, 1, self.d_feature, -1)  # 把hidden_dimension 和 patch 维度合并
            skip = skip + self.custom_linear(skip_inp, weights[-1], biases[-1])  # 基于随机生成的矩阵进行linear,linear 即为线性attention

        # 上述适当总结：每次的x经过layer切分成patch 经过一定的维度变化，patch和d_feature 维度合并，然后基于随机生成的矩阵进行linear生成当前层的一个变量，
        # 当层数增加时，当前层的变量会包含上一层的信息，通过这种patch信息的整合，patch间信息的整合，来提取序列信息，最后通过linear来映射到目标序列长度
        x = torch.relu(skip).squeeze(1)
        prediction = self.projections(x).transpose(2, 1)
        return prediction


class Layer(nn.Module):
    def __init__(self, device, input_dim, num_nodes, cuts, cut_size, factorized):
        super(Layer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.cuts = cuts
        self.cut_size = cut_size
        self.temporal_embeddings = nn.Parameter(torch.rand(cuts, 1, 1, self.num_nodes, 5).to(device),
                                                requires_grad=True).to(device)

        self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(5, input_dim)]) for _ in range(cuts)])

        self.out_net1 = nn.Sequential(*[
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh(),
        ])

        self.out_net2 = nn.Sequential(*[
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        ])

        self.temporal_att = TemporalAttention(input_dim, factorized=factorized)
        self.weights_generator_distinct = WeightGenerator(input_dim, input_dim, mem_dim=5, num_nodes=num_nodes,
                                                          factorized=factorized, device=self.device,number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(input_dim, input_dim, mem_dim=None, num_nodes=num_nodes,
                                                        factorized=False,device=self.device,number_of_weights=2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: B T N C
        batch_size = x.size(0)

        data_concat = None
        out = 0

        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        for i in range(self.cuts):
            # shape is (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]

            if i != 0:
                out = self.out_net1(out) * self.out_net2(out)

            emb = self.embeddings_generator[i](self.temporal_embeddings[i]).repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([emb, t], dim=1)  # emb shape: bs,1,d_feature,hidden_dimension     t shape:bs,cut_size,d_feature,hidden_dimension

            # attention q用的是随机初始化，编码而来的向量，k，v用的都是  q与t的concat
            out = self.temporal_att(t[:, :1, :, :], t, t, weights_distinct, biases_distinct, weights_shared,
                                    biases_shared)

            if data_concat == None:
                data_concat = out
            else:
                data_concat = torch.cat([data_concat, out], dim=1)

        return self.dropout(data_concat)


class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class TemporalAttention(nn.Module):
    def __init__(self, in_dim, factorized):
        super(TemporalAttention, self).__init__()
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        # query 随机初始化的变量，经过一定的fc后输入到这里
        # k，v为    query与patch版本的序列的concat

        # weights_distinct   两个变量，shape均为：d_feature,hidden_dimension,hidden_dimension
        # biases_distinct    两个变量，shape均为：d_feature,hidden_dimension
        # weights_shared     两个变量，shape均为：hidden_dimension,hidden_dimension
        # biases_shared     两个变量  shape均为: 1,hidden_dimension
        batch_size = query.shape[0]

        # [batch_size, num_step, N, K * head_size]
        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])      # 内部就是矩阵乘法
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])  # 内部就是矩阵乘法

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)  # 将hidden_dimension的维度self.headsize等分，然后concat在bs维度上
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        # query: [K * batch_size, N, 1, head_size]
        # key:   [K * batch_size, N, head_size, num_step]
        # value: [K * batch_size, N, num_step, head_size]
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        # normalize the attention scores
        attention = torch.softmax(attention, dim=-1)  # 对 patch+1 维度求softmax  把随机变量加入q，k，v有什么好处呢？？？

        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection     # 没太明白这两个模块的作用
        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.tanh(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, device,number_of_weights=4):
        super(WeightGenerator, self).__init__()
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to(device)
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 5 * 5)  # 5似乎是论文建议的，这里不做修改
            ])
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))  # self.momery  # bs,5 随机初始化而来   self.generator 内部就是几层fc
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B  # 返回随机初始化的可训练的两个输出，每个输出内包含两个变量


def get_multiply(list):
    init = 1
    for i in list:
        init *= i
    return init