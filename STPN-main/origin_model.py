# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:13:39 2022

@author: AA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class nconv(nn.Module):
#归一化图卷积层
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
"""
输入 x 的形状：(N, C, V, L)，即：
N: Batch size
C: 特征通道数
V: 节点数量
L: 时间步或序列长度
邻接矩阵 A 的形状：(V, V)（或 (V, W)，但通常 W=V）。

实现了图结构上的特征传播：节点 v 的特征通过邻接矩阵 A 向其邻居传播。
输入 x 是 [batch, channel, node, time]，邻接矩阵 A 是 [node, node]。
使用 torch.einsum 实现批量矩阵乘法。
该层模拟图卷积中的邻居聚合步骤，利用归一化的邻接矩阵 A 对节点特征进行传播。
每个节点的特征被替换为其邻居特征的加权和，权重由 A 决定。
若 A 是标准化的（如 D⁻¹A 或对称归一化 D⁻½AD⁻½），则实现了归一化图卷积的核心操作。
"""

class linear(nn.Module):# 定义一个继承自nn.Module的类，用于线性变换
#线性变换层
#实例化的时候传入c_in和c_out，调用这个实例的时候传入x
    def __init__(self,c_in,c_out):# 构造函数，接收输入通道数c_in和输出通道数c_out
        super(linear,self).__init__()# 调用父类nn.Module的构造函数，正确初始化模块
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        # 定义一个1x1的卷积层，通过卷积核大小为1实现通道间的线性变换
        # c_in: 输入特征图的通道数
        # c_out: 输出特征图的通道数
        # kernel_size=(1,1): 卷积核不进行空间维度的信息混合，仅调整通道
        # padding=(0,0): 不填充输入特征图，保持空间尺寸不变
        # stride=(1,1): 步幅为1，确保输出空间尺寸与输入一致
        # bias=True: 启用偏置参数，增加模型的表达能力
    def forward(self,x):# 定义前向传播过程
        return self.mlp(x) # 将输入x通过1x1卷积层进行线性变换，输出结果

"""作用是维度变换（特征压缩/扩展），但保持空间结构。"""

class gcn(nn.Module):
#图卷积网络层
    def __init__(self,c_in,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.order = order

    def forward(self,x,support):
        #通过邻接矩阵传播信息
        out = [x]# 保存原始特征
        for a in support:# 遍历每个邻接矩阵 一阶传播（直接邻居）
            x1 = self.nconv(x,a)#图卷积操作
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)# 把自己和邻居的特征拼接
        return h
"""
假设邻接矩阵为 A，输入特征为 X：
一阶：X₁ = A X
二阶：X₂ = A X₁ = A² X
k阶：Xₖ = Aᵏ X
通过拼接 [X, X₁, X₂]，模型可以同时捕获局部（直接邻居）和全局（高阶邻居）的结构信息。
out = [
    x,                # 原始输入 (C通道)
    A_1 * x,          # 邻接矩阵 A_1 的一阶传播 (C通道)
    A_1^2 * x,        # 邻接矩阵 A_1 的二阶传播 (C通道)
    A_2 * x,          # 邻接矩阵 A_2 的一阶传播 (C通道)
    A_2^2 * x,        # 邻接矩阵 A_2 的二阶传播 (C通道)
    A_3 * x,          # 邻接矩阵 A_3 的一阶传播 (C通道)
    A_3^2 * x         # 邻接矩阵 A_3 的二阶传播 (C通道)
]
输出形状: (N, C × (1 + support_len × order), V, T)

示例：C=64, support_len=3, order=2 → 通道数=64×7=448

最终形状: (N, 448, V, T)
"""

class learnEmbedding(nn.Module):
#位置编码层
    def __init__(self, d_model):
        super(learnEmbedding, self).__init__()
        self.factor = nn.parameter.Parameter(torch.randn(1,),requires_grad=True)
        self.d_model = d_model

    def forward(self, x):
        """其实就是前一半是sin函数的x和div_term的乘积，后面一般是cos函数的x和div_term的乘积，最后拼接在一块长度一共为d_model"""
        div = torch.arange(0, self.d_model, 2)#生成一个从0开始，步长为2的等差数列，直到d_model(不包含d_model)
        div_term = torch.exp(div * self.factor)#div 中的每个元素对应一个频率分量的基础频率。
        if len(x.shape) == 2:
            v1 = torch.sin(torch.einsum('bt, f->btf', x, div_term))
            v2 = torch.cos(torch.einsum('bt, f->btf', x, div_term))
            """
            bt：输入 x 的维度标签（Batch, Time）。
            f：频率分量 div_term 的维度标签。
            btf：输出相位矩阵的维度标签（Batch, Time, Frequency）。
            x 的形状为 (B, T)，div_term 的形状为 (d_model//2,)。
            einsum 将 x 的每个时间步与 div_term 的每个频率分量相乘，生成形状为 (B, T, d_model//2) 的相位矩阵。
            sin 和 cos 生成各占一半维度的编码，最终拼接为 (B, T, d_model)。
            sin_enc = [sin(0.5), sin(0.6107), sin(0.7459)] ≈ [0.4794, 0.5737, 0.6784]
            cos_enc = [cos(0.5), cos(0.6107), cos(0.7459)] ≈ [0.8776, 0.8193, 0.7374]
            final_encoding = [0.4794, 0.5737, 0.6784, 0.8776, 0.8193, 0.7374]  # shape (6,)
            """
        else:
            v1 = torch.sin(torch.einsum('bvz, f->bvzf', x, div_term))
            v2 = torch.cos(torch.einsum('bvz, f->bvzf', x, div_term))
            """
            if 分支：处理二维输入（如时间序列），生成形状为 (B, T, d_model) 的编码。
            else 分支：处理高维输入（如图结构或视频数据），生成形状为 (B, V, Z, T, d_model) 的编码。
            设计意图：通过爱因斯坦求和和条件判断，确保位置编码层能灵活适配不同结构的输入数据。
            """
        return torch.cat([v1, v2], -1)

""""
d_model 是嵌入向量的维度，决定了模型对输入特征的表达能力。
在位置编码中，它控制编码后的向量长度，需设为偶数以确保正弦和余弦分量均分维度。
合理选择 d_model 是平衡模型性能和计算效率的关键。
"""
class ATT(nn.Module):
#自注意力层
    def __init__(self,c_in, d = 16, device = 'cuda'):
        super(ATT,self).__init__()
        self.d = d
        self.qm = nn.Linear(in_features = c_in, out_features = d, bias = False)
        self.km = nn.Linear(in_features = c_in, out_features = d, bias = False)
        self.device = device

    def forward(self,x, y):
        if len(x.shape) == 3:
            query = self.qm(y)
            key = self.km(x)
            attention = torch.einsum('btf,bpf->btp', query, key)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        else:
            query = self.qm(y)
            key = self.km(x)
            attention = torch.einsum('bvzf,buzf->bvu', query, key)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=2)
        return attention

class STMH_GCNN_layer(nn.Module):
#结合图卷积神经网络和多头注意力
#“我请了多个小专家，分别研究过去时间的哪些点最重要；
# 然后结合机场之间的关系，来做更聪明的时空预测。”
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, V, T_in)` format
        - Input[1]: Input random walk matrix in a list :math:`(V, V)` format
        - INput[2]: Input time label :math:`(N, T)`
        - Output[0]: Output graph sequence in :math:`(N, out_channels, V, T_out)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self, in_channels, out_channels, emb_size, dropout, time_d = 16, heads = 4, support_len = 3, order = 2, final_layer = False):
        """
        emb_size:时间嵌入的维度，dropout：Dropout的比例，time_d：时间步长（可能是时间位置嵌入长度）
        heads：多头注意力机制的头数，support_len：支持的图邻接矩阵个数（多个图结构）
        order：GCN 的传播阶数（1阶表示 A，2阶表示 A²）
        """
        super(STMH_GCNN_layer,self).__init__()
        self.gcn = gcn(in_channels,support_len=support_len,order=order)
        gc_in = (order*support_len+1)*in_channels
        #图卷积后通道数量 = 所有阶数聚合的拼接结果。

        self.out = linear(gc_in, out_channels)
        #self.out：一个 1×1 卷积（相当于线性层）做通道压缩/映射

        self.temb = nn.ModuleList()# 时间嵌入模块列表

        self.tgraph = nn.ModuleList()# 注意力模块列表
        for i in range(heads):
            self.temb.append(learnEmbedding(emb_size))
            self.tgraph.append(ATT(emb_size, time_d))
            """
            temb[i]：第 i 个注意力头的时间嵌入模块。
            tgraph[i]：第 i 个注意力头的注意力模块。
            """
        if in_channels != out_channels:
            self.residual=nn.Sequential(linear(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual=nn.Identity()
            """
            如果输入通道数 ≠ 输出通道数，需对残差进行升/降维。
            否则直接使用恒等映射（Identity）。
            """
        self.prelu = nn.PReLU()
        #prelu：激活函数。
        self.final_layer = final_layer
        #final_layer：决定是否使用残差+激活结构。
        self.dropout = dropout
        #dropout：防止过拟合。
        self.heads = heads
        #heads：记录头数，用于后续计算。

    def forward(self, x, t_in, supports, t_out = None):
        """
        x：输入数据 [N, C, V, T_in],其中N是批次大小，C是通道数，V是节点数，T_in是输入时间步数。
        t_in：输入时间标签 [N, T_in]
        supports：邻接矩阵（支持多个）
        t_out：如果需要预测未来时间，则有输出时间标签 [N, T_out]（否则默认和输入时间一致）
        """
        t_att = []#存储每个头的注意力权重，形状需要确定。
        for i in range(self.heads):
            # 生成键（Key）和查询（Query）的时间嵌入
            k_emb = self.temb[i](t_in) # 形状: (N, T_in, emb_size)
            if t_out == None:
                q_emb = k_emb# 形状: (N, T_out, emb_size)
            else:
                q_emb = self.temb[i](t_out)# 形状: (N, T_out, emb_size)
            t_att.append(self.tgraph[i](k_emb, q_emb)) # 形状: (N, T_out, T_in)

        res=self.residual(x)## 形状: (N, C, V, T_in) → (N, out_channels, V, T_in)

        #初始头处理
        xt = torch.einsum('ncvt,npt->ncvp', (x, t_att[0]))# 形状: (N, C, V, T_out)
        """
        输入：x 形状：(N, C, V, T_in)
        t_att[i] 形状：(N, T_out, T_in)
        公式：ncvt,npt→ncvp
        """
        #其余头相加
        for i in range(self.heads - 1):
            xt += torch.einsum('ncvt,npt->ncvp', (x, t_att[i+1]))

        x = self.gcn(xt, supports)
        # 输入形状: (N, C, V, T_out) → 输出形状: (N, gc_in, V, T_out)

        x = self.out(x)
        #通过 1x1 卷积 将通道数从 gc_in 映射到 out_channels。

        if not self.final_layer:
            x = x+res
            x = self.prelu(x)
        """
        如果 final_layer=True（最后一层），跳过残差连接，直接输出。
        否则，将GCN输出与残差路径相加，并通过激活函数增强非线性。
        """
        x = F.dropout(x, self.dropout, training=self.training)
        """
        作用：防止过拟合，仅在训练模式 (self.training=True) 时生效。
        """
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class STPN(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, V, T_in)` format
        - Input[1]: Input time label :math:`(N, T_in)` format
        - Input[2]: Input random walk matrix in a list :math:`(V, V)` format
        - Input[3]: Output time label :math:`(N, T_out)`
        - Input[4]: Input covariate sequence in :math:`(N, V, T_out)`
        - Output[0]: Output graph sequence in :math:`(N, out_channels, V, T_out)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            :out_channels= dimension of coordinates
            +
    """
    def __init__(self, h_layers, in_channels, hidden_channels, out_channels, emb_size, dropout, wemb_size = 4, time_d = 4, heads = 4, support_len = 3, order = 2, num_weather = 8, use_se = True, use_cov = True):
        super(STPN,self).__init__()
        """
        h_layers:隐藏层数
        in_channels：输入特征通道数
        hidden_channels:各隐藏层的通道数列表 比如为[64,64,128]
        out_channels:输出特征通道数
        emb_size:时间嵌入维度
        dropout：Dropout比例
        wemb_size:协变量(如天气)嵌入维度
        time_d：时间注意力维度
        """
        """
        Params:
            - h_layers: number of hidden STS GCNs
            - in_channels: num = 2 for delay prediction
            - hidden_channels: a list of dimensions for hidden features
            - out_channels: num = 2 for delay prediction
            - emb_size: embedding size for self attention model
            - dropout: dropout rate
            - wemb_size: covariate embedding size
            - time_d: d for self attention model
            - heads: number of attention head
            - supports_len: number of spatial adjacency matrix
            - order: order of diffusion convolution
            - num_weather: number of weather condition
            - use_se: whether use SE block
            - use_cov: whether use weather information
        """

        self.h_layers = h_layers
        self.convs = nn.ModuleList() # 存储所有时空层
        self.se = nn.ModuleList() # SE模块列表
        self.use_se = use_se
        self.use_cov = use_cov

        # 输入层：处理初始输入（可能融合协变量）
        if self.use_cov:
            #根据是否使用协变量（如天气数据），决定输入层的结构。
            self.convs.append(STMH_GCNN_layer(in_channels+ wemb_size, hidden_channels[0], emb_size, dropout, time_d, heads, support_len, order, False))
            #2. **当`self.use_cov`为True时**：
            # - 创建一个新的`STMH_GCNN_layer`，其输入通道数是`in_channels + wemb_size`，因为需要将原始输入和协变量嵌入后的数据拼接。
            self.w_fc = nn.Linear(14, wemb_size)  # 14维输入映射到嵌入空间
            """
            通道数变化：
            原始输入通道 (in_channels) + 天气嵌入维度 (wemb_size) → 隐藏层首通道数 (hidden_channels[0])
            例如：in_channels=2, wemb_size=4 → 融合后输入通道为6
            """
        else:
            self.convs.append(STMH_GCNN_layer(in_channels, hidden_channels[0], emb_size, dropout, time_d, heads, support_len, order, False))

        for i in range(h_layers):
            if self.use_se:
                self.se.append(SELayer(hidden_channels[i]))
            self.convs.append(STMH_GCNN_layer(hidden_channels[i], hidden_channels[i+1], emb_size, dropout, time_d, heads, support_len, order, False))
        self.final_conv = STMH_GCNN_layer(hidden_channels[h_layers] , out_channels, emb_size, dropout, time_d, heads, support_len, order, True)

    def forward(self, x, t_in, supports, t_out, w_type):
        if self.use_cov:
            # 新代码
            w_vec = self.w_fc(w_type)  # 输入shape:(batch, N, T, 14),表示每个节点每个时间步的 14 个协变量特征
                                        #输出shape:(batch, N, T, wemb_size),
            w_vec = w_vec.permute(0, 3, 1, 2)  # 调整为 (batch, wemb_size, N, T)

            x = torch.cat([x, w_vec], 1)# 结果 shape: (batch, in_channels + wemb_size, N, T)

        for i in range(self.h_layers + 1):
            x = self.convs[i](x, t_in, supports)
            if i < self.h_layers and self.use_se:
                x = self.se[i](x)
        """
        逐层调用卷积层 STMH_GCNN_layer。
        如果当前层不是最后一层，并且启用了 SE 模块，就加入通道注意力。
        """
        out = self.final_conv(x, t_in, supports, t_out)
        return out