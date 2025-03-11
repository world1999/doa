import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAModule(nn.Module):
    """卷积+批归一化+激活模块"""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):#self.dropout = nn.Dropout(0.3)
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, c_v, C_QK, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.C_QK = C_QK
        self.c_v = c_v

        # 线性变换矩阵
        self.W_Q = torch.nn.Linear(c_v, C_QK )
        self.W_K = torch.nn.Linear(c_v, C_QK )
        self.W_V = torch.nn.Linear(c_v, c_v)

        # # 输出的线性层
        # self.W_out = torch.nn.Linear(C_V , c_v)

    def forward(self, X_input):
        # 输入维度: [batch_size, h1, h2, c_v]
        batch_size, h1, h2, c_v = X_input.size()

        # 通过线性变换得到 Q, K, V
        Q = self.W_Q(X_input)  # [batch_size, h1, h2, C_QK ]
        K = self.W_K(X_input)  # [batch_size, h1, h2, C_QK]
        V = self.W_V(X_input)  # [batch_size, h1, h2, c_v ]

        # 将 Q, K, V 展开为多头的形式
        Q = Q.view(batch_size, h1 * h2, self.n_head, self.C_QK//self.n_head)  # [batch_size, h1*h2, n_head, C_QK/self.n_head]
        K = K.view(batch_size, h1 * h2, self.n_head, self.C_QK//self.n_head)  # [batch_size, h1*h2, n_head, C_QK/self.n_head]
        V = V.view(batch_size, h1 * h2, self.n_head, self.c_v//self.n_head)  # [batch_size, h1*h2, n_head, C_V/self.n_head]

        # 转置 Q 和 K，准备进行矩阵相乘
        Q = Q.transpose(1, 2)  # [batch_size, n_head, h1*h2, C_QK/self.n_head]
        K = K.transpose(1, 2)  # [batch_size, n_head, h1*h2, C_QK/self.n_head]
        V = V.transpose(1, 2)  # [batch_size, n_head, h1*h2, C_V/self.n_head]

        # 计算 Q 和 K 的点积并进行缩放
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, n_head, h1*h2, h1*h2]
        scores = scores /(self.C_QK ** 0.5)  # 缩放

        # 对 scores 进行 softmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, n_head, h1*h2, h1*h2]

        # 将注意力权重应用到 V 上
        output = torch.matmul(attention_weights, V)  # [batch_size, n_head, h1*h2, c_v/self.n_head]

        # 拼接所有头的输出
        output = output.transpose(1, 2)  # [batch_size, h1*h2, n_head, c_v/self.n_head]
        output = output.contiguous().view(batch_size, h1*h2,c_v)  # [batch_size, h1*h2, c_v]

        # # 最终输出通过线性变换
        # output = self.W_out(output)  # [batch_size, h1*h2, c_v]

        # 恢复输出形状 [batch_size, h1, h2, c_v]
        output = output.view(batch_size, h1, h2, c_v)

        return output


class LKR_MHSA(nn.Module):
    """局部核缩减多头自注意力模块"""

    def __init__(self, in_ch, num_heads=8,N=16):
        super().__init__()
        # 第一分支卷积 (空间维度缩减)
        self.conv_h = nn.Conv2d(in_ch, in_ch, kernel_size=(N//2, 1), stride=1, padding=0)
        # 第二分支卷积
        self.conv_w = nn.Conv2d(in_ch, in_ch, kernel_size=(1, N//2), stride=1, padding=0)
        # 第三分支MHSA
        self.mhsa =MultiHeadAttention(in_ch, in_ch, num_heads)
        # 合并后的处理
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B, C, H, W = x.shape  # 原始输入形状 (B,64,8,8)

        # 分支1：高度维度缩减
        branch_h = self.conv_h(x)  # (B,64,1,8)
        branch_h = branch_h.permute(0, 2, 3, 1)  # (B,1,8,64)

        # 分支2：宽度维度缩减
        branch_w = self.conv_w(x)  # (B,64,8,1)
        branch_w = branch_w.permute(0, 3, 2, 1)  # (B,1,8,64)

        # 分支3：MHSA处理
        x_mhsa = x.permute(0, 2, 3, 1)  # (B,8,8,64)
        attn_out = self.mhsa(x_mhsa)# (B,8,8,64)


        # 原始输入分支
        original = x.permute(0, 2, 3, 1)  # (B,8,8,64)

        # 动态维度拼接
        merged = torch.cat([
            branch_h,
            branch_w,
            attn_out,
            original
        ], dim=1)  # (B,18,8,64)

        # 转换为通道优先格式
        merged = merged.permute(0, 3, 1, 2)  # (B,64,18,8)
        merged = self.bn(merged)
        merged = self.act(merged)
        return self.dropout(merged)


class My_transform_Model(nn.Module):
    def __init__(self, num_classes=481,N=16):
        super().__init__()
        # 输入层 (自动处理通道顺序)

        self.num_classes = num_classes
        self.cba1 = CBAModule(4, 32, 3, 1, 1)
        self.conv1 = nn.Conv2d(32, 64, 2, stride=2, padding=0)
        # self.cba2 = CBAModule(32, 64, 2, 2, 0)
        self.lkr_mhsa = LKR_MHSA(64,8,N)#包含归一化、激活、dropout层
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.GELU()
        self.cba3 = CBAModule(64, 128, 2, 2, (1, 0))
        self.mhsa1 = MultiHeadAttention(128, 128, 8)# 不包含归一化、激活、dropout层
        self.cba4 = CBAModule(128, 256, 2, 2, (1, 0))
        self.mhsa2 = MultiHeadAttention(256, 256, 8)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(256*14*4, num_classes)
        # self.fc = nn.Linear(256 * 10* 4, num_classes)# kernel size 16*1
        self.fc = nn.Linear(256 * 6 * 2, num_classes)#8*1
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 输入形状处理 (batchsize,H,W,4) -> (batchsize,4,H,W)
        x = x.permute(0, 3, 1, 2).float()

        # 前向传播
        x = self.cba1(x)  # (B,32,16,16)
        x = self.dropout(x)
        x = self.conv1(x)  # (B,64,8,8)
        x = self.lkr_mhsa(x)  # (B,64,18,8) 已经包含归一化、激活、dropout层
        x = self.cba3(x)  # (B,128,10,4)
        x= x.permute(0, 2, 3, 1)
        x = self.mhsa1(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)
        x = self.cba4(x)  # (B,256,6,2)
        x = x.permute(0, 2, 3, 1)
        x = self.mhsa2(x)
        x = self.dropout(x)
        # x = x.permute(0, 3, 1, 2)

        # 输出处理
        x = self.flatten(x)  # (B,6*2*256)
        # 动态调整全连接层的输入尺寸
        if self.fc is None or self.fc.in_features != x.shape[1]:
            self.fc = nn.Linear(x.shape[1], self.num_classes).to(x.device)
        x = self.fc(x)
        return F.softmax(x, dim=1)


# 测试模型
if __name__ == "__main__":
    model = My_transform_Model()
    dummy_input = torch.randn(2, 16, 16, 4)  # (B,H,W,C)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
