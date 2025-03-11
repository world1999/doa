import torch
import torch.nn.functional as F


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
        scores = scores // (self.C_QK ** 0.5)  # 缩放

        # 对 scores 进行 softmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, n_head, h1*h2, h1*h2]

        # 将注意力权重应用到 V 上
        output = torch.matmul(attention_weights, V)  # [batch_size, n_head, h1*h2, c_v/self.n_head]

        # 拼接所有头的输出
        output = output.transpose(1, 2)  # [batch_size, h1*h2, n_head, c_v/self.n_head]
        output = output.contiguous().view(batch_size, h1 * h2,
                                          self.c_v)  # [batch_size, h1*h2, c_v]

        # # 最终输出通过线性变换
        # output = self.W_out(output)  # [batch_size, h1*h2, c_v]

        # 恢复输出形状 [batch_size, h1, h2, c_v]
        output = output.view(batch_size, h1, h2, self.c_v)

        return output


# 假设输入的维度是 [batch_size, h1, h2, c_v]
batch_size = 20
h1 = 4
h2 = 4
c_v = 16
C_QK = 8
n_head = 2

# 创建一个示例输入数据
X_input = torch.randn(batch_size, h1, h2, c_v)

# 创建多头注意力模型
attention = MultiHeadAttention(c_v, C_QK, n_head)

# 获取输出
output = attention(X_input)
print("Output shape:", output.shape)  # 应该是 [batch_size, h1, h2, c_v]
