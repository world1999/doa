import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionModule(nn.Module):
    def __init__(self, cv, nhead, cqk):
        super(MultiheadAttentionModule, self).__init__()

        self.nhead = nhead
        self.cv = cv
        self.cqk = cqk

        # Linear layers for Q, K, V
        self.WQ = nn.Linear(cv, cqk)
        self.WK = nn.Linear(cv, cqk)
        self.WV = nn.Linear(cv, cv)

        # Multihead Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=cqk, num_heads=nhead)

    def forward(self, X_input):
        # Reshape the input to the appropriate size
        batch_size = X_input.size(1)  # Assuming input shape: (h1 * h2, batch_size, cv)

        # Compute Q, K, V
        Q = self.WQ(X_input)
        K = self.WK(X_input)
        V = self.WV(X_input)

        # Q = X_input
        # K = X_input
        # V = X_input
        # Perform multi-head attention
        attn_output, attn_weights = self.multihead_attn(Q, K, V,average_attn_weights=False)

        # Return the attention output
        return attn_output


# Define parameters
cv = 64  # Example channel size
nhead = 8  # Number of attention heads
cqk = 128  # Example dimension of query/key tensor

# Create random input tensor (h1 * h2, batch_size, cv)
X_input = torch.rand((32, 10, cv))  # Example shape (h1 * h2, batch_size, cv)

# Instantiate the model
model = MultiheadAttentionModule(cv=cv, nhead=nhead, cqk=cqk)

# Forward pass through the model
output = model(X_input)
print(output.shape)
