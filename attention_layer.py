import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with multi-head attention."""
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.0):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)  # Linear transformation
        self.a = nn.Linear(2 * out_features, 1, bias=False)  # Attention coefficient
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, h, adj):
        """
        Forward pass for the attention layer.

        Args:
            h (torch.Tensor): Node features (N, in_features).
            adj (torch.Tensor): Adjacency matrix (N, N).

        Returns:
            torch.Tensor: Updated node features (N, out_features).
        """
        N = h.size(0)  # Number of nodes

        # Linear transformation and reshape for multi-head attention
        h_trans = self.W(h)  # Shape: (N, out_features * num_heads)
        h_trans = h_trans.view(N, self.num_heads, -1)  # Shape: (N, num_heads, out_features)

        # Compute attention scores
        scores = torch.zeros(N, self.num_heads, N, device=h.device)  # Initialize scores
        for i in range(self.num_heads):
            h_i = h_trans[:, i, :]  # Shape: (N, out_features)
            h_j = h_i.unsqueeze(1)  # Shape: (N, 1, out_features)
            h_i_repeated = h_i.unsqueeze(0).repeat(N, 1, 1)  # Shape: (N, N, out_features)
            attention_input = torch.cat([h_j.repeat(1, N, 1), h_i_repeated], dim=-1)
            scores[:, i, :] = self.leaky_relu(self.a(attention_input)).squeeze(-1)  # Shape: (N, N)

        # Mask scores with adjacency matrix
        scores = scores.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (N, num_heads, N)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        # Aggregate node features
        h_trans = h_trans.permute(1, 0, 2)  # Shape: (num_heads, N, out_features)
        attention_weights = attention_weights.permute(1, 0, 2)  # Shape: (num_heads, N, N)

        out = torch.matmul(attention_weights, h_trans)  # Shape: (num_heads, N, out_features)
        out = out.permute(1, 0, 2).contiguous().view(N, -1)  # Shape: (N, num_heads * out_features)

        return out

