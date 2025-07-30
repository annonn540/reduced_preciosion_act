import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_class):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation_class()
        
    def forward(self, x, adj):
        h = self.linear(x)
        
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        output = torch.bmm(adj, h)
        
        return self.activation(output)

class FP32GCN(nn.Module):
    def __init__(self, input_features, hidden_dim=64, num_classes=10, activation="relu", num_nodes=None):
        super().__init__()
        
        activation_map = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "leaky_relu": lambda: nn.LeakyReLU(),
            "prelu": lambda: nn.PReLU(),
            "swish": nn.SiLU,
            "softplus": nn.Softplus
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        
        act = activation_map[activation]
        
        self.num_nodes = num_nodes
        if num_nodes is None:
            self.grid_size = 16
            self.num_nodes = self.grid_size * self.grid_size
            self.patch_size = 2 
        else:
            import math
            self.grid_size = int(math.sqrt(num_nodes))
            self.num_nodes = num_nodes
            self.patch_size = 32 // self.grid_size
        
        if isinstance(input_features, (list, tuple)) and len(input_features) == 3:
            channels = input_features[0]
            self.node_features = channels * self.patch_size * self.patch_size
        else:
            self.node_features = input_features // self.num_nodes
        
        self.gc1 = GraphConvLayer(self.node_features, hidden_dim, act)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim, act)
        self.gc3 = GraphConvLayer(hidden_dim, hidden_dim // 2, act)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.num_nodes * (hidden_dim // 2), num_classes)
        )
        
        self.register_buffer('adj_matrix', self._create_grid_adjacency())
        
    def _create_grid_adjacency(self):
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = i * self.grid_size + j
                
                adj[node_id, node_id] = 1.0
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                        neighbor_id = ni * self.grid_size + nj
                        adj[node_id, neighbor_id] = 1.0
        
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        norm_adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        return norm_adj
    
    def _image_to_graph(self, x):
        batch_size, channels, height, width = x.shape
        
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, self.grid_size, self.grid_size, 
                               self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, self.num_nodes, -1)
        
        return x
    
    def forward(self, x):
        if x.dim() == 4:
            x = self._image_to_graph(x)
        elif x.dim() == 2: 
            batch_size = x.size(0)
            x = x.view(batch_size, self.num_nodes, -1)
        
        x = self.gc1(x, self.adj_matrix)
        x = F.dropout(x, 0.3, training=self.training)
        
        x = self.gc2(x, self.adj_matrix)
        x = F.dropout(x, 0.3, training=self.training)
        
        x = self.gc3(x, self.adj_matrix)
        x = F.dropout(x, 0.5, training=self.training)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x