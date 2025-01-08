import torch
import torch.nn as nn
import torch.nn.functional as F
class LearnableDelta(nn.Module):
    def __init__(self, N, D):
        super(LearnableDelta, self).__init__()
        self.N = N
        self.D = D
        self.alpha = nn.Parameter(torch.ones(N, N, D+1) / (D+1))  

    def forward(self):

        alpha = F.softmax(self.alpha, dim=-1)  
        return alpha  


def pearson_correlation_batch(X_i, X_j):
    epsilon = 1e-8
    X_i_mean = X_i.mean(dim=2, keepdim=True)  
    X_j_mean = X_j.mean(dim=2, keepdim=True)  
    X_i_centered = X_i - X_i_mean  
    X_j_centered = X_j - X_j_mean  

    numerator = torch.matmul(X_i_centered, X_j_centered.transpose(1, 2))  
    denominator = torch.sqrt(
        torch.sum(X_i_centered ** 2, dim=2, keepdim=True) * torch.sum(X_j_centered ** 2, dim=2).unsqueeze(1)
    ) + epsilon  

    corr = numerator / denominator  
    return corr


def generate_soft_delay_adj_batch(X, delta_module):
    batch_size, T, N, F = X.shape
    alpha = delta_module()  
    D = delta_module.D
    corr_matrix = torch.zeros(batch_size, N, N, device=X.device)

    for d in range(D + 1):
        if d < T:
            if d == 0:
                X_i = X[:, :T - d]  
                X_j = X[:, :T - d]  
            else:
                X_i = X[:, :T - d]  
                X_j = X[:, d:]    

           
            X_i = X_i.reshape(-1, N, F)  
            X_j = X_j.reshape(-1, N, F)  


            corr = pearson_correlation_batch(X_i, X_j)  
            corr = corr.view(batch_size, T - d, N, N).mean(dim=1)  

            corr_matrix += alpha[:, :, d].unsqueeze(0) * corr  

    return corr_matrix  

class VariableLagAttention(nn.Module):
    def __init__(self, in_features, out_features, lag_window, dropout_rate=0.5):
        super(VariableLagAttention, self).__init__()
        self.lag_window = lag_window
        self.out_features = out_features
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.out_linear = nn.Linear(out_features * 2, out_features) 

    def forward(self, X):
        batch_size, T, N, F = X.shape
        device = X.device


        K_t = self.activation(self.W_k(X))  
        V_t = self.activation(self.W_v(X))  
        Q_t = self.activation(self.W_q(X))  

        output = torch.zeros(batch_size, T, N, self.out_features, device=device)

        for t in range(T):
            Q = Q_t[:, t]  
            K_list = []
            V_list = []
            for l in range(self.lag_window):
                t_lag = t - l
                if t_lag >= 0:
                    K_list.append(K_t[:, t_lag])  
                    V_list.append(V_t[:, t_lag])  
                else:
                    K_list.append(torch.zeros(batch_size, N, self.out_features, device=device))
                    V_list.append(torch.zeros(batch_size, N, self.out_features, device=device))

            K_lag = torch.stack(K_list, dim=1)  
            V_lag = torch.stack(V_list, dim=1) 
            K_lag = K_lag.view(batch_size, self.lag_window * N, self.out_features)  
            V_lag = V_lag.view(batch_size, self.lag_window * N, self.out_features)  
            attention_scores = torch.matmul(Q, K_lag.transpose(1, 2))  
            d_k = Q.shape[-1]
            attention_scores = attention_scores / (d_k ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  
            aggregated = torch.matmul(attention_weights, V_lag)  
            h_t = torch.cat([Q, aggregated], dim=-1)  
            h_t = self.out_linear(h_t)  
            h_t = self.activation(h_t)
            output[:, t] = self.dropout(h_t)  

        return output 


class MultiHeadVariableLagAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, lag_window, dropout_rate=0.5):
        super(MultiHeadVariableLagAttention, self).__init__()
        self.num_heads = num_heads
        self.lag_window = lag_window
        self.attention_heads = nn.ModuleList([
            VariableLagAttention(in_features, out_features, lag_window, dropout_rate) for _ in range(num_heads)
        ])

    def forward(self, X):
        outputs = []
        for attention in self.attention_heads:
            outputs.append(attention(X))  
        return torch.cat(outputs, dim=-1)  


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super(DynamicGraphConvolution, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()  

    def forward(self, X, adj):
        batch_size, T, N, F = X.shape
        h = self.activation(self.W(X))  
        h = torch.einsum('bmn,btno->btmo', adj, h)  
        return self.dropout(h)


class MultiHeadAttentionDynamicGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, lag_window, D, N, dropout_rate=0.5):
        super(MultiHeadAttentionDynamicGCN, self).__init__()
        self.attention = MultiHeadVariableLagAttention(
            in_features, hidden_features, num_heads, lag_window, dropout_rate)
        self.gcn = DynamicGraphConvolution(hidden_features * num_heads, out_features, dropout_rate)
        self.delta = LearnableDelta(N=N, D=D)  

    def forward(self, X):
        batch_size, T, N, F = X.shape
        attn_output = self.attention(X) 
        adj_matrix = generate_soft_delay_adj_batch(X, self.delta)  
        gcn_output = self.gcn(attn_output, adj_matrix) 
        return gcn_output  


class ClassificationModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, aggregated_features):
        return self.fc(aggregated_features)  


class FullModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads,
                 lag_window, D, N, num_classes, dropout_rate=0.5):
        super(FullModel, self).__init__()
        self.out_features = out_features
        self.mh_attention_gcn = MultiHeadAttentionDynamicGCN(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,  
            num_heads=num_heads,
            lag_window=lag_window,
            D=D,
            N=N,
            dropout_rate=dropout_rate
        )
        self.embedding_norm = nn.LayerNorm(out_features)
        self.embedding_fc = nn.Linear(out_features, out_features)  
        self.classifier = ClassificationModel(feature_dim=out_features, num_classes=num_classes)

    def forward(self, X, return_embedding=False):
        batch_size, T, N, F = X.shape
        node_features = self.mh_attention_gcn(X) 
        graph_features = node_features.mean(dim=2) 
        sample_representation = graph_features.mean(dim=1)  
        sample_representation = self.embedding_norm(sample_representation)
        sample_representation = self.embedding_fc(sample_representation)
        sample_representation = torch.relu(sample_representation)
        classification_output = self.classifier(sample_representation) 
        if return_embedding:
            return classification_output, sample_representation
        else:
            return classification_output








