import torch
import torch.nn as nn
import torch.nn.functional as F
import apex

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class DiffLoss(nn.Module):

    def __init__(self, args):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        # input1 (B,N,D)    input2 (B,N,D)

        batch_size = input1.size(0)
        N = input1.size(1)
        input1 = input1.view(batch_size, -1)  # (B,N*D)
        input2 = input2.view(batch_size, -1)  # (B, N*D)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True) # (1,N*D)
        input2_mean = torch.mean(input2, dim=0, keepdims=True) # (1,N*D)
        input1 = input1 - input1_mean     # (B,N*D)
        input2 = input2 - input2_mean     # (B,N*D)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach() # (B,1)
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6) # (B,N*D)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach() # (B,1)
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6) # (B,N*D)


        diff_loss = 1.0/(torch.mean(torch.norm(input1_l2-input2_l2,p=2,dim=1)))
  
        return diff_loss

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation = False, num_relation=-1,relation_coding='hard',relation_dim=50):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features # 输入特征维度
        self.out_features = out_features # 输出特征维度
        self.alpha = alpha
        self.concat = concat
        self.relation = relation

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if self.relation:
            if relation_coding=='hard':
                emb_matrix = torch.eye(num_relation)  # num_relation 只有relation=True时 有效
                self.relation_embedding = torch.nn.Embedding.from_pretrained(emb_matrix, freeze = True)  # 每种关系 用one-hot向量表示 且不训练
                self.a = nn.Parameter(torch.empty(size=(2*out_features + num_relation, 1)))
            elif relation_coding=='soft':
                self.relation_embedding = torch.nn.Embedding(num_relation,relation_dim)
                self.a = nn.Parameter(torch.empty(size=(2*out_features + relation_dim, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj):
        # h (B,N,D_in)
        Wh = torch.matmul(h, self.W) # (B, N, D_out)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (B, N, N, 2*D_out)

        if self.relation:
            long_adj = adj.clone().type(torch.LongTensor).cuda()
            relation_one_hot = self.relation_embedding(long_adj)  # 得到每个关系对应的one-hot 固定表示

            #print(relation_one_hot.shape)

            a_input = torch.cat([a_input, relation_one_hot], dim = -1)  # （B, N, N, 2*D_out+num_relation）

        #print(a_input.shape)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # (B, N , N)  所有部分都参与了计算 包括填充和没有关系连接的节点

        zero_vec = -9e15*torch.ones_like(e)  #计算mask 
        #print(adj.shape)
        #print(e.shape)
        # TODO: Solve empty graph issue here!
        attention = torch.where(adj > 0, e, zero_vec) # adj中非零位置 对应e的部分 保留，零位置(填充或没有关系连接)置为非常小的负数
        attention = F.softmax(attention, dim=2) # B, N, N
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (B,N,N_out)

        h_prime = self.layer_norm(h_prime)


        if self.concat:
            return F.gelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1] # N
        B = Wh.size()[0] # B
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        #print('Wh', Wh.shape)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)  # (B, N*N, 2*D_out)
        # all_combinations_matrix.shape == (B, N * N, 2 * out_features)

        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class RGAT(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout = 0.2, alpha = 0.2, nheads = 2, num_relation=-1):
        """Dense version of GAT."""
        super(RGAT, self).__init__()
        self.dropout = dropout
    
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, relation = True, num_relation=num_relation) for _ in range(nheads)] # 多头注意力
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False, relation = True, num_relation=num_relation) # 恢复到正常维度
        
        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1) # (B,N,num_head*N_out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.gelu(self.out_att(x, adj))  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x
