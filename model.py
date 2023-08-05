from model_utils import RGAT, DiffLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGATs(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        
        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        SpkGAT = []
        DisGAT = []
        for _ in range(args.gnn_layers):
            SpkGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=6))
            DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=18))

        self.SpkGAT = nn.ModuleList(SpkGAT)
        self.DisGAT = nn.ModuleList(DisGAT)


        self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        self.diff_loss = DiffLoss(args)
        self.beta = 0.3

        in_dim = args.hidden_dim *2 + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.drop = nn.Dropout(args.dropout)

       

    def forward(self, utterance_features, semantic_adj, structure_adj):
        '''
        :param tutterance_features: (B, N, emb_dim)
        :param xx_adj: (B, N, N)
        :return:
        '''
        batch_size = utterance_features.size(0)
        H0 = F.relu(self.fc1(utterance_features)) # (B, N, hidden_dim)
        H = [H0]
        diff_loss = 0
        for l in range(self.args.gnn_layers):
            if l==0:
                H1_semantic = self.SpkGAT[l](H[l], semantic_adj)
                H1_structure = self.DisGAT[l](H[l], structure_adj)
            else:
                H1_semantic = self.SpkGAT[l](H[2*l-1], semantic_adj)
                H1_structure = self.DisGAT[l](H[2*l], structure_adj)


            diff_loss = diff_loss + self.diff_loss(H1_semantic, H1_structure)
            # BiAffine 

            A1 = F.softmax(torch.bmm(torch.matmul(H1_semantic, self.affine1), torch.transpose(H1_structure, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(H1_structure, self.affine2), torch.transpose(H1_semantic, 1, 2)), dim=-1)

            H1_semantic_new = torch.bmm(A1, H1_structure)
            H1_structure_new = torch.bmm(A2, H1_semantic)

            H1_semantic_out = self.drop(H1_semantic_new) if l < self.args.gnn_layers - 1 else H1_semantic_new
            H1_structure_out = self.drop(H1_structure_new) if l <self.args.gnn_layers - 1 else H1_structure_new


            H.append(H1_semantic_out)
            H.append(H1_structure_out)

        H.append(utterance_features) 

        H = torch.cat([H[-3],H[-2],H[-1]], dim = 2) #(B, N, 2*hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        logits = self.out_mlp(H)
        return logits, self.beta * (diff_loss/self.args.gnn_layers)







