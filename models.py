import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool
from torch_geometric.utils import dropout_adj
from util_functions import *
import pdb
import time


class GNN(torch.nn.Module):
    def __init__(self, dataset, circRNA_similarity, disease_similarity, gconv=GCNConv, latent_dim=[32, 32, 32, 1],
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.circRNA_similarity = circRNA_similarity
        self.disease_similarity = disease_similarity

        self.regression = regression
        self.adj_dropout = adj_dropout 
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        self.lin3 = Linear(self.circRNA_similarity.shape[0] + self.disease_similarity.shape[0], 1)
        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
 



class LGCDA(GNN):
    def __init__(self, dataset, circRNA_similarity, disease_similarity, gconv=RGCNConv, latent_dim=[32, 32, 32, 32],
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):
        super(LGCDA, self).__init__(
            dataset, circRNA_similarity, disease_similarity, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.circRNA_similarity = circRNA_similarity
        self.disease_similarity = disease_similarity


        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2*sum(latent_dim)+n_side_features, 128)


    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)

        concat_states = torch.cat(concat_states, 1)


        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))

        # circRNA
        circRNA_node = [u for uu in data.temp_u_nodes for u in uu]
        circsims_temp = []
        for i in range(len(circRNA_node)):
            circsim_temp = self.circRNA_similarity[circRNA_node[i]]
            circsims_temp.append(circsim_temp)
        circsims_arr = np.array(circsims_temp).reshape(len(circRNA_node), self.circRNA_similarity.shape[0])
        circsims_tensor = torch.from_numpy(circsims_arr)
        circsims_floattensor = torch.FloatTensor(circsims_tensor)

        # disease
        disease_node = [v for vv in data.temp_v_nodes for v in vv]
        diseasesims_temp = []
        for i in range(len(disease_node)):
            diseasesim_temp = self.disease_similarity[disease_node[i]]
            diseasesims_temp.append(diseasesim_temp)
        diseasesims_arr = np.array(diseasesims_temp).reshape(len(disease_node), self.disease_similarity.shape[0])
        diseasesims_tensor = torch.from_numpy(diseasesims_arr)
        diseasesims_floattensor = torch.FloatTensor(diseasesims_tensor)


        sim = torch.cat([circsims_floattensor, diseasesims_floattensor], 1)
        sim = self.lin3(sim)


        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            lam = 0.0001
            return x[:, 0] * self.multiply_by + lam*sim[:,0]
        else:
            return F.log_softmax(x, dim=-1)
