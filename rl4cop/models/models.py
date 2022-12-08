import math

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_scatter

import rl4cop.utils.utils as utils


class MHAEncoderLayer(torch.nn.Module):

    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()

        self.n_heads = n_heads

        self.Wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4), torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = torch.nn.BatchNorm1d(embedding_dim)
        self.norm2 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x, mask=None):
        q = utils.make_heads(self.Wq(x), self.n_heads)
        k = utils.make_heads(self.Wk(x), self.n_heads)
        v = utils.make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(
            utils.multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class MHAEncoder(torch.nn.Module):

    def __init__(self,
                 n_layers,
                 n_heads,
                 embedding_dim,
                 input_dim,
                 add_init_projection=True):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(
                input_dim, embedding_dim)
        self.attn_layers = torch.nn.ModuleList([
            MHAEncoderLayer(embedding_dim=embedding_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        if hasattr(self, 'init_projection_layer'):
            x = self.init_projection_layer(x)
        for layer in self.attn_layers:
            x = layer(x, mask)
        return x


class DGL_ResGatedGraphConv(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout=0.0,
                 batch_norm=True,
                 residual=True):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.B = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.C = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.D = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.E = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = torch.nn.BatchNorm1d(output_dim)
        self.bn_node_e = torch.nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (
            g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels)


class DGL_ResGatedGraphEncoder(torch.nn.Module):

    def __init__(
        self,
        embedding_dim,
        input_h_dim,
        input_e_dim,
        n_layers,
        n_neighbors,
        dropout=0.0,
        batch_norm=True,
        residual=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_neighbors = n_neighbors

        self.embedding_h = torch.nn.Linear(input_h_dim, embedding_dim)
        self.embedding_e = torch.nn.Linear(input_e_dim, embedding_dim)
        self.layers = torch.nn.ModuleList([
            DGL_ResGatedGraphConv(embedding_dim, embedding_dim, dropout,
                                  batch_norm, residual) for _ in range(n_layers)
        ])

    def forward(self, batch):
        batched_graphs = utils.make_graphs(batch, self.n_neighbors, 'dgl')

        h = self.embedding_h(batched_graphs.ndata['feat'].float())
        e = self.embedding_e(batched_graphs.edata['feat'].float())

        for conv in self.layers:
            h, e = conv(batched_graphs, h, e)

        return h.view(batched_graphs.batch_size, -1, self.embedding_dim)


class PYG_DeepGCNEncoder(torch.nn.Module):

    def __init__(
        self,
        embedding_dim,
        input_h_dim,
        input_e_dim,
        n_layers,
        n_neighbors,
        conv_type='gen_conv',
        n_heads=8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv_type = conv_type
        self.n_neighbors = n_neighbors

        self.node_encoder = torch.nn.Linear(input_h_dim, embedding_dim)
        self.edge_encoder = torch.nn.Linear(input_e_dim, embedding_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            if conv_type == 'agnn_conv':
                conv = pyg.nn.AGNNConv(embedding_dim, embedding_dim)
            elif conv_type == 'graph_conv':
                conv = pyg.nn.GCNConv(embedding_dim, embedding_dim)
            elif conv_type == 'gat_conv':
                conv = pyg.nn.GATConv(embedding_dim,
                                      embedding_dim // n_heads,
                                      heads=n_heads)
            # elif conv_type == 'resgatedgraph_conv':
            #     conv = pyg.nn.ResGatedGraphConv(embedding_dim, embedding_dim)
            elif conv_type == 'transformer_conv':
                conv = pyg.nn.TransformerConv(embedding_dim,
                                              embedding_dim // n_heads,
                                              heads=n_heads,
                                              edge_dim=embedding_dim)
            elif conv_type == 'gen_conv':
                conv = pyg.nn.GENConv(embedding_dim,
                                      embedding_dim,
                                      aggr='softmax',
                                      t=1.0,
                                      learn_t=True,
                                      num_layers=2,
                                      norm='layer')
            norm = torch.nn.BatchNorm1d(embedding_dim)
            act = torch.nn.ReLU(inplace=True)
            layer = pyg.nn.DeepGCNLayer(conv, norm, act, block='res+')
            self.layers.append(layer)

    def forward(self, batch):
        batched_graphs = utils.make_graphs(batch, self.n_neighbors, 'pyg')

        x = batched_graphs.x
        edge_index = batched_graphs.edge_index
        edge_attr = batched_graphs.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for layer in self.layers:
            if self.conv_type in ['transformer_conv', 'gen_conv']:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

        return x.view(batched_graphs.num_graphs, -1, self.embedding_dim)


class LoopDecoder(torch.nn.Module):

    def __init__(self,
                 embedding_dim,
                 n_heads=8,
                 tanh_clipping=10.0,
                 n_decoding_neighbors=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.n_decoding_neighbors = n_decoding_neighbors

        self.Wq_graph = torch.nn.Linear(embedding_dim,
                                        embedding_dim,
                                        bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim,
                                        embedding_dim,
                                        bias=False)
        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, coordinates, embeddings, group_ninf_mask):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        # q_graph.hape = [B, n_heads, 1, key_dim]
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)
        self.q_graph = utils.make_heads(self.Wq_graph(graph_embedding),
                                        self.n_heads)
        self.q_first = None
        self.glimpse_k = utils.make_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = utils.make_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node):
        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)

        # q_graph.shape = [B, n_heads, 1, key_dim]
        # q_first.shape = q_last.shape = [B, n_heads, G, key_dim]
        if self.q_first is None:
            self.q_first = utils.make_heads(self.Wq_first(last_node_embedding),
                                            self.n_heads)

        q_last = utils.make_heads(self.Wq_last(last_node_embedding),
                                  self.n_heads)
        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_graph + self.q_first + q_last

        if self.n_decoding_neighbors is not None:
            D = self.coordinates.size(-1)
            K = torch.count_nonzero(self.group_ninf_mask[0, 0] == 0.0).item()
            K = min(self.n_decoding_neighbors, K)
            last_node_coordinate = self.coordinates.gather(
                dim=1, index=last_node.unsqueeze(-1).expand(B, G, D))
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            distances[self.group_ninf_mask == -np.inf] = np.inf
            indices = distances.topk(k=K, dim=-1, largest=False).indices
            glimpse_mask = torch.ones_like(self.group_ninf_mask) * (-np.inf)
            glimpse_mask.scatter_(dim=-1,
                                  index=indices,
                                  src=torch.zeros_like(glimpse_mask))
        else:
            glimpse_mask = self.group_ninf_mask
        attn_out = utils.multi_head_attention(q=glimpse_q,
                                              k=self.glimpse_k,
                                              v=self.glimpse_v,
                                              mask=glimpse_mask)

        # mha_out.shape = [B, G, H]
        # score.shape = [B, G, N]
        final_q = self.multi_head_combine(attn_out)
        score = torch.matmul(final_q, self.logit_k) / math.sqrt(H)
        score_clipped = self.tanh_clipping * torch.tanh(score)
        score_masked = score_clipped + self.group_ninf_mask

        probs = F.softmax(score_masked, dim=2)
        assert (probs == probs).all(), "Probs should not contain any nans!"
        return probs


class PathDecoder(torch.nn.Module):

    def __init__(self,
                 embedding_dim,
                 n_heads=8,
                 tanh_clipping=10.0,
                 n_decoding_neighbors=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.n_decoding_neighbors = n_decoding_neighbors

        self.Wq_graph = torch.nn.Linear(embedding_dim,
                                        embedding_dim,
                                        bias=False)
        self.Wq_source = torch.nn.Linear(embedding_dim,
                                         embedding_dim,
                                         bias=False)
        self.Wq_target = torch.nn.Linear(embedding_dim,
                                         embedding_dim,
                                         bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim,
                                         embedding_dim,
                                         bias=False)
        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_source = None  # saved q2, for multi-head attention
        self.q_target = None  # saved q3, for multi-head attention
        self.q_first = None  # saved q4, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, coordinates, embeddings, group_ninf_mask, source_node,
              target_node, first_node):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        B, N, H = embeddings.shape
        G = group_ninf_mask.size(1)
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)
        # q_graph.hape = [B, n_heads, 1, key_dim]
        self.q_graph = utils.make_heads(self.Wq_graph(graph_embedding),
                                        self.n_heads)
        # q_source.hape = [B, n_heads, 1, key_dim] - [B, n_heads, G, key_dim]
        source_node_index = source_node.view(B, G, 1).expand(B, G, H)
        source_node_embedding = self.embeddings.gather(1, source_node_index)
        self.q_source = utils.make_heads(self.Wq_source(source_node_embedding),
                                         self.n_heads)
        # q_target.hape = [B, n_heads, 1, key_dim]
        target_node_index = target_node.view(B, G, 1).expand(B, G, H)
        target_node_embedding = self.embeddings.gather(1, target_node_index)
        self.q_target = utils.make_heads(self.Wq_target(target_node_embedding),
                                         self.n_heads)
        # q_first.hape = [B, n_heads, 1, key_dim]
        first_node_index = first_node.view(B, G, 1).expand(B, G, H)
        first_node_embedding = self.embeddings.gather(1, first_node_index)
        self.q_first = utils.make_heads(self.Wq_first(first_node_embedding),
                                         self.n_heads)
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.glimpse_k = utils.make_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = utils.make_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node):
        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        # q_last.shape = q_last.shape = [B, n_heads, G, key_dim]
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = utils.make_heads(self.Wq_last(last_node_embedding),
                                  self.n_heads)
        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_graph + self.q_source + self.q_target + self.q_first + q_last

        if self.n_decoding_neighbors is not None:
            D = self.coordinates.size(-1)
            K = torch.count_nonzero(self.group_ninf_mask[0, 0] == 0.0).item()
            K = min(self.n_decoding_neighbors, K)
            last_node_coordinate = self.coordinates.gather(
                dim=1, index=last_node.unsqueeze(-1).expand(B, G, D))
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            distances[self.group_ninf_mask == -np.inf] = np.inf
            indices = distances.topk(k=K, dim=-1, largest=False).indices
            glimpse_mask = torch.ones_like(self.group_ninf_mask) * (-np.inf)
            glimpse_mask.scatter_(dim=-1,
                                  index=indices,
                                  src=torch.zeros_like(glimpse_mask))
        else:
            glimpse_mask = self.group_ninf_mask
        attn_out = utils.multi_head_attention(q=glimpse_q,
                                              k=self.glimpse_k,
                                              v=self.glimpse_v,
                                              mask=glimpse_mask)

        # mha_out.shape = [B, G, H]
        # score.shape = [B, G, N]
        final_q = self.multi_head_combine(attn_out)
        score = torch.matmul(final_q, self.logit_k) / math.sqrt(H)
        score_clipped = self.tanh_clipping * torch.tanh(score)
        score_masked = score_clipped + self.group_ninf_mask

        probs = F.softmax(score_masked.float(), dim=2, dtype=torch.float32).type_as(score_masked)
        assert (probs == probs).all(), "Probs should not contain any nans!"
        return probs


class AddPathDecoder(torch.nn.Module):

    def __init__(self,
                 embedding_dim,
                 n_heads=8,
                 tanh_clipping=10.0,
                 n_decoding_neighbors=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = int(embedding_dim /n_heads)
        self.tanh_clipping = tanh_clipping
        self.n_decoding_neighbors = n_decoding_neighbors

        self.Wq_graph = torch.nn.Linear(embedding_dim,
                                        embedding_dim,
                                        bias=False)
        self.Wq_source = torch.nn.Linear(embedding_dim,
                                         embedding_dim,
                                         bias=False)
        self.Wq_target = torch.nn.Linear(embedding_dim,
                                         embedding_dim,
                                         bias=False)
        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.query_att = torch.nn.Linear(self.embedding_dim, self.n_heads)
        self.key_att = torch.nn.Linear(self.embedding_dim, self.n_heads)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_source = None  # saved q2, for multi-head attention
        self.q_target = None  # saved q3, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, coordinates, embeddings, group_ninf_mask, source_node,
              target_node):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        B, N, H = embeddings.shape
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)
        # q_graph.hape = [B, n_heads, 1, key_dim]
        self.q_graph = utils.make_heads(self.Wq_graph(graph_embedding),
                                        self.n_heads)
        # q_source.hape = [B, n_heads, 1, key_dim]
        source_node_index = source_node.view(B, 1, 1).expand(B, 1, H)
        source_node_embedding = self.embeddings.gather(1, source_node_index)
        self.q_source = utils.make_heads(self.Wq_source(source_node_embedding),
                                         self.n_heads)
        # q_target.hape = [B, n_heads, 1, key_dim]
        target_node_index = target_node.view(B, 1, 1).expand(B, 1, H)
        target_node_embedding = self.embeddings.gather(1, target_node_index)
        self.q_target = utils.make_heads(self.Wq_target(target_node_embedding),
                                         self.n_heads)
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.glimpse_k = utils.make_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = utils.make_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node):
        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        # q_last.shape = q_last.shape = [B, n_heads, G, key_dim]
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = utils.make_heads(self.Wq_last(last_node_embedding),
                                  self.n_heads)
        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_graph + self.q_source + self.q_target + q_last

        if self.n_decoding_neighbors is not None:
            raise NotImplementedError
        else:
            glimpse_mask = self.group_ninf_mask
        
        # attn_out = utils.multi_head_attention(q=glimpse_q.float(),
        #                                       k=self.glimpse_k.float(),
        #                                       v=self.glimpse_v.float(),
        #                                       mask=glimpse_mask).type_as(glimpse_q)
        
        # fastformer additive attention
        # query_for_score.shape = batch_size, num_head, seq_len
    #     query_for_score = self.query_att(glimpse_q).transpose(1, 2) / self.head_dim**0.5
    #     if not glimpse_mask is None:
    #         query_for_score += glimpse_mask[:, None, :, :].expand_as(query_for_score)

    #     # mha_out.shape = [B, G, H]
    #     # score.shape = [B, G, N]
    #     final_q = self.multi_head_combine(attn_out)
    #     score = torch.matmul(final_q.float(), self.logit_k.float()).type_as(final_q) / math.sqrt(H)
    #     score_clipped = self.tanh_clipping * torch.tanh(score)
    #     score_masked = score_clipped + self.group_ninf_mask

    #     probs = F.softmax(score_masked, dim=2)
    #     assert (probs == probs).all(), "Probs should not contain any nans!"
    #     return probs


    # def forward(self, hidden_states, attention_mask):
    #     # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
    #     batch_size, seq_len, _ = hidden_states.shape
    #     mixed_query_layer = self.query(hidden_states)
        
    #     # batch_size, num_head, seq_len
    #     query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
    #     # add attention mask
    #     if not attention_mask is None:
    #         query_for_score += attention_mask

    #     # batch_size, num_head, 1, seq_len
    #     query_weight = self.softmax(query_for_score).unsqueeze(2)
        
    #     # batch_size, num_head, seq_len, head_dim
    #     query_layer = self.transpose_for_scores(mixed_query_layer)

    #     # batch_size, num_head, head_dim, 1
    #     pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
    #     pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
    #     # batch_size, num_head, seq_len, head_dim

    #     # batch_size, num_head, seq_len
    #     for i in range(self.num_layer):
    #         mixed_key_layer = self.key[i](hidden_states)
    #         mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
    #         query_key_score=(self.key_att[i](mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
    #         # add attention mask
    #         if not attention_mask is None:
    #             query_key_score +=attention_mask
    #         # batch_size, num_head, 1, seq_len
    #         query_key_weight = self.softmax(query_key_score).unsqueeze(2)
    #         key_layer = self.transpose_for_scores(mixed_query_key_layer)
    #         pooled_key = torch.matmul(query_key_weight, key_layer)
    #         pooled_query_repeat=pooled_key


    #     #query = value
    #     weighted_value =(pooled_key * query_layer).transpose(1, 2)
    #     weighted_value = weighted_value.reshape(
    #         weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
    #     weighted_value = self.transform(weighted_value) + mixed_query_layer
      
    #     return weighted_value