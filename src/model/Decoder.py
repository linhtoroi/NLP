import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.utils import ops

class Decoder(nn.Module):
    def __init__(self, decoder_input_dim, decoder_hidden_dim, num_rnn_layers, out_vocab,\
         rnn_layer_dropout, rnn_weight_dropout, ff_dropouts, attn_params, batch_first=True):
        super().__init__()

        self.vocab = out_vocab
        self.vocab_size = out_vocab.size

        self.query_dim = attn_params[0][0] # == decoder_hidden_dim
        self.context_dim = sum([v_dim for _, v_dim, _, _ in attn_params]) # == encoder_hidden_dim
        self.rnn = nn.LSTM(input_size=self.query_dim + self.context_dim,
                                     hidden_size=decoder_hidden_dim,
                                     bidirectional=False,
                                     num_layers=num_rnn_layers,
                                     dropout=rnn_layer_dropout,
                                     batch_first=batch_first)
        attn_query_dim, attn_value_dim, attn_num_heads, attn_dropout = attn_params[0]
        attn_ = AttentionDotProduct(dropout=attn_dropout, causal=False)
        self.attn = MultiHead(attn_query_dim,
                              attn_value_dim,
                              attn_value_dim,
                              attn_num_heads,
                              attn_)

        self.attn_combine = ConcatAndProject(decoder_hidden_dim + self.context_dim, decoder_hidden_dim, ff_dropouts[0], activation='tanh')

        self.out = LogSoftmaxOutput(decoder_hidden_dim, out_vocab.size)

        self.pointer_switch = PointerSwitch(query_dim=self.query_dim, key_dim=self.context_dim, input_dropout=ff_dropouts[0])



    def forward(self, input_embedded, hidden, encoder_hiddens, encoder_hidden_masks, pointer_context=None,
                vocab_masks=None, memory_masks=None, encoder_ptr_value_ids=None, decoder_ptr_value_ids=None,
                last_output=None):

        batch_size = len(input_embedded)

        if pointer_context:
            p_pointer, attn_weights = pointer_context
        else:
            p_pointer = ops.zeros_var_cuda([batch_size, 1, 1])
            attn_weights = self.zeros_var_cuda([batch_size, self.attn.num_heads, 1, encoder_hiddens.size(1)])
        
        outputs, hiddens = [], []
        seq_attn_weights = []
        seq_p_pointers = []

        for i in range(input_embedded.size(1)):
            input_ = input_embedded[:, i:i + 1, :]
            if self.training and decoder_ptr_value_ids is not None:
                last_output = decoder_ptr_value_ids[:, i:i + 1]
            else:
                assert(last_output is not None)
            # tính selective read
            select_attn = self.selective_read(encoder_ptr_value_ids, encoder_hiddens,
                                         attn_weights[:, -1, :, :], last_output)
            
            input_sa = torch.cat([input_, select_attn], dim=2)
            output, hidden = self.rnn(input_sa, hidden)
            hiddens.append(hidden)

            # multi_head_attention 
            attn_vec, attn_weights = self.attn(output, encoder_hiddens, encoder_hiddens, encoder_hidden_masks)
            p_pointer = self.pointer_switch(output, attn_vec)
            output = self.attn_combine(output, attn_vec)

            gen_logit = self.out(output)
            gen_prob = torch.exp(gen_logit)

            point_prob = attn_weights[:, -1, :, :]

            weighted_point_prob = p_pointer * point_prob

            gen_prob_zeros_pad = ops.zeros_var_cuda((batch_size, 1, encoder_hiddens.size(1)))
            weighted_gen_prob = torch.cat([(1 - p_pointer) * gen_prob, gen_prob_zeros_pad], dim=2)
            point_gen_prob = weighted_gen_prob.scatter_add_(index=encoder_ptr_value_ids.unsqueeze(1),
                                                            src=weighted_point_prob, dim=2)

            point_gen_logit = ops.safe_log(point_gen_prob)

            outputs.append(point_gen_logit), seq_attn_weights.append(attn_weights),\
            seq_p_pointers.append(p_pointer)

        if self.training:
            return torch.cat(outputs, dim=1), hidden, \
                   (torch.cat(seq_p_pointers, dim=1), torch.cat(seq_attn_weights, dim=2))
        else:
            return torch.cat(outputs, dim=1), hidden, \
                   (torch.cat(seq_p_pointers, dim=1), torch.cat(seq_attn_weights, dim=2))


    def zeros_var_cuda(self, s, requires_grad=False, dtype=torch.float32):
        return torch.zeros(s, requires_grad=requires_grad, dtype=dtype).cuda()
    
    def selective_read(self, encoder_ptr_value_ids, memory_hiddens, attn_weights, last_output):
        point_mask = (encoder_ptr_value_ids == last_output).float()
        weights = point_mask * attn_weights.squeeze(1)
        batch_size = memory_hiddens.size(0)
        weight_normalizer = weights.sum(dim=1)
        weight_normalizer = weight_normalizer + (weight_normalizer == 0).float() * float(np.finfo(float).eps)
        return (weights.unsqueeze(2) * memory_hiddens).sum(dim=1, keepdim=True) / weight_normalizer.view(batch_size, 1, 1)

class MultiHead(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, attention):
        super().__init__()
        self.attention = attention
        self.wq = nn.Linear(query_dim, key_dim, bias=False)
        self.wk = nn.Linear(key_dim, key_dim, bias=False)
        self.wv = nn.Linear(value_dim, value_dim, bias=False)
        self.num_heads = num_heads
        self.wo = nn.Linear(value_dim, key_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.chunk(self.num_heads, -1) for x in (query, key, value))
        multi_head_attn_vecs, multi_head_attn_weights = [], []
        for q, k, v in zip(query, key, value):
            head_attn_vec, head_attn_weights = self.attention(q, k, v, mask)
            multi_head_attn_vecs.append(head_attn_vec)
            multi_head_attn_weights.append(head_attn_weights.unsqueeze(1))
        multi_head_attn_weights = torch.cat(multi_head_attn_weights, dim=1)
        multi_head_attn = torch.cat(multi_head_attn_vecs, dim=2)
        return self.wo(multi_head_attn), multi_head_attn_weights

class AttentionDotProduct(nn.Module):

    def __init__(self, dropout, causal, return_attn_vec=True, return_normalized_weights=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.return_attn_vec = return_attn_vec
        self.return_normalized_weights = return_normalized_weights

    def forward(self, query, key, value, mask=None):
        # [batch_size, query_seq_len, key_seq_len]
        attn_weights = self.matmul(query, key.transpose(1, 2))
        if (query.size(1) == key.size(1)) and self.causal:
            causal_mask = self.fill_var_cuda((query.size(1), key.size(1)), 1).triu(1)
            attn_weights -= causal_mask.unsqueeze(0) * 1e31
        if mask is not None:
            attn_weights.data.masked_fill_(mask.unsqueeze(1).expand_as(attn_weights), -1e31)
        attn_weights /= np.sqrt(key.size(-1))
        if self.return_normalized_weights:
            attn_weights = F.softmax(attn_weights, -1)

        if self.return_attn_vec:
            assert(self.return_normalized_weights)
            # [batch_size, query_seq_len, value_dim]
            attn_vec = self.matmul(attn_weights, self.dropout(value))
            return attn_vec, attn_weights
        else:
            return attn_weights

    def matmul(self, x, y):
        if x.dim() == y.dim():
            return x @ y
        elif x.dim() == y.dim() - 1:
            return (x.unsqueeze(-2) @ y).squeeze_(-2)
        elif x.dim() - 1 == y.dim():
            return (x @ y.unsqueeze(-2)).squeeze_(-2)
        else:
            raise AttributeError('matmul: Unmatched input dimension x: {} y: {}'.format(x.size(), y.size()))

    def fill_var_cuda(self, s, value, dtype=None, requires_grad=False):
        return torch.zeros(s, dtype=dtype, requires_grad=requires_grad).cuda() + value

class PointerSwitch(nn.Module):

    def __init__(self, query_dim, key_dim, input_dropout):
        super().__init__()
        self.project = ConcatAndProject(query_dim + key_dim, 1, input_dropout, activation=None)

    def forward(self, query, key):
        return torch.sigmoid(self.project(query, key))

class ConcatAndProject(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation=None, bias=True):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.linear1 = Linear(input_dim, output_dim, bias=bias)
        self.activation = activation

    def forward(self, *args):
        input = self.input_dropout(torch.cat(args, dim=-1))
        if self.activation is None:
            return self.linear1(input)
        else:
            return getattr(torch, self.activation)(self.linear1(input))

class Linear(nn.Linear): 
    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)

class LogSoftmaxOutput(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = Linear(input_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.linear(x))