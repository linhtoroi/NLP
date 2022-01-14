import torch
from torch import nn
from src.model import Embedding
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils import ops

class Encoder(nn.Module):
    def __init__(self, in_vocab, input_dim, hidden_dim, num_layers, rnn_layer_dropout,feat_emb_dropout, res_dropout,
                 ff_dropouts, improvement=False):
        super().__init__()
        self.in_vocab = in_vocab
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.improvement = improvement
        if improvement:
            self.transformer_encoder = TransformerEncoder_FFN(input_dim, hidden_dim)
        else:
            self.bilstm_encoder = RNNEncoder(input_dim, hidden_dim, num_layers, rnn_layer_dropout)
        self.text_encoder = RNNEncoder(hidden_dim, hidden_dim, num_layers, rnn_layer_dropout)
        self.schema_encoder = SchemaEncoder(self.hidden_dim, self.hidden_dim, feat_emb_dropout, res_dropout, ff_dropouts)
    def forward(self, inputs_embedded, input_masks, text_masks, schema_masks, feature_ids,
                transformer_output_value_masks=None, return_separate_hiddens=False):
        # BiLSTM all sequence (question+schema)
        if self.improvement:
            encoder_base_hiddens = self.transformer_encoder(inputs_embedded, input_masks)
        else:
            encoder_base_hiddens, _ = self.bilstm_encoder(inputs_embedded, input_masks)
        text_start_offset = 1
        text_embedded = encoder_base_hiddens[:, text_start_offset:text_masks.size(1) + text_start_offset, :]
        # BiLSTM only question
        text_hiddens, hidden = self.text_encoder(text_embedded, text_masks)
        constant_hiddens = text_hiddens
        constant_hidden_masks = text_masks

        # get schema part of encoder_base_hidden => schema_hidden
        schema_hiddens, schema_hidden_masks = self.batch_binary_lookup_3D(
            encoder_base_hiddens, schema_masks, pad_value=0)

        # Schema encoder
        schema_hiddens = self.schema_encoder(schema_hiddens, feature_ids)

        # Merge text and schema encodings
        encoder_hiddens, encoder_hidden_masks = self.merge_padded_seq_3D(
            constant_hiddens, constant_hidden_masks, schema_hiddens, schema_hidden_masks)
        return encoder_hiddens, encoder_hidden_masks, constant_hidden_masks, schema_hidden_masks, hidden

    def merge_padded_seq_3D(self, hiddens1, masks1, hidden2, masks2):
        batch_size = len(hiddens1)
        seq_len1 = masks1.size(1) - masks1.sum(dim=1)
        seq_len2 = masks2.size(1) - masks2.sum(dim=1)
        merged_size = seq_len1 + seq_len2
        max_merged_size = int(merged_size.max())
        res1 = max_merged_size - hiddens1.size(1)
        merged_hiddens = torch.cat([hiddens1, self.zeros_var_cuda([batch_size, res1, hiddens1.size(2)])], dim=1)
        scatter_index2 = seq_len1.unsqueeze(1) + self.batch_arange_cuda(batch_size, hidden2.size(1))
        scatter_index_masks2 = (scatter_index2 < max_merged_size)
        scatter_index2 *= scatter_index_masks2.long()
        merged_hiddens.scatter_add_(index=scatter_index2.unsqueeze(2).expand_as(hidden2),
                                    src=hidden2 * scatter_index_masks2.unsqueeze(2).float(), dim=1)
        merged_hidden_masks = self.batch_arange_cuda(batch_size, max_merged_size) >= merged_size.unsqueeze(1)
        return merged_hiddens, merged_hidden_masks

    def batch_binary_lookup_3D(self, M, b_idx, pad_value):
        # Pad binary indices
        batch_size = M.size(0)
        hidden_dim = M.size(2)
        seq_len = b_idx.sum(1, keepdim=True)
        max_seq_len = int(seq_len.max())
        output_masks = self.batch_arange_cuda(batch_size, max_seq_len) >= seq_len
        pad_len = max_seq_len - seq_len
        max_pad_len = int(pad_len.max())
        M = torch.cat([M, self.fill_var_cuda([batch_size, max_pad_len, hidden_dim], pad_value, dtype=M.dtype)], dim=1)
        pad_b_idx = self.batch_arange_cuda(batch_size, max_pad_len) < pad_len
        b_idx = torch.cat([b_idx, pad_b_idx], dim=1)
        output = M[b_idx].view(batch_size, max_seq_len, hidden_dim)
        return output, output_masks

    def batch_arange_cuda(self, batch_size, x, dtype=torch.long):
        return self.zeros_var_cuda(batch_size, dtype=dtype).unsqueeze(1) + \
            self.arange_cuda(x, dtype=dtype).unsqueeze(0)
    
    def zeros_var_cuda(self, s, requires_grad=False, dtype=torch.float32):
        return torch.zeros(s, requires_grad=requires_grad, dtype=dtype).cuda()

    def arange_cuda(self, x, dtype=torch.long):
        return torch.arange(x, dtype=dtype).cuda()

    def fill_var_cuda(self, s, value, dtype=None, requires_grad=False):
        return torch.zeros(s, dtype=dtype, requires_grad=requires_grad).cuda() + value
    
    
class TransformerEncoder_FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TransformerEncoder_FFN, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.FFN = nn.Linear(input_dim, hidden_dim)
    def forward(self, input, mask):
        return self.FFN(self.transformer_encoder(input, src_key_padding_mask=torch.transpose(mask, 0, 1)))

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_layer_dropout, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.input_size = input_dim
        self.hidden_size = int(hidden_dim/2)
        self.bidirectional = True
        self.num_layers = num_layers
        self.num_directions = 2
        self.dropout = rnn_layer_dropout
        self.rnn = nn.LSTM(input_dim,
                           self.hidden_size,
                           num_layers=num_layers,
                           dropout=rnn_layer_dropout,
                           bidirectional=True,
                           batch_first=True)

    def forward(self, inputs, input_masks, hidden = None):
        if input_masks is None:
            input_masks = ops.int_zeros_var_cuda([inputs.size(0), inputs.size(1)])
        max_seq_len = input_masks.size(1)
        input_sizes = max_seq_len - torch.sum(input_masks, dim=1).long()
        sorted_input_sizes, sorted_indices = torch.sort(input_sizes.cpu(), descending=True)
        inputs = inputs[sorted_indices] if self.batch_first else inputs[:, sorted_indices]
        # [batch_size, seq_len, input_dim]
        packed_inputs = pack(inputs, sorted_input_sizes, batch_first=self.batch_first)

        outputs, (h, c) = self.rnn(packed_inputs, hidden)

        # Unpack
        outputs, _ = unpack(outputs, batch_first=self.batch_first)
        _, rev_indices = torch.sort(sorted_indices, descending=False)
        # [batch_size, seq_len, num_directions*hidden_dim]
        outputs = outputs[rev_indices] if self.batch_first else outputs[:, rev_indices]
        # [num_layers*num_directions, batch_size, hidden_size]
        h = h[:, rev_indices, :]
        assert(h.size(0) == self.num_layers * self.num_directions)
        assert(h.size(2) == self.rnn.hidden_size)
        # [num_layers*num_directions, batch_size, hidden_size]
        c = c[:, rev_indices, :]
        assert (c.size(0) == self.num_layers * self.num_directions)
        assert (c.size(2) == self.rnn.hidden_size)

        if self.bidirectional:
            h = self.pack_bidirectional_lstm_state(h, self.num_layers)
            c = self.pack_bidirectional_lstm_state(c, self.num_layers)
        return outputs, (h, c)

    def pack_bidirectional_lstm_state(self, state, num_layers):
        _, batch_size, hidden_dim = state.size()
        layers = state.view(num_layers, 2, batch_size, hidden_dim).transpose(1, 2).contiguous()
        state = layers.view(num_layers, batch_size, -1)
        return state

class SchemaEncoder(nn.Module):
    def __init__(self, hidden_dim, feat_dim, feat_emb_dropout, res_dropout, ff_dropouts):
        super().__init__()
        self.primary_key_embeddings = Embedding.EmbeddingOther(2, feat_dim, dropout=feat_emb_dropout, requires_grad=True)
        self.foreign_key_embeddings = Embedding.EmbeddingOther(2, feat_dim, dropout=feat_emb_dropout, requires_grad=True)
        self.field_type_embeddings = Embedding.EmbeddingOther(6, feat_dim, dropout=feat_emb_dropout, requires_grad=True)
        self.feature_fusion_layer = Feedforward(
                hidden_dim + 3 * feat_dim, hidden_dim, hidden_dim, dropouts=ff_dropouts)

    def forward(self, input_hiddens, feature_ids):
        table_masks, _ = feature_ids[3]
        field_masks = (1 - table_masks).unsqueeze(2).float()
        field_type_embeddings = self.field_type_embeddings(feature_ids[2][0]) * field_masks
        primary_key_embeddings = self.primary_key_embeddings(feature_ids[0][0]) * field_masks
        foreign_key_embeddings = self.foreign_key_embeddings(feature_ids[1][0]) * field_masks
        schema_hiddens = self.feature_fusion_layer(torch.cat([input_hiddens,
                                                                primary_key_embeddings,
                                                                foreign_key_embeddings,
                                                                field_type_embeddings], dim=2))
        return schema_hiddens
    
class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropouts, activation='relu', bias1=True, bias2=True):
        super().__init__()
        self.activation = activation 
        self.input_dropout = nn.Dropout(dropouts[0])
        self.hidden_dropout = nn.Dropout(dropouts[1])
        self.linear1 = Linear(input_dim, hidden_dim, bias=bias1)
        self.linear2 = Linear(hidden_dim, output_dim, bias=bias2)

    def forward(self, x):
        return self.linear2(
                 self.hidden_dropout(
                   getattr(torch, self.activation)(
                     self.linear1(
                        self.input_dropout(x)))))

class Linear(nn.Linear):
    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)