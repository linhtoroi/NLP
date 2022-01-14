import torch
from torch import nn
from src.model.Encoder import Encoder
from src.model.Decoder import Decoder
from src.model.Embedding import EmbeddingBert
from src.model import Embedding
from src.utils.trans import bert_utils
from src.model.beam_search import beam_search
import os


class NL2SQL(nn.Module):
    def __init__(self, params, in_vocab, out_vocab):
        super().__init__()
        self.in_vocab = in_vocab
        self.schema_graphs = None

        # param:
        self.training = params.train
        self.inference = params.inference
        self.improvement = params.improvement

        # param tokenize, data:
        self.tu = bert_utils
        self.dataset_name = params.dataset_name
        self.model_id = params.model_id
        self.max_out_seq_len = params.max_out_seq_len


        # param encoder:
        self.pretrained_bert = params.pretrained_bert
        self.fix_pretrained_bert_parameters = params.fix_pretrained_bert_parameters
        self.pretrained_bert_embedding_dropout = params.pretrained_bert_embedding_dropout_rate
        self.encoder_embeddings = EmbeddingBert(self.pretrained_bert, dropout=self.pretrained_bert_embedding_dropout,
                                                         requires_grad=(not self.fix_pretrained_bert_parameters))

        
        self.encoder_input_dim = params.encoder_input_dim
        self.encoder_hidden_dim = params.encoder_hidden_dim
        self.num_rnn_layers = params.num_rnn_layers
        self.rnn_layer_dropout = params.rnn_layer_dropout_rate
        self.ff_dropouts = (params.ff_input_dropout_rate, params.ff_hidden_dropout_rate, params.ff_output_dropout_rate)
        self.feat_emb_dropout = params.feat_emb_dropout
        self.res_dropout = params.res_dropout
        self.encoder = Encoder(self.in_vocab,
                                self.encoder_input_dim,
                                self.encoder_hidden_dim,
                                self.num_rnn_layers,
                                self.rnn_layer_dropout,
                                self.feat_emb_dropout, 
                                self.res_dropout,
                                self.ff_dropouts,
                                self.improvement)
        # param decoder when training:
        self.share_vocab = params.share_vocab
        self.out_vocab = out_vocab
        self.output_vocab_size = self.out_vocab.size
        self.decoder_input_dim = params.decoder_input_dim
        self.decoder_hidden_dim = params.decoder_input_dim
        self.decoder_embedding_dropout = params.decoder_embedding_dropout_rate
        if self.share_vocab == True:
            self.decoder_embeddings = self.encoder_embeddings  
        else:
            self.decoder_embeddings = Embedding.EmbeddingOther(
                self.output_vocab_size, self.decoder_input_dim, dropout=self.decoder_embedding_dropout, requires_grad=True)
        
        self.rnn_weight_dropout = params.rnn_weight_dropout
        # self.num_const_attn_layers = params.num_const_attn_layers
        self.cross_attn_num_heads = params.cross_attn_num_heads
        self.attn_dropout = params.cross_attn_dropout_rate

        self.decoder = Decoder(self.decoder_input_dim,
                                     self.decoder_hidden_dim,
                                     self.num_rnn_layers,
                                     self.out_vocab,
                                     self.rnn_layer_dropout,
                                     self.rnn_weight_dropout,
                                     self.ff_dropouts,
                                     [(self.decoder_hidden_dim,
                                       self.encoder_hidden_dim,
                                       self.cross_attn_num_heads,
                                       self.attn_dropout)])

        # param decoder when inference:
        self.decoding_algorithm = params.decoding_algorithm
        self.beam_size = params.beam_size
        self.bs_alpha = params.bs_alpha

        # init
        self.xavier_initialization = params.xavier_initialization

    def forward(self, encoder_ptr_input_ids, encoder_ptr_value_ids, 
                text_masks, schema_masks,
                feature_ids,
                transformer_output_value_masks=None, schema_memory_masks=None,
                decoder_input_ids=None, decoder_ptr_value_ids=None):
        # Get input, input_mask and embedding
        inputs, input_masks = encoder_ptr_input_ids
        segment_ids, position_ids = self.get_segment_and_position_ids(inputs)
        inputs_embedded, _ = self.encoder_embeddings(inputs, input_masks, token_type_ids=segment_ids, position_ids=position_ids)

        # Encoder
        encoder_hiddens, encoder_hidden_masks, \
        constant_hidden_masks, schema_hidden_masks, hidden = self.encoder(inputs_embedded,
                                                                            input_masks,
                                                                            text_masks,
                                                                            schema_masks,
                                                                            feature_ids,
                                                                            transformer_output_value_masks)
        # Decoder
        if self.training:
            targets_embedded = self.decoder_embeddings(decoder_input_ids)
            outputs = self.decoder(targets_embedded,
                                   hidden,
                                   encoder_hiddens,
                                   encoder_hidden_masks,
                                   memory_masks=schema_memory_masks,
                                   encoder_ptr_value_ids=encoder_ptr_value_ids,
                                   decoder_ptr_value_ids=decoder_ptr_value_ids)
            return outputs[0]
        elif self.inference:
            with torch.no_grad():
                targets_embedded = self.decoder_embeddings(decoder_input_ids)
                outputs = self.decoder(targets_embedded,
                                    hidden,
                                    encoder_hiddens,
                                    encoder_hidden_masks,
                                    memory_masks=schema_memory_masks,
                                    encoder_ptr_value_ids=encoder_ptr_value_ids,
                                    decoder_ptr_value_ids=decoder_ptr_value_ids)
                return outputs[0]
        else:
            with torch.no_grad():
                if self.decoding_algorithm == 'beam-search':
                    table_masks, _ = feature_ids[3]
                    table_pos, _ = feature_ids[4]
                    if table_pos is not None:
                        table_field_scope, _ = feature_ids[5]
                        db_scope = (table_pos, table_field_scope)
                    else:
                        db_scope = None
                    return beam_search(self.bs_alpha,
                                       self.model_id,
                                       self.decoder,
                                       self.decoder_embeddings,
                                       self.max_out_seq_len,
                                       self.beam_size,
                                       hidden,
                                       encoder_hiddens=encoder_hiddens,
                                       encoder_masks=encoder_hidden_masks,
                                       constant_hidden_masks=constant_hidden_masks,
                                       schema_hidden_masks=schema_hidden_masks,
                                       table_masks=table_masks,
                                       encoder_ptr_value_ids=encoder_ptr_value_ids,
                                       schema_memory_masks=schema_memory_masks,
                                       db_scope=db_scope,
                                       no_from=(self.dataset_name == 'wikisql'))
                else:
                    raise NotImplementedError

    def get_segment_and_position_ids(self, encoder_input_ids):
        batch_size, input_size = encoder_input_ids.size()
        position_ids = torch.arange(input_size, dtype=torch.long).cuda().unsqueeze(0).expand_as(encoder_input_ids)
        seg1_end_pos = torch.nonzero(encoder_input_ids == self.tu.sep_id)[:, 1].view(batch_size, 2)[:, 0]
        segment_ids = (position_ids > seg1_end_pos.unsqueeze(1)).long()
        position_ids = None
        return segment_ids, position_ids

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file)
            self.load_state_dict(checkpoint['model_state_dict'])
            if self.training:
                self.start_step = checkpoint['interval_step_id'] + 1
                if 'optimizer_state_dict' in checkpoint:
                    self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'lr_scheduler_dict' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_dict'])
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))
