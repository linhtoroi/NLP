from torch import nn
from transformers import BertModel

class EmbeddingBert(nn.Module):
    """
    Pre-trained BERT embeddings
    """
    def __init__(self, model, dropout=0.0, requires_grad=False):
        super().__init__()
        self.bert_embedding = BertModel.from_pretrained(model)    
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.requires_grad = requires_grad

    def forward(self, inputs, input_masks, token_type_ids=None, position_ids=None):
        # Embedding = bert pre-trained and get hidden state last layer, pooler_output
        last_hidden_states, pooler_output = (self.bert_embedding(
            inputs, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=(~input_masks)))
        return self.dropout(last_hidden_states), pooler_output

class EmbeddingOther(nn.Module):

    def __init__(self, vocab_size, vocab_dim, dropout, requires_grad, pretrained_vocab_vectors=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.embeddings = nn.Embedding(vocab_size, vocab_dim)
        self.embeddings.weight.requires_grad = requires_grad
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embeddings(x))
