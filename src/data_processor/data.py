import src.utils.trans.bert_utils as bu
from src.data_processor.vocab_utils import Vocabulary
from src.data_processor.sql_reserved_tokens import sql_reserved_tokens

def load_vocabs(param):
    tu = bu
    text_vocab = Vocabulary(tag='text', func_token_index=None, tu=tu)
    for v in tu.tokenizer.vocab:
        text_vocab.index_token(v, in_vocab=True, check_for_seen_vocab=True)
    vocabs = {
        'text': text_vocab,
        'sql': sql_reserved_tokens
    }
    return vocabs