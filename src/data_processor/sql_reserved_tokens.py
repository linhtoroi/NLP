from src.data_processor.vocab_utils import SQLVocabulary, functional_token_index

digits = {
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12',
    '##0',
    '##1',
    '##2',
    '##3',
    '##4',
    '##5'
}

sql_reserved_tokens = SQLVocabulary('sql_reserved_tokens', functional_token_index)
sql_reserved_tokens.index_token('!')
sql_reserved_tokens.index_token('"')
sql_reserved_tokens.index_token('%')
sql_reserved_tokens.index_token('(')
sql_reserved_tokens.index_token(')')
sql_reserved_tokens.index_token('*')
sql_reserved_tokens.index_token('+')
sql_reserved_tokens.index_token(',')
sql_reserved_tokens.index_token('-')
sql_reserved_tokens.index_token('.')
sql_reserved_tokens.index_token('/')
sql_reserved_tokens.index_token('0')
sql_reserved_tokens.index_token('1')
sql_reserved_tokens.index_token('2')
sql_reserved_tokens.index_token('3')
sql_reserved_tokens.index_token('4')
sql_reserved_tokens.index_token('5')
sql_reserved_tokens.index_token('6')
sql_reserved_tokens.index_token('7')
sql_reserved_tokens.index_token('8')
sql_reserved_tokens.index_token('9')
sql_reserved_tokens.index_token('10')
sql_reserved_tokens.index_token('11')
sql_reserved_tokens.index_token('12')
sql_reserved_tokens.index_token('##0')
sql_reserved_tokens.index_token('##1')
sql_reserved_tokens.index_token('##2')
sql_reserved_tokens.index_token('##3')
sql_reserved_tokens.index_token('##4')
sql_reserved_tokens.index_token('##5')
# sql_reserved_tokens.index_token('##6')
# sql_reserved_tokens.index_token('##7')
# sql_reserved_tokens.index_token('##8')
# sql_reserved_tokens.index_token('##9')
sql_reserved_tokens.index_token(':')
sql_reserved_tokens.index_token('<')
sql_reserved_tokens.index_token('<=')
sql_reserved_tokens.index_token('!=')
sql_reserved_tokens.index_token('=')
sql_reserved_tokens.index_token('==')
sql_reserved_tokens.index_token('>')
sql_reserved_tokens.index_token('>=')
sql_reserved_tokens.index_token('alias')
sql_reserved_tokens.index_token('all')
sql_reserved_tokens.index_token('and')
sql_reserved_tokens.index_token('as')
sql_reserved_tokens.index_token('asc')
sql_reserved_tokens.index_token('avg')
sql_reserved_tokens.index_token('between')
sql_reserved_tokens.index_token('case')
sql_reserved_tokens.index_token('count')
sql_reserved_tokens.index_token('curdate')
sql_reserved_tokens.index_token('derived_field')
sql_reserved_tokens.index_token('derived_table')
sql_reserved_tokens.index_token('desc')
sql_reserved_tokens.index_token('distinct')
sql_reserved_tokens.index_token('else')
sql_reserved_tokens.index_token('end')
sql_reserved_tokens.index_token('exists')
sql_reserved_tokens.index_token('except')
sql_reserved_tokens.index_token('from')
sql_reserved_tokens.index_token('group by')
sql_reserved_tokens.index_token('having')
sql_reserved_tokens.index_token('in')
sql_reserved_tokens.index_token('inner')
sql_reserved_tokens.index_token('intersect')
sql_reserved_tokens.index_token('is')
sql_reserved_tokens.index_token('join')
sql_reserved_tokens.index_token('left')
sql_reserved_tokens.index_token('like')
sql_reserved_tokens.index_token('limit')
sql_reserved_tokens.index_token('lower')
sql_reserved_tokens.index_token('max')
sql_reserved_tokens.index_token('min')
sql_reserved_tokens.index_token('n')
sql_reserved_tokens.index_token('no')
sql_reserved_tokens.index_token('not')
sql_reserved_tokens.index_token('null')
sql_reserved_tokens.index_token('on')
sql_reserved_tokens.index_token('or')
sql_reserved_tokens.index_token('order by')
sql_reserved_tokens.index_token('outer')
sql_reserved_tokens.index_token('select')
sql_reserved_tokens.index_token('sum')
sql_reserved_tokens.index_token('t')
sql_reserved_tokens.index_token('then')
sql_reserved_tokens.index_token('union')
sql_reserved_tokens.index_token('when')
sql_reserved_tokens.index_token('where')
sql_reserved_tokens.index_token('y')
sql_reserved_tokens.index_token('year')
sql_reserved_tokens.index_token('yes')


sql_reserved_tokens_revtok = SQLVocabulary('sql_reserved_tokens_revtok', functional_token_index)
sql_reserved_tokens_revtok.index_token(' ! ')
sql_reserved_tokens_revtok.index_token(' "')
sql_reserved_tokens_revtok.index_token(' " ')
sql_reserved_tokens_revtok.index_token(' ( ')
sql_reserved_tokens_revtok.index_token(' ) ')
sql_reserved_tokens_revtok.index_token(' * ')
sql_reserved_tokens_revtok.index_token(' + ')
sql_reserved_tokens_revtok.index_token(' , ')
sql_reserved_tokens_revtok.index_token(' - ')
sql_reserved_tokens_revtok.index_token(' / ')
sql_reserved_tokens_revtok.index_token(' 0 ')
sql_reserved_tokens_revtok.index_token(' 1 ')
sql_reserved_tokens_revtok.index_token(' 2 ')
sql_reserved_tokens_revtok.index_token(' 3 ')
sql_reserved_tokens_revtok.index_token(' 4 ')
sql_reserved_tokens_revtok.index_token(' 5 ')
sql_reserved_tokens_revtok.index_token(' 6 ')
sql_reserved_tokens_revtok.index_token(' 7 ')
sql_reserved_tokens_revtok.index_token(' 8 ')
sql_reserved_tokens_revtok.index_token(' 9 ')
sql_reserved_tokens_revtok.index_token(' < ')
sql_reserved_tokens_revtok.index_token(' <= ')
sql_reserved_tokens_revtok.index_token(' != ')
sql_reserved_tokens_revtok.index_token(' = ')
sql_reserved_tokens_revtok.index_token(' > ')
sql_reserved_tokens_revtok.index_token(' >= ')
sql_reserved_tokens_revtok.index_token(' == ')
sql_reserved_tokens_revtok.index_token(' ALL ')
sql_reserved_tokens_revtok.index_token(' AND ')
sql_reserved_tokens_revtok.index_token(' AS ')
sql_reserved_tokens_revtok.index_token(' ASC ')
sql_reserved_tokens_revtok.index_token(' AVG ')
sql_reserved_tokens_revtok.index_token(' BETWEEN ')
sql_reserved_tokens_revtok.index_token(' BY ')
sql_reserved_tokens_revtok.index_token(' CASE ')
sql_reserved_tokens_revtok.index_token(' COUNT ')
sql_reserved_tokens_revtok.index_token(' CURDATE ')
sql_reserved_tokens_revtok.index_token(' DERIVED_FIELD ')
sql_reserved_tokens_revtok.index_token(' DERIVED_TABLE ')
sql_reserved_tokens_revtok.index_token(' DESC ')
sql_reserved_tokens_revtok.index_token(' DISTINCT ')
sql_reserved_tokens_revtok.index_token(' ELSE ')
sql_reserved_tokens_revtok.index_token(' END ')
sql_reserved_tokens_revtok.index_token(' EXISTS ')
sql_reserved_tokens_revtok.index_token(' EXCEPT ')
sql_reserved_tokens_revtok.index_token(' FROM ')
sql_reserved_tokens_revtok.index_token(' GROUP ')
sql_reserved_tokens_revtok.index_token(' HAVING ')
sql_reserved_tokens_revtok.index_token(' IN ')
sql_reserved_tokens_revtok.index_token(' INNER ')
sql_reserved_tokens_revtok.index_token(' INTERSECT ')
sql_reserved_tokens_revtok.index_token(' IS ')
sql_reserved_tokens_revtok.index_token(' JOIN ')
sql_reserved_tokens_revtok.index_token(' LEFT ')
sql_reserved_tokens_revtok.index_token(' LIKE ')
sql_reserved_tokens_revtok.index_token(' LIMIT ')
sql_reserved_tokens_revtok.index_token(' LOWER ')
sql_reserved_tokens_revtok.index_token(' MAX ')
sql_reserved_tokens_revtok.index_token(' MIN ')
sql_reserved_tokens_revtok.index_token(' N ')
sql_reserved_tokens_revtok.index_token(' NOT ')
sql_reserved_tokens_revtok.index_token(' NULL ')
sql_reserved_tokens_revtok.index_token(' ON ')
sql_reserved_tokens_revtok.index_token(' OR ')
sql_reserved_tokens_revtok.index_token(' ORDER ')
sql_reserved_tokens_revtok.index_token(' OUTER ')
sql_reserved_tokens_revtok.index_token(' SELECT ')
sql_reserved_tokens_revtok.index_token(' SUM ')
sql_reserved_tokens_revtok.index_token(' THEN ')
sql_reserved_tokens_revtok.index_token(' UNION ')
sql_reserved_tokens_revtok.index_token(' WHEN ')
sql_reserved_tokens_revtok.index_token(' WHERE ')
sql_reserved_tokens_revtok.index_token(' Y ')
sql_reserved_tokens_revtok.index_token(' YES ')
sql_reserved_tokens_revtok.index_token(' YEAR ')
sql_reserved_tokens_revtok.index_token('" ')
sql_reserved_tokens_revtok.index_token('%')
sql_reserved_tokens_revtok.index_token('% ')
sql_reserved_tokens_revtok.index_token('(')
sql_reserved_tokens_revtok.index_token('( ')
sql_reserved_tokens_revtok.index_token(') ')
sql_reserved_tokens_revtok.index_token('. ')
sql_reserved_tokens_revtok.index_token('.')
sql_reserved_tokens_revtok.index_token('- ')
sql_reserved_tokens_revtok.index_token('/')
sql_reserved_tokens_revtok.index_token('-')
sql_reserved_tokens_revtok.index_token(':')
sql_reserved_tokens_revtok.index_token('0 ')
sql_reserved_tokens_revtok.index_token('1 ')
sql_reserved_tokens_revtok.index_token('2 ')
sql_reserved_tokens_revtok.index_token('3 ')
sql_reserved_tokens_revtok.index_token('4 ')
sql_reserved_tokens_revtok.index_token('5 ')
sql_reserved_tokens_revtok.index_token('6 ')
sql_reserved_tokens_revtok.index_token('7 ')
sql_reserved_tokens_revtok.index_token('8 ')
sql_reserved_tokens_revtok.index_token('9 ')
sql_reserved_tokens_revtok.index_token('alias0 ')
sql_reserved_tokens_revtok.index_token('alias1 ')
sql_reserved_tokens_revtok.index_token('alias2 ')
sql_reserved_tokens_revtok.index_token('alias3 ')
sql_reserved_tokens_revtok.index_token('alias4 ')
sql_reserved_tokens_revtok.index_token('alias5 ')
sql_reserved_tokens_revtok.index_token('alias6 ')
sql_reserved_tokens_revtok.index_token('alias7 ')
sql_reserved_tokens_revtok.index_token('alias8 ')
sql_reserved_tokens_revtok.index_token('alias9 ')


from src.data_processor.vocab_utils import Vocabulary

field_types = Vocabulary('field_types')
field_types.index_token('not_a_field')
field_types.index_token('text')
field_types.index_token('number')
field_types.index_token('time')
field_types.index_token('boolean')
field_types.index_token('others')


field_vocab = Vocabulary('field')
field_vocab.index_token('*')


arithmetic_ops = Vocabulary('arithmetic_ops')
arithmetic_ops.index_token('+')
arithmetic_ops.index_token('-')
arithmetic_ops.index_token('*')
arithmetic_ops.index_token('/')


aggregation_ops = Vocabulary('aggregation_ops')
aggregation_ops.index_token('')
aggregation_ops.index_token('max')
aggregation_ops.index_token('min')
aggregation_ops.index_token('count')
aggregation_ops.index_token('sum')
aggregation_ops.index_token('avg')


condition_ops = Vocabulary('condition_ops')
condition_ops.index_token('=')
condition_ops.index_token('>')
condition_ops.index_token('<')
condition_ops.index_token('>=')
condition_ops.index_token('<=')
condition_ops.index_token('BETWEEN')
condition_ops.index_token('LIKE')
condition_ops.index_token('IN')


logical_ops = Vocabulary('logical_ops')
logical_ops.index_token('AND')
logical_ops.index_token('OR')


int_vocab = Vocabulary('numerical_value')


value_vocab = Vocabulary('value')
value_vocab.index_token('t')
value_vocab.index_token('f')
value_vocab.index_token('m')
value_vocab.index_token('yes')
value_vocab.index_token('no')
value_vocab.index_token('"')
value_vocab.index_token('!')
value_vocab.index_token('%')
value_vocab.index_token('(')
value_vocab.index_token(')')
value_vocab.index_token('[')
value_vocab.index_token(']')
value_vocab.index_token('{')
value_vocab.index_token('}')
value_vocab.index_token('^')
value_vocab.index_token('$')
value_vocab.index_token('.')
value_vocab.index_token('-')
value_vocab.index_token('*')
value_vocab.index_token('+')
value_vocab.index_token('?')
value_vocab.index_token('|')
value_vocab.index_token('/')
value_vocab.index_token('\\')
value_vocab.index_token(':')


value_vocab_revtok = Vocabulary('value_revtok')
value_vocab_revtok.index_token(' T ')
value_vocab_revtok.index_token(' F ')
value_vocab_revtok.index_token(' M ')
value_vocab_revtok.index_token(' yes ')
value_vocab_revtok.index_token(' no ')
value_vocab_revtok.index_token(' "')
value_vocab_revtok.index_token(' " ')
value_vocab_revtok.index_token('" ')
value_vocab_revtok.index_token('!')
value_vocab_revtok.index_token('% ')
value_vocab_revtok.index_token('%')
value_vocab_revtok.index_token('(')
value_vocab_revtok.index_token(')')
value_vocab_revtok.index_token('[')
value_vocab_revtok.index_token(']')
value_vocab_revtok.index_token('{')
value_vocab_revtok.index_token('}')
value_vocab_revtok.index_token('^')
value_vocab_revtok.index_token('$')
value_vocab_revtok.index_token('.')
value_vocab_revtok.index_token('-')
value_vocab_revtok.index_token('*')
value_vocab_revtok.index_token('+')
value_vocab_revtok.index_token('?')
value_vocab_revtok.index_token('|')
value_vocab_revtok.index_token('/')
value_vocab_revtok.index_token('/ ')
value_vocab_revtok.index_token('\\')
value_vocab_revtok.index_token('\\ ')
value_vocab_revtok.index_token(':')