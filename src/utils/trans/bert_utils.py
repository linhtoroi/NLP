from transformers import BertTokenizer

# load hàm tokenize bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bt = tokenizer

# Lấy token padding, cls, sep, mask, unknow
pad_token = tokenizer.pad_token
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
mask_token = tokenizer.mask_token
unk_token = tokenizer.unk_token

# Convert những token đó sang ids
pad_id = bt.convert_tokens_to_ids(pad_token)
cls_id = bt.convert_tokens_to_ids(cls_token)
sep_id = bt.convert_tokens_to_ids(sep_token)
mask_id = bt.convert_tokens_to_ids(mask_token)
unk_id = bt.convert_tokens_to_ids(unk_token)

# Thêm những token đánh dấu bắt đầu bảng, trường, giá trị, * - cột đầu tiên, khóa chính, khóa ngoài, khóa ngoại đến bảng nào, khóa ngoại đến cột nào, và convert sang ids
table_marker = '[unused50]'
field_marker = '[unused51]'
value_marker = '[unused49]'

asterisk_marker = '*'
primary_key_marker = '[unused53]'
foreign_key_marker = '[unused54]'
foreign_key_ref_table_marker = '[unused55]'
foreign_key_ref_field_marker = '[unused56]'
table_marker_id = bt.convert_tokens_to_ids(table_marker)
field_marker_id = bt.convert_tokens_to_ids(field_marker)
value_marker_id = bt.convert_tokens_to_ids(value_marker)
asterisk_marker_id = bt.convert_tokens_to_ids(asterisk_marker)
primary_key_marker_id = bt.convert_tokens_to_ids(primary_key_marker)
foreign_key_marker_id = bt.convert_tokens_to_ids(foreign_key_marker)
foreign_key_ref_table_marker_id = bt.convert_tokens_to_ids(foreign_key_ref_table_marker)
foreign_key_ref_field_marker_id = bt.convert_tokens_to_ids(foreign_key_ref_field_marker)

# Thêm token loại dữ liệu text, số, time, boolean, other và convert sang ids
text_field_marker = '[unused61]'
number_field_marker = '[unused62]'
time_field_marker = '[unused63]'
boolean_field_marker = '[unused64]'
other_field_marker = '[unused65]'
text_field_marker_id = bt.convert_tokens_to_ids(text_field_marker)
number_field_marker_id = bt.convert_tokens_to_ids(number_field_marker)
time_field_marker_id = bt.convert_tokens_to_ids(time_field_marker)
boolean_field_marker_id = bt.convert_tokens_to_ids(boolean_field_marker)
other_field_marker_id = bt.convert_tokens_to_ids(other_field_marker)


typed_field_markers = [
    text_field_marker,
    number_field_marker,
    time_field_marker,
    boolean_field_marker,
    other_field_marker
]
