import numpy as np
import torch
from src.utils import ops

BRIDGE=2

def beam_search(alpha, model, decoder, decoder_embeddings, num_steps, beam_size, encoder_final_hidden,
                encoder_hiddens=None, encoder_masks=None, encoder_ptr_value_ids=None, constant_hiddens=None,
                constant_hidden_masks=None, schema_hiddens=None, schema_hidden_masks=None, table_masks=None,
                schema_memory_masks=None, db_scope=None, no_from=False, start_embedded=None):
    

    def compute_memory_inputs(constant_seq_len, schema_seq_len, table_masks):
        """
        :return memory_inputs: [batch_size, memory_size] marked value_id, table_id, field_id
            memory_inputs[i]: [value_id, value_id, ..., table_id, field_id, ..., table_id, ...]
        """
        memory_size = constant_seq_len + schema_seq_len
        memory_max_size = int(max(memory_size))
        # Get mask of all question
        memory_input_constant_masks = (ops.batch_arange_cuda(batch_size, memory_max_size) <
                                       constant_seq_len.unsqueeze(1)).long()
        # Get mask of all schema
        memory_input_schema_masks = 1 - memory_input_constant_masks

        # Get the memory_inputs with value position replaced with value_id and field position replaced with field_id
        memory_inputs = memory_input_constant_masks * decoder.vocab.value_id + \
                        memory_input_schema_masks * decoder.vocab.field_id

        # Reshape memory inputs of all instances in batch into 1 array  
        memory_inputs = memory_inputs.view(-1)
        # Get the position (x,y) of table id of schema of all instances in batch: x: table_idx1, y: table_idx2
        table_idx1, table_idx2 = torch.nonzero(table_masks).split(1, dim=1)
        # Get the position (x*memory_max_size + seq_len_of_instance_in_batch_x + y) (Example: x*72 + 24 + y)
        table_pos_ = table_idx1 * memory_max_size + constant_seq_len[table_idx1] + table_idx2
        # Replace table position in memory_inputs with table_id
        memory_inputs[table_pos_] = decoder.vocab.table_id

        # Reshape memory inputs of all instances in batch into origin size [batch_size, memory_batch_size]
        memory_inputs = memory_inputs.view(batch_size, memory_max_size)

        # Get table, field mask
        memory_input_table_masks = (memory_inputs == decoder.vocab.table_id).long()
        memory_input_field_masks = ops.int_ones_var_cuda(memory_input_table_masks.size()) if no_from else \
                                   ops.int_zeros_var_cuda(memory_input_table_masks.size())

        return memory_inputs, memory_input_table_masks, memory_input_field_masks, memory_input_constant_masks

    def get_vocab_cat_masks():
        v_clause_mask = ops.int_var_cuda(decoder.vocab.clause_mask)
        v_op_mask = ops.int_var_cuda(decoder.vocab.op_mask)
        v_join_mask = ops.int_var_cuda(decoder.vocab.join_mask)
        v_others_mask = 1 - v_clause_mask - v_op_mask - v_join_mask
        return v_clause_mask, v_op_mask, v_join_mask, v_others_mask

    def offset_hidden(h, beam_offset):
        if isinstance(h, tuple):
            return torch.index_select(h[0], 1, beam_offset), torch.index_select(h[1], 1, beam_offset)
        else:
            return torch.index_select(h, 1, beam_offset)

    def update_beam_search_history(history, state, beam_offset, offset_dim, seq_dim, offset_state=False):
        if history is None:
            return state
        else:
            history = torch.index_select(history, offset_dim, beam_offset)
            if offset_state:
                state = torch.index_select(state, offset_dim, beam_offset)
            return torch.cat([history, state], seq_dim)

    if encoder_hiddens is None:
        batch_size = constant_hiddens.size(0)
    else:
        batch_size = encoder_hiddens.size(0)
    full_size = batch_size * beam_size

    # Get vocab id
    start_id = decoder.vocab.start_id
    eos_id = decoder.vocab.eos_id
    digit_0_id = decoder.vocab.to_idx('0')
    digit_1_id = decoder.vocab.to_idx('1')
    digit_2_id = decoder.vocab.to_idx('2')
    digit_3_id = decoder.vocab.to_idx('3')
    digit_4_id = decoder.vocab.to_idx('4')
    digit_5_id = decoder.vocab.to_idx('5')
    digit_6_id = decoder.vocab.to_idx('6')
    digit_7_id = decoder.vocab.to_idx('7')
    digit_8_id = decoder.vocab.to_idx('8')
    digit_9_id = decoder.vocab.to_idx('9')
    digit_10_id = decoder.vocab.to_idx('10')
    digit_11_id = decoder.vocab.to_idx('11')
    digit_12_id = decoder.vocab.to_idx('12')
    digit_s0_id = decoder.vocab.to_idx('##0')
    digit_s1_id = decoder.vocab.to_idx('##1')
    digit_s2_id = decoder.vocab.to_idx('##2')
    digit_s3_id = decoder.vocab.to_idx('##3')
    digit_s4_id = decoder.vocab.to_idx('##4')
    digit_s5_id = decoder.vocab.to_idx('##5')
    seen_eos = ops.byte_zeros_var_cuda([full_size, 1])
    seq_len = 0

    # Repeat encoder_final_hidden beam_size times => [1, batch_size*beam_size, hidden_dim]
    if type(encoder_final_hidden) is tuple:
        assert(len(encoder_final_hidden) == 2)
        hidden = (ops.tile_along_beam(encoder_final_hidden[0], beam_size, dim=1),
                  ops.tile_along_beam(encoder_final_hidden[1], beam_size, dim=1))
    elif encoder_final_hidden is not None:
        hidden = ops.tile_along_beam(encoder_final_hidden, beam_size, dim=1)
    else:
        hidden = None

    # Get question len from mask
    constant_seq_len = constant_hidden_masks.size(1) - constant_hidden_masks.sum(dim=1)

    vocab_masks = None
    memory_masks = None

    if model == BRIDGE:
        # Get schema len from mask
        schema_seq_len = schema_hidden_masks.size(1) - schema_hidden_masks.sum(dim=1)

        # memory input marked with value_id, table_id, field_id
        memory_inputs, m_table_masks, m_field_masks, m_value_masks = \
            compute_memory_inputs(constant_seq_len, schema_seq_len, table_masks)

        # db_scope = (table_position, table_field_scope)
        if db_scope is not None:
            memory_masks = ops.int_zeros_var_cuda([batch_size, memory_inputs.size(1)])
            table_pos, table_field_scopes = db_scope
            table_memory_pos = constant_seq_len.unsqueeze(1) * (table_pos > 0).long() + table_pos
            table_memory_pos = ops.tile_along_beam(table_memory_pos, beam_size)
            table_field_scopes = ops.tile_along_beam(table_field_scopes, beam_size)
            db_scope = (table_memory_pos, table_field_scopes)
    else:
        memory_inputs = None

    if model == BRIDGE:
        # Repeat encoder_hiddens, encoder_masks beam_size times => [1, batch_size*beam_size, hidden_dim], [1, batch_size*hidden_dim]
        encoder_hiddens = ops.tile_along_beam(encoder_hiddens, beam_size)
        encoder_masks = ops.tile_along_beam(encoder_masks, beam_size)

        
        if memory_masks is not None:
            memory_masks = ops.tile_along_beam(memory_masks, beam_size)
            m_table_masks = ops.tile_along_beam(m_table_masks, beam_size)
            m_field_masks = ops.tile_along_beam(m_field_masks, beam_size)
            m_value_masks = ops.tile_along_beam(m_value_masks, beam_size)

        if memory_inputs is not None:
            constant_seq_len = ops.tile_along_beam(constant_seq_len, beam_size)
            memory_inputs = ops.tile_along_beam(memory_inputs, beam_size)

        if encoder_ptr_value_ids is not None:
            encoder_ptr_value_ids = ops.tile_along_beam(encoder_ptr_value_ids, beam_size)
            
        seq_p_pointers = None
        ptr_context = None
        seq_text_ptr_weights = None
    else:
        raise NotImplementedError

    pred_score = 0
    outputs, hiddens = None, (None, None)

    for i in range(num_steps):
        if i > 0:
            if model == BRIDGE:
                # [batch_size, 1]
                vocab_mask = (inputt < decoder.vocab_size).long()
                point_mask = 1 - vocab_mask
                memory_pos = (inputt - decoder.vocab_size) * point_mask
                memory_input = ops.batch_lookup(memory_inputs, memory_pos, vector_output=False)
                input_ = vocab_mask * inputt + point_mask * memory_input
                digit_mask = ((inputt == digit_0_id) |
                       (inputt == digit_1_id) |
                       (inputt == digit_2_id) |
                       (inputt == digit_3_id) |
                       (inputt == digit_4_id) |
                       (inputt == digit_5_id) |
                       (inputt == digit_6_id) |
                       (inputt == digit_7_id) |
                       (inputt == digit_8_id) |
                       (inputt == digit_9_id) |
                       (inputt == digit_10_id) |
                       (inputt == digit_11_id) |
                       (inputt == digit_12_id) |
                       (inputt == digit_s0_id) |
                       (inputt == digit_s1_id) |
                       (inputt == digit_s2_id) |
                       (inputt == digit_s3_id) |
                       (inputt == digit_s4_id) |
                       (inputt == digit_s5_id))
                vocab_mask[digit_mask] = 0
                input_[digit_mask] = decoder.vocab.value_id
                if db_scope is not None:
                    # [full_size, 3 (table, field, value）]
                    input_types = ops.long_var_cuda([decoder.vocab.table_id,
                                                     decoder.vocab.field_id,
                                                     decoder.vocab.value_id]).unsqueeze(0).expand([input_.size(0), 3])
                    # [full_size, 4 (vocab, table, field, value)]
                    input_type = torch.cat([vocab_mask, (input_ == input_types).long()], dim=1)
                    # [full_size, max_num_tables], [full_size, max_num_tables, max_num_fields_per_table]
                    table_memory_pos, table_field_scopes = db_scope
                    # update vocab masks
                    # vocab_masks = torch.index_select(vocab_masks, 0, beam_offset)
                    # update memory masks
                    m_field_masks = torch.index_select(m_field_masks, 0, beam_offset)
                    # [full_size, max_num_tables]
                    table_input_mask = (memory_pos == table_memory_pos)
                    if table_input_mask.max() > 0:
                        # [full_size, 1, max_num_fields_per_table]
                        db_scope_update_idx, _ = ops.batch_binary_lookup_3D(
                            table_field_scopes, table_input_mask, pad_value=0)
                        assert(db_scope_update_idx.size(1) == 1)
                        db_scope_update_idx.squeeze_(1)
                        db_scope_update_mask = (db_scope_update_idx > 0)
                        db_scope_update_idx = constant_seq_len.unsqueeze(1) + db_scope_update_idx
                        # db_scope_update: [full_size, memory_seq_len] binary mask in which the newly included table
                        # fields are set to 1 and the rest are set to 0
                        m_field_masks.scatter_(index=constant_seq_len.unsqueeze(1),
                                               src=ops.int_ones_var_cuda([batch_size*beam_size, 1]), dim=1)
                        m_field_masks.scatter_add_(index=db_scope_update_idx, src=db_scope_update_mask.long(), dim=1)
                        m_field_masks = (m_field_masks > 0).long()
                    # Heuristics:
                    # - table/field only appear after SQL keywords
                    # - value only appear after SQL keywords or other value token
                    memory_masks = input_type[:, 0].unsqueeze(1) * m_table_masks + \
                                   input_type[:, 0].unsqueeze(1) * m_field_masks + \
                                   (input_type[:, 0] + input_type[:, 3]).unsqueeze(1) * m_value_masks
            else:
                input_ = inputt
            input_embedded = decoder_embeddings(input_)
        else:
            if start_embedded is None:
                inputt = ops.int_fill_var_cuda([full_size, 1], start_id)
                input_embedded = decoder_embeddings(inputt)
            else:
                raise NotImplementedError
        if model == BRIDGE:
            output, hidden, ptr_context = decoder(
                input_embedded,
                hidden,
                encoder_hiddens,
                encoder_masks,
                ptr_context,
                vocab_masks=vocab_masks,
                memory_masks=memory_masks,
                encoder_ptr_value_ids=encoder_ptr_value_ids, decoder_ptr_value_ids = None, last_output=inputt)
        else:
            raise NotImplementedError

        # [full_size, vocab_size]
        output.squeeze_(1)
        vocab_size = output.size(1)

        seq_len += (1 - seen_eos.float())
        n_len_norm_factor = torch.pow(5 + seq_len, alpha) / np.power(5 + 1, alpha)
        # [full_size, vocab_size]
        if i == 0:
            raw_scores = \
                output + (ops.arange_cuda(beam_size).repeat(batch_size) > 0).float().unsqueeze(1) * (-ops.HUGE_INT)
        else:
            raw_scores = (pred_score * len_norm_factor + output * (1 - seen_eos.float())) / n_len_norm_factor
            eos_mask = ops.ones_var_cuda([1, vocab_size])
            eos_mask[0, eos_id] = 0
            raw_scores += (seen_eos.float() * eos_mask) * (-ops.HUGE_INT)

        len_norm_factor = n_len_norm_factor
        # [batch_size, beam_size * vocab_size]
        raw_scores = raw_scores.view(batch_size, beam_size * vocab_size)
        # [batch_size, beam_size]
        log_pred_prob, pred_idx = torch.topk(raw_scores, beam_size, dim=1)
        # [full_size]
        beam_offset = (pred_idx // vocab_size + ops.arange_cuda(batch_size).unsqueeze(1) * beam_size).view(-1)
        # [full_size, 1]
        pred_idx = (pred_idx % vocab_size).view(full_size, 1)
        log_pred_prob = log_pred_prob.view(full_size, 1)

        # update search history and save output
        # [num_layers*num_directions, full_size, hidden_dim]
        hidden = offset_hidden(hidden, beam_offset)
        # [num_layers*num_directions, full_size, seq_len, hidden_dim]
        hiddens = (
            update_beam_search_history(hiddens[0], hidden[0].unsqueeze(2), beam_offset, 1, 2),
            update_beam_search_history(hiddens[1], hidden[1].unsqueeze(2), beam_offset, 1, 2)
        )

        if outputs is not None:
            seq_len = torch.index_select(seq_len, 0, beam_offset)
            len_norm_factor = torch.index_select(len_norm_factor, 0, beam_offset)
            seen_eos = torch.index_select(seen_eos, 0, beam_offset)
        seen_eos = seen_eos | (pred_idx == eos_id)
        outputs = update_beam_search_history(outputs, pred_idx, beam_offset, 0, 1)
        pred_score = log_pred_prob

        inputt = pred_idx

        # save attention weights for interpretation and sanity checking
        if model == BRIDGE:
            ptr_context = (torch.index_select(ptr_context[0], 0, beam_offset),
                           torch.index_select(ptr_context[1], 0, beam_offset))
            seq_text_ptr_weights = update_beam_search_history(
                seq_text_ptr_weights, ptr_context[1], beam_offset, 0, 2)
            seq_p_pointers = update_beam_search_history(
                seq_p_pointers, ptr_context[0].squeeze(2), beam_offset, 0, 1)
        else:
            raise NotImplementedError

    if model == BRIDGE:
        output_obj = outputs, pred_score, seq_p_pointers, seq_text_ptr_weights, seq_len
    else:
        raise NotImplementedError

    return output_obj

