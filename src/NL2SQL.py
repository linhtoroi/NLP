from re import T
from src.model.model import NL2SQL
from src.data_processor import data
from src.data_processor.processor_utils import get_table_aware_transformer_encoder_inputs
from src.utils import ops
from torch import nn, optim
import torch.nn.functional as F
import src.data_processor.vectorizers as vec
import moz_sp
import torch
import copy
import random
from tqdm import tqdm
import wandb
import os
import numpy as np
import src.eval.evaluate as spider_eval_tools
from src.eval import eval_tools
import collections
import pickle
import shutil
from torch.optim import Optimizer
from src.model.schema_graph import SchemaGraph, SchemaGraphs
from src.data_processor import tokenizers
from src.data_processor import data_loader
from src.data_processor.path_utils import get_processed_data_path
from src.model.model_pre import Text2SQLWrapper

class Params:
    def __init__(self, improvement=False):
        self.improvement = improvement
        self.data_dir="data/spider"
        self.db_dir="data/spider/database"
        self.dataset_name="spider"
        self.model="bridge"
        self.question_split=True
        self.query_split=False
        self.question_only=True
        self.normalize_variables=False
        self.denormalize_sql=True
        self.omit_from_clause=False
        self.no_join_condition=False
        self.table_shuffling=True
        self.use_lstm_encoder=True
        self.use_meta_data_encoding=True
        self.use_graph_encoding=False
        self.use_typed_field_markers=False
        self.use_picklist=True
        self.anchor_text_match_threshold=0.85
        self.no_anchor_text=False
        self.top_k_picklist_matches=2
        self.sql_consistency_check=True
        self.atomic_value_copy=False
        self.process_sql_in_execution_order=True
        self.share_vocab=False
        self.sample_ground_truth=False
        self.save_nn_weights_for_visualizations=False
        self.vocab_min_freq=0
        self.text_vocab_min_freq=0
        self.program_vocab_min_freq=0
        self.max_in_seq_len=512
        self.max_out_seq_len=60
        self.num_steps=100000
        self.curriculum_interval=0
        self.num_peek_steps=1000
        self.num_accumulation_steps=2
        self.save_best_model_only=True
        self.train_batch_size=4
        self.dev_batch_size=4
        self.encoder_input_dim=768
        self.encoder_hidden_dim=400
        self.decoder_input_dim=400
        self.num_rnn_layers=1
        self.num_const_attn_layers=0
        self.use_oracle_tables=False
        self.num_random_tables_added=0
        self.use_additive_features=False
        self.schema_augmentation_factor=1
        self.random_field_order=False
        self.data_augmentation_factor=1
        self.augment_with_wikisql=False
        self.num_values_per_field=0
        self.pretrained_bert="bert-base-uncased"
        self.fix_pretrained_bert_parameters=False
        self.bert_finetune_rate=0.00006
        self.learning_rate=0.0005
        self.learning_rate_scheduler="inverse-square"
        self.trans_learning_rate_scheduler="inverse-square"
        self.warmup_init_lr=0.0005
        self.warmup_init_ft_lr=0.00003
        self.num_warmup_steps=4000
        self.pretrained_bert_embedding_dropout_rate=0.3
        self.feat_emb_dropout = 0
        self.res_dropout = 0.2
        self.pretrained_lm_dropout_rate=0
        self.rnn_layer_dropout_rate=0
        self.rnn_weight_dropout=0
        self.cross_attn_dropout_rate=0
        self.cross_attn_num_heads=8
        self.res_input_dropout_rate=0.2
        self.res_layer_dropout_rate=0
        self.ff_input_dropout_rate=0.4
        self.ff_hidden_dropout_rate=0.0
        self.grad_norm=0.3
        self.decoding_algorithm="beam-search"
        self.beam_size=16
        self.bs_alpha=1.05
        self.data_parallel=False
        self.process_data=False
        self.process_new_data_split=False
        self.process_data = False
        self.process_new_data_split = False
        self.inference= False
        self.ensemble_inference= False
        self.predict_tables= False
        self.train=False
        self.test= False
        self.fine_tune= False
        self.demo= False
        self.demo_db=None
        self.data_statistics= False
        self.search_random_seed= False
        self.eval= False
        self.eval_by_relation_type= False
        self.error_analysis= False
        self.model_root_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'save_model')
        self.viz_root_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'viz')
        self.model_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'save_model')
        self.viz_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'viz')
        self.save_all_checkpoints= False
        self.gpu = 0
        self.leaderboard_submission= False
        self.codalab_data_dir=None
        self.codalab_db_dir=None
        self.checkpoint_path=None
        self.prediction_path=None
        self.use_pred_tables= False
        self.read_picklist= False
        self.enumerate_ground_truth= False
        self.model_id=None
        self.loss='cross_entropy'
        self.decoder_hidden_dim=-1
        self.rnn_input_dropout_rate=0.0
        self.schema_hidden_dim=200
        self.schema_dropout_rate=0.0
        self.schema_rnn_num_layers=1
        self.schema_rnn_input_dropout_rate=0.0
        self.schema_rnn_layer_dropout_rate=0.0
        self.schema_rnn_weight_dropout_rate=0.0
        self.encoder_tf_hidden_dim=-1
        self.decoder_tf_hidden_dim=-1
        self.sa_num_layers=2
        self.sa_num_heads=1
        self.sa_input_dropout_rate=0.0
        self.sa_dropout_rate=0.0
        self.ff_output_dropout_rate=0.0
        self.decoder_embedding_dropout_rate = 0
        self.seed=543
        self.num_epochs=200
        self.num_wait_epochs=200
        self.num_peek_epochs=2
        self.start_epoch=0
        self.num_wait_steps=20000
        self.num_log_steps=500
        self.start_step=0
        self.margin=0
        self.optimizer='adam'
        self.adam_beta1=0.9
        self.adam_beta2=0.999
        self.adam_eps=1e-8
        self.xavier_initialization=True
        self.random_parameters=False
        self.execution_guided_decoding= False
        self.grid_search= False
        self.atomic_value = False

class NL2SQLmodel():
    def __init__(self, params):
        vocabs = data.load_vocabs(params)
        in_vocab = vocabs['text']
        out_vocab = vocabs['sql']
        self.model = NL2SQL(params=params, in_vocab=in_vocab, out_vocab=out_vocab)
        self.params = params
        self.start_step = 0
        self.optim = None
        if params.optimizer == 'adam':
            self.optim = optim.Adam(
            [
                {'params': [p for n, p in self.model.named_parameters() if not 'trans_parameters' in n and p.requires_grad]},
                {'params': [p for n, p in self.model.named_parameters() if 'trans_parameters' in n and p.requires_grad],
                 'lr': params.bert_finetune_rate}
            ], lr=params.learning_rate)
        else:
            raise NotImplementedError

        if self.params.loss == 'cross_entropy':
            self.loss_fun = MaskedCrossEntropyLoss(self.model.out_vocab.pad_id)

        self.lr_scheduler = InverseSquareRootScheduler(
                    self.optim, [self.params.warmup_init_lr, self.params.warmup_init_ft_lr], [self.params.num_warmup_steps, self.params.num_warmup_steps],
                    self.params.num_steps)

        _, _, self.output_post_process, _ = tokenizers.get_tokenizers(self.params)


    def train(self):
        dataset = self.load_processed_data(self.params)
        train_data = dataset['train']
        print('{} training examples loaded'.format(len(train_data)))
        dev_data = dataset['dev']
        print('{} dev examples loaded'.format(len(dev_data)))

        if self.params.xavier_initialization:
            self.initialize_module(self.model, 'xavier')

        self.model.schema_graphs = dataset['schema']

        if self.params.checkpoint_path is not None:
            self.model.load_checkpoint(self.param.checkpoint_path)

        train_batch_size = self.params.train_batch_size

        epoch_losses = []
        best_dev_metrics = 0
        dev_metrics_history = []

        num_steps = self.params.num_steps * self.params.num_accumulation_steps
        num_peek_steps = self.params.num_peek_steps * self.params.num_accumulation_steps

        random.shuffle(train_data)

        step_id, example_id = 0, 0
        self.optim.zero_grad()
        self.model.train()

        for interval_step_id in range(self.start_step, num_steps, num_peek_steps):
            self.model.train()
            self.model.training = True
            for s_id in tqdm(range(num_peek_steps)):
                step_id = interval_step_id + s_id
                batch_end = example_id + train_batch_size
                if batch_end > len(train_data):
                    random.shuffle(train_data)
                    example_id, batch_end = 0, train_batch_size

                mini_batch = train_data[example_id:batch_end]
                example_id = batch_end

                # Get data
                formatted_batch = self.format_batch(mini_batch)
                encoder_input_ids = formatted_batch[0]
                decoder_input_ids = formatted_batch[1][0] if self.model.training else None

                encoder_ptr_input_ids = formatted_batch[2]
                encoder_ptr_value_ids, _ = formatted_batch[3]
                decoder_ptr_value_ids = formatted_batch[4][0] if self.model.training else None
                text_masks = self.get_text_masks(encoder_input_ids)

                transformer_output_value_masks = formatted_batch[5][0]
                schema_masks = self.get_schema_masks(encoder_ptr_input_ids[0])
                schema_memory_masks = formatted_batch[6][0]
                feature_ids = formatted_batch[8]

                # Forward model NL2SQL
                outputs = self.model.forward(encoder_ptr_input_ids, encoder_ptr_value_ids,
                                       text_masks, schema_masks, feature_ids,
                                       transformer_output_value_masks=transformer_output_value_masks,
                                       schema_memory_masks=schema_memory_masks,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_ptr_value_ids=decoder_ptr_value_ids)

                # Shift decoder_ptr_value_ids 1 step to left, and padding
                left_shift_targets = ops.left_shift_pad(decoder_ptr_value_ids, self.model.out_vocab.pad_id)
                # Calculate loss between output and left_shift_targets 
                loss = self.loss_fun(outputs, left_shift_targets)
                loss /= self.params.num_accumulation_steps
                loss.backward()
                epoch_losses.append(float(loss) * self.params.num_accumulation_steps)

                if (step_id + 1) % self.params.num_accumulation_steps == 0:
                    if self.params.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_norm)
                    self.lr_scheduler.step()
                    self.optim.step()
                    self.optim.zero_grad()

            if step_id > 0 and (step_id + 1) % num_peek_steps == 0:
                stdout_msg = 'Step {}: average training loss = {}'.format(
                    step_id / self.params.num_accumulation_steps, np.mean(epoch_losses))
                print(stdout_msg)
                epoch_losses = []
            
            if step_id > 0 and (step_id + 1) % num_peek_steps == 0:
                # save model best
                self.model.eval()
                self.model.training = False
                if self.params.process_sql_in_execution_order:
                    pred_restored_cache = self.load_pred_restored_cache()
                    pred_restored_cache_size = sum(len(v) for v in pred_restored_cache.values())
                else:
                    pred_restored_cache = None

                # inference return output
                output_dict = self.inference(dev_data, restore_clause_order=self.params.process_sql_in_execution_order,
                                                pred_restored_cache=pred_restored_cache,
                                                check_schema_consistency_=self.params.sql_consistency_check,
                                                inline_eval=True, verbose=False)


                # evaluate
                metrics = eval_tools.get_exact_match_metrics(dev_data, output_dict['pred_decoded'])
                dev_metrics_history.append(metrics)
                
                # evaluate metric is Exact Match (EM), predicted SQL query match 100% ground truth SQL
                eval_metrics = metrics['top_1_em']
                
                print('Dev set performance:')
                print('Top-1 exact match: {}'.format(metrics['top_1_em']))
                print('Top-3 exact match: {}'.format(metrics['top_3_em']))

                # save best model after compare
                if eval_metrics >= best_dev_metrics:
                    best_dev_metrics = eval_metrics
                    self.save_checkpoint(step_id, step_id / num_peek_steps, output_dict['pred_decoded'], is_best=True)
            
                # Save cache
                if self.params.process_sql_in_execution_order:
                    new_pred_restored_cache_size = sum(len(v) for v in output_dict['pred_restored_cache'].values())
                    newly_cached_size = new_pred_restored_cache_size - pred_restored_cache_size
                    if newly_cached_size > 0:
                        self.save_pred_restored_cache(output_dict['pred_restored_cache'], newly_cached_size)


    def predict(self, question, file_db_name, db_name, db_path=None):
        """
        Get SQL query from question

        :param question: question
        :param file_db_name: name of database file
        :param db_name: name of database
        
        :return sql_query: SQL query
        :rtype: string
        """
        data_dir = 'data/'

        # Get path of database
        if db_path is not None:
            db_path = os.path.join(self.params.db_dir, db_path, '{}.sqlite'.format(db_path))

        schema = SchemaGraph(db_name, db_path=db_path)
        
        # Load schema from json file
        import json
        in_json = os.path.join(self.params.data_dir, file_db_name+'.json')
        with open(in_json) as f:
            table = json.load(f)
            schema.load_data_from_spider_json(table[0])

        # Init model predict
        t2sql = Text2SQLWrapper(self.params, schema, self)
        
        # Inference
        output = t2sql.process(question, schema.name)
        translatable = output['translatable']
        sql_query = output['sql_query']
        confusion_span = output['confuse_span']
        replacement_span = output['replace_span']

        return sql_query


    def inference(self, examples, decode_str_output=True, restore_clause_order=False, pred_restored_cache=None,
                  check_schema_consistency_=True, inline_eval=False, verbose=False):
        """
        Inference data
        :param examples: data
        :param decode_str_output:
        :param restore_clause_order: process sql in execution order
        :param pred_restored_cache:
        :param check_schema_consistency_: sql consistency check
        :param inline_eval:
        :param verbose:
        """

        pred_list, pred_score_list, pred_decoded_list, pred_decoded_score_list = [], [], [], []
        if restore_clause_order:
            if pred_restored_cache is None:
                pred_restored_cache = dict()

        for batch_start_id in tqdm(range(0, len(examples), self.params.dev_batch_size)):
            # Get dev data with batch_size = 8
            mini_batch = examples[batch_start_id:batch_start_id + self.params.dev_batch_size]
            formatted_batch = self.format_batch(mini_batch)

            encoder_input_ids = formatted_batch[0]
            decoder_input_ids = formatted_batch[1][0] if self.model.training else None

            encoder_ptr_input_ids = formatted_batch[2]
            encoder_ptr_value_ids, _ = formatted_batch[3]
            decoder_ptr_value_ids = formatted_batch[4][0] if self.model.training else None
            text_masks = self.get_text_masks(encoder_input_ids)

            transformer_output_value_masks = formatted_batch[5][0]
            schema_masks = self.get_schema_masks(encoder_ptr_input_ids[0])
            schema_memory_masks = formatted_batch[6][0]
            feature_ids = formatted_batch[8]

            # Forward through encoder and beam search
            outputs = self.model.forward(encoder_ptr_input_ids, encoder_ptr_value_ids,
                                       text_masks, schema_masks, feature_ids,
                                       transformer_output_value_masks=transformer_output_value_masks,
                                       schema_memory_masks=schema_memory_masks)

            # outputs, pred_score, seq_p_pointers, seq_text_ptr_weights, seq_len
            preds, pred_scores, text_p_pointers, text_ptr_weights, seq_len = outputs
            text_p_pointers.unsqueeze_(2)
            p_pointers = torch.cat([1 - text_p_pointers, text_p_pointers], dim=2)

            pred_list.append(preds)
            pred_score_list.append(pred_scores)
            if decode_str_output or verbose:
                for i in range(len(mini_batch)):
                    example = mini_batch[i]
                    db_name = example.db_name
                    schema = self.model.schema_graphs[db_name]
                    table_po, field_po = None, None
                    if self.params.use_oracle_tables:
                        if self.params.num_random_tables_added > 0:
                            table_po, field_po = formatted_batch[-1][i]

                    exp_output_strs, exp_output_scores, exp_seq_lens, exp_correct = [], [], [], []

                    if inline_eval:
                        gt_program_list = example.program_list

                    if self.params.decoding_algorithm == 'beam-search':
                        for j in range(self.params.beam_size):
                            beam_id = i * self.params.beam_size + j
                            post_processed_output = self.post_process_nn_output(
                                beam_id, example.dataset_id, example, preds, schema, text_ptr_weights, p_pointers,
                                table_po=table_po, field_po=field_po, verbose=verbose)
                            if post_processed_output:
                                pred_sql = post_processed_output[0]
                                if restore_clause_order:
                                    if pred_restored_cache and db_name in pred_restored_cache and \
                                            pred_sql in pred_restored_cache[db_name]:
                                        restored_pred, grammatical, schema_consistent = pred_restored_cache[db_name][pred_sql]
                                    else:
                                        # Loi o dong nay
                                        restored_pred, grammatical, schema_consistent = moz_sp.restore_clause_order(
                                            pred_sql, schema, check_schema_consistency_=check_schema_consistency_,
                                            verbose=verbose)
                                        if pred_restored_cache and check_schema_consistency_:
                                            if db_name not in pred_restored_cache:
                                                pred_restored_cache[db_name] = dict()
                                            pred_restored_cache[db_name][pred_sql] = restored_pred, grammatical, \
                                                                                     schema_consistent
                                    if check_schema_consistency_ and not schema_consistent:
                                        restored_pred = None
                                    pred_sql = restored_pred
                                else:
                                    if check_schema_consistency_:
                                        if not moz_sp.check_schema_consistency(
                                                pred_sql, schema, in_execution_order=self.params.process_sql_in_execution_order):
                                            pred_sql = None
                            else:
                                pred_sql = None
                            if pred_sql:
                                exp_output_strs.append(pred_sql)
                                exp_output_scores.append(float(pred_scores[beam_id]))
                                exp_seq_lens.append(int(seq_len[beam_id]))
                                if inline_eval:
                                    results = eval_tools.eval_prediction(
                                        pred=pred_sql,
                                        gt_list=gt_program_list,
                                        dataset_id=example.dataset_id,
                                        db_name=example.db_name,
                                        in_execution_order=(self.params.process_sql_in_execution_order and
                                                            not restore_clause_order))
                                    correct, _, _ = results
                                    exp_correct.append(correct)
                                    correct_ = correct[1] if isinstance(correct, tuple) else correct
                                    if correct_:
                                        break
                    else:
                        raise NotImplementedError
                    num_preds = len(exp_output_strs)
                    pred_decoded_list.append(exp_output_strs)
                    pred_decoded_score_list.append(exp_output_scores[:num_preds])

                    if not pred_decoded_list[-1] and not self.params.demo:
                        pred_decoded_list[-1].append(self.get_dummy_prediction(schema))
                        pred_decoded_score_list[-1].append(-ops.HUGE_INT)

        out_dict = dict()
        out_dict['preds'] = ops.pad_and_cat(pred_list, self.model.out_vocab.pad_id)
        out_dict['pred_scores'] = torch.cat(pred_score_list)
        if decode_str_output:
            out_dict['pred_decoded'] = pred_decoded_list
            out_dict['pred_decoded_scores'] = pred_decoded_score_list
        if restore_clause_order:
            out_dict['pred_restored_cache'] = pred_restored_cache

        return out_dict
        
    def load_processed_data(self, params):
        """
        Load preprocessed data file.
        """
        if params.process_sql_in_execution_order:
            split = 'test' if params.test else 'dev'
            pred_restored_cache_path = os.path.join(
                params.model_dir, '{}.eo.pred.restored.pkl'.format(split))
            if not os.path.exists(pred_restored_cache_path):
                cache_path = os.path.join(params.data_dir, '{}.eo.pred.restored.pkl'.format(split))
                # cache_path = '/home/linhhoang/NL2SQL/data/spider/dev.eo.pred.restored.pkl'
                if not os.path.exists(cache_path):
                    pred_restored_cache = collections.defaultdict(dict)
                    with open(cache_path, 'wb') as o_f:
                        pickle.dump(pred_restored_cache, o_f)
                shutil.copyfile(cache_path, pred_restored_cache_path)
                print('execution order restoration cache copied')
                print('source: {}'.format(cache_path))
                print('dest: {}'.format(pred_restored_cache_path))
                print()
        in_pkl = get_processed_data_path(params)
        # in_pkl = '/home/linhhoang/NL2SQL/data/spider/spider.bridge.question-split.ppl-0.85.2.dn.eo.bert.pkl'
        print('loading preprocessed data: {}'.format(in_pkl))
        with open(in_pkl, 'rb') as f:
            return pickle.load(f)

    def load_pred_restored_cache(self):
        split = 'test' if self.params.test else 'dev'
        pred_restored_cache_path = os.path.join(
            self.params.model_dir, '{}.eo.pred.restored.pkl'.format(split))
        if os.path.exists(pred_restored_cache_path):
            with open(pred_restored_cache_path, 'rb') as f:
                pred_restored_cache = pickle.load(f)
                pred_restored_cache_size = sum([len(pred_restored_cache[k]) for k in pred_restored_cache])
                print('{} pre-computed prediction order reconstruction cached'.format(pred_restored_cache_size))
            return pred_restored_cache
        else:
            return dict()
    


    def save_checkpoint(self, checkpoint_id, interval_step_id, predictions, loss=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param predictions: List of predicted strings.
        :param step_id: Training interval step id.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['model_state_dict'] = self.model.state_dict()
        if self.optim:
            checkpoint_dict['optimizer_state_dict'] = self.optim.state_dict()
        if self.lr_scheduler:
            checkpoint_dict['lr_scheduler_dict'] = self.lr_scheduler.state_dict()
        checkpoint_dict['interval_step_id'] = interval_step_id
        checkpoint_dict['loss'] = loss

        out_tar = os.path.join(self.params.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.params.model_dir, 'model-best.{}.tar'.format(self.params.beam_size))
            if os.path.exists(out_tar):
                shutil.copyfile(out_tar, best_path)
            else:
                torch.save(checkpoint_dict, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print('=> saving checkpoint to \'{}\''.format(out_tar))

        with open(os.path.join(self.params.model_dir, 'best_dev_iteration.{}.dat'.format(self.params.beam_size)), 'w') as o_f:
            o_f.write('{}'.format(checkpoint_id))
        out_txt = os.path.join(self.params.model_dir, 'predictions.{}.txt'.format(self.params.beam_size))
        with open(out_txt, 'w') as o_f:
            for pred_sql in predictions:
                o_f.write('{}\n'.format(pred_sql[0]))
            print('=> Model predictions saved to {}'.format(out_txt))

    def save_pred_restored_cache(self, pred_restored_cache, newly_cached_size):
        split = 'test' if self.params.test else 'dev'
        pred_restored_cache_path = os.path.join(
            self.params.model_dir, '{}.eo.pred.restored.pkl'.format(split))
        if os.path.exists(pred_restored_cache_path):
            shutil.copyfile(pred_restored_cache_path, pred_restored_cache_path + '.copy')
        with open(pred_restored_cache_path, 'wb') as o_f:
            pickle.dump(pred_restored_cache, o_f)
            print('{} sql order restoration newly cached'.format(newly_cached_size))

    def format_batch(self, mini_batch):
        # Get encoder_input_ids by batch, = text_ids and pad_id
        encoder_input_ids = ops.pad_batch([exp.text_ids for exp in mini_batch], self.model.in_vocab.pad_id)
        # Get decoder_input_ids by batch, = text_ids and pad_id if training else = None
        if self.model.training:
            decoder_input_ids = ops.pad_batch([exp.program_singleton_field_input_ids for exp in mini_batch], self.model.out_vocab.pad_id)
        else:
            decoder_input_ids = None

        table_samples = []
        encoder_ptr_input_ids, encoder_ptr_value_ids, decoder_ptr_value_ids = [], [], []
        primary_key_ids, foreign_key_ids, field_type_ids, table_masks, table_positions, table_field_scopes, \
            field_table_pos, transformer_output_value_masks, schema_memory_masks = [], [], [], [], [], [], [], [], []
        

        for exp in mini_batch:
            schema_graph = self.model.schema_graphs.get_schema(exp.db_id)

            if self.model.training:
                gt_table_names = [token for token, t in
                                        zip(exp.program_singleton_field_tokens, exp.program_singleton_field_token_types) if t == 0]
                gt_tables = set([schema_graph.get_table_id(t_name) for t_name in gt_table_names])
                tables = list(range(schema_graph.num_tables))
                if self.params.table_shuffling:
                    table_to_drop = random.choice(tables)
                    if table_to_drop not in gt_tables:
                        if random.uniform(0, 1) < 0.3:
                            tables = [x for x in tables if x != table_to_drop]
                    table_po, field_po = schema_graph.get_schema_perceived_order(
                        tables, random_table_order=True, random_field_order=self.params.random_field_order)
                else:
                    table_po, field_po = schema_graph.get_schema_perceived_order(
                        tables, random_table_order=False, random_field_order=self.params.random_field_order)
                
                question_encoding = exp.text if self.params.use_picklist else None

                schema_features, matched_values = schema_graph.get_serialization(
                        self.model.tu, flatten_features=True, table_po=table_po, field_po=field_po,
                        use_typed_field_markers=self.params.use_typed_field_markers,
                        use_graph_encoding=self.params.use_graph_encoding,
                        question_encoding = question_encoding,
                        top_k_matches=self.params.top_k_picklist_matches,
                        num_values_per_field=self.params.num_values_per_field,
                        no_anchor_text=self.params.no_anchor_text,
                        verbose=False)
                
                ptr_input_tokens, ptr_input_values, num_excluded_tables, num_excluded_fields = \
                        get_table_aware_transformer_encoder_inputs(
                            exp.text_ptr_values, exp.text_tokens, schema_features, self.model.tu)

                num_included_nodes = schema_graph.get_num_perceived_nodes(tables) + 1 \
                                         - num_excluded_tables - num_excluded_fields
                
                encoder_ptr_input_ids.append(self.model.tu.tokenizer.convert_tokens_to_ids(ptr_input_tokens))

                primary_key_ids.append(schema_graph.get_primary_key_ids(num_included_nodes, table_po, field_po))
                foreign_key_ids.append(schema_graph.get_foreign_key_ids(num_included_nodes, table_po, field_po))
                field_type_ids.append(schema_graph.get_field_type_ids(num_included_nodes, table_po, field_po))
                table_masks.append(schema_graph.get_table_masks(num_included_nodes, table_po, field_po))

                constant_memory_features = exp.text_tokens

                constant_ptr_value_ids, constant_unique_input_ids = vec.vectorize_ptr_in(
                        constant_memory_features, self.model.out_vocab)

                encoder_ptr_value_ids.append(
                    constant_ptr_value_ids + [self.model.out_vocab.size + len(constant_memory_features) + x
                                                for x in range(num_included_nodes)])

                program_field_ptr_value_ids = \
                    vec.vectorize_field_ptr_out(exp.program_singleton_field_tokens,
                                                exp.program_singleton_field_token_types,
                                                self.model.out_vocab, constant_unique_input_ids,
                                                max_memory_size=len(constant_memory_features),
                                                schema=schema_graph,
                                                num_included_nodes=num_included_nodes)
                decoder_ptr_value_ids.append(program_field_ptr_value_ids)

            else:
                    encoder_ptr_input_ids = [exp.ptr_input_ids for exp in mini_batch]
                    encoder_ptr_value_ids = [exp.ptr_value_ids for exp in mini_batch]
                    decoder_ptr_value_ids = [exp.program_text_and_field_ptr_value_ids for exp in mini_batch] \
                        if self.model.training else None
                    primary_key_ids = [exp.primary_key_ids for exp in mini_batch]
                    foreign_key_ids = [exp.foreign_key_ids for exp in mini_batch]
                    field_type_ids = [exp.field_type_ids for exp in mini_batch]
                    table_masks = [exp.table_masks for exp in mini_batch]

                    table_pos, table_field_scope = schema_graph.get_table_scopes(schema_graph.num_nodes)
                    table_positions.append(table_pos)
                    table_field_scopes.append(table_field_scope)
                    if self.params.read_picklist:
                        transformer_output_value_masks.append(exp.transformer_output_value_mask)

        encoder_ptr_input_ids = ops.pad_batch(encoder_ptr_input_ids, self.model.in_vocab.pad_id)
        encoder_ptr_value_ids = ops.pad_batch(encoder_ptr_value_ids, self.model.in_vocab.pad_id)

        schema_memory_masks = ops.pad_batch(schema_memory_masks, pad_id=0) \
            if (self.params.use_pred_tables and not self.model.training) else (None, None)

        decoder_ptr_value_ids = ops.pad_batch(decoder_ptr_value_ids, self.model.out_vocab.pad_id) \
            if self.model.training else None

        primary_key_ids = ops.pad_batch(primary_key_ids, self.model.in_vocab.pad_id)
        foreign_key_ids = ops.pad_batch(foreign_key_ids, self.model.in_vocab.pad_id)
        field_type_ids = ops.pad_batch(field_type_ids, self.model.in_vocab.pad_id)
        table_masks = ops.pad_batch(table_masks, pad_id=0)

        transformer_output_value_masks = ops.pad_batch(transformer_output_value_masks, pad_id=0, dtype=torch.uint8) \
            if self.params.read_picklist else (None, None)

        if not self.model.training:
            table_positions = ops.pad_batch(table_positions, pad_id=-1) \
                if self.params.process_sql_in_execution_order else (None, None)
            table_field_scopes = ops.pad_batch_2D(table_field_scopes, pad_id=0) \
                if self.params.process_sql_in_execution_order else (None, None)

        graphs = None

        return encoder_input_ids, decoder_input_ids, encoder_ptr_input_ids, encoder_ptr_value_ids, \
                decoder_ptr_value_ids, transformer_output_value_masks, schema_memory_masks, graphs, \
                (primary_key_ids, foreign_key_ids, field_type_ids, table_masks, table_positions,
                table_field_scopes, field_table_pos), table_samples

    def get_text_masks(self, encoder_input_ids):
        return encoder_input_ids[1]

    def get_schema_masks(self, encoder_input_ptr_ids, transformer_output_masks=None):
        if transformer_output_masks is not None:
            encoder_input_ptr_ids, _ = ops.batch_binary_lookup(
                encoder_input_ptr_ids, transformer_output_masks, pad_value=self.model.in_vocab.pad_id)
        if self.params.use_typed_field_markers:
            schema_masks = (encoder_input_ptr_ids == self.model.tu.table_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.text_field_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.number_field_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.time_field_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.boolean_field_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.other_field_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.asterisk_marker_id)
        else:
            schema_masks = (encoder_input_ptr_ids == self.model.tu.table_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.field_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.primary_key_marker_id) | \
                           (encoder_input_ptr_ids == self.model.tu.asterisk_marker_id)
        return schema_masks

    def initialize_module(self, model, method='xavier'):
        print('Model initialization ({})'.format(method))
        print('--------------------------')
        num_display = 500
        count = 0
        if method == 'xavier':
            for name, param in model.named_parameters():
                if 'bert_embedding' in name:
                    print('{} (skipped)'.format(name))
                    continue
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    if count < num_display:
                        print(name)
                elif ('weight' in name or name.endswith('embeddings')) and 'norm' not in name:
                    nn.init.xavier_normal_(param)
                    if count < num_display:
                        print('{} done'.format(name))
                count += 1
        if count >= num_display:
            print('...')
        print('--------------------------')


    def post_process_nn_output(self, idx, dataset_id, example, decoder_outputs, schema=None,
                               text_ptr_weights=None, p_pointers=None, table_po=None, field_po=None, verbose=False):
        decoder_output = ops.var_to_numpy(decoder_outputs[idx])
        out_tokens = self.de_vectorize(decoder_output, self.model.out_vocab, example.text_ptr_values, schema,
                                        table_po=table_po, field_po=field_po, post_process=self.output_post_process, return_tokens=True)
        if self.params.no_join_condition:
            assert(schema is not None)
            try:
                out_tokens = moz_sp.add_join_condition(out_tokens, schema)
            except ValueError as e:
                if verbose:
                    print(str(e))
                return None
        output_str = self.output_post_process(out_tokens)
        output_str = output_str.replace(self.model.out_vocab.num_token, '1').replace('<NUM>', '1')
        output_str = output_str.replace(self.model.out_vocab.str_token, '"string"').replace('<STRING>', "string")

        return output_str,

    def de_vectorize(self, vec_cpu, rev_vocab, memory, schema, table_po=None, field_po=None, post_process=None,
                           return_tokens=False):
        tokens = []
        for j in range(len(vec_cpu)):
            token_id = int(vec_cpu[j])
            if j == 0 and token_id == rev_vocab.start_id:
                continue
            if token_id == rev_vocab.eos_id or token_id == rev_vocab.pad_id:
                break
            if token_id < rev_vocab.size:
                tokens.append(rev_vocab.to_token(token_id))
            else:
                memory_pos = token_id - rev_vocab.size
                if memory_pos < len(memory):
                    tokens.append(memory[memory_pos])
                else:
                    schema_pos = memory_pos - len(memory)
                    tokens.append(schema.get_signature_by_schema_pos(schema_pos, table_po=table_po, field_po=field_po))
        if return_tokens:
            return tokens
        s = post_process(tokens)
        return s

    def get_dummy_prediction(self, schema):
        return 'SELECT * FROM {}'.format(schema.table_rev_index[0].name)

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, inputs, targets):
        target_mask = (targets != self.pad_idx)
        target_vec_mask = target_mask.unsqueeze(-1).expand_as(inputs)
        vocab_size = inputs.size(-1)
        masked_inputs = inputs[target_vec_mask].view(-1, vocab_size)
        masked_targets = targets[target_mask]
        if masked_targets.nelement() == 0:
            return 0
        loss = F.nll_loss(masked_inputs, masked_targets)
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        if loss > 1e8:
            import pdb
            pdb.set_trace()
        return loss

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class InverseSquareRootScheduler(_LRScheduler):
    """
    Code adaped from
        https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py

    Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:
        lrs = torch.linspace(params.warmup_init_lr, params.lr, params.warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
         decay_factor = params.lr * sqrt(params.warmup_updates)
    """

    def __init__(self, optimizer, warmup_init_lrs, num_warmup_steps, num_steps, target_lrs=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        if target_lrs is None:
            target_lrs = [0 for _ in self.base_lrs]
        assert(len(self.base_lrs) == len(warmup_init_lrs) == len(num_warmup_steps) == len(target_lrs))
        self.num_steps = num_steps
        self.warmup_init_lrs = warmup_init_lrs
        self.num_warmup_steps = num_warmup_steps
        self.target_lrs = target_lrs
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                    zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_bases = [(base_lr * num_warmup_step ** 0.5)
                              for base_lr, num_warmup_step in
                                zip(self.base_lrs, self.num_warmup_steps)]
        if target_lrs is None:
            self.offset_factors = [0 for _ in self.base_lrs]
        else:
            self.offset_factors = [(decay_base * self.num_steps ** -0.5 - target_lr) / self.num_steps
                                     for decay_base, target_lr in
                                        zip(self.decay_bases, self.target_lrs)]
        self.step(last_epoch + 1)

    def get_lr(self):
        return [self.update_lr(warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor)
                for warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor in
                    zip(self.warmup_init_lrs, self.num_warmup_steps, self.lr_linear_steps, self.decay_bases,
                        self.offset_factors)]

    def update_lr(self, warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor):
        num_steps = (self.last_epoch + 1)
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + num_steps * lr_linear_step
        else:
            lr = decay_base * num_steps ** -0.5 - offset_factor * num_steps
        return lr
