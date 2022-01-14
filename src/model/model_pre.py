from src.data_processor import data_loader
from src.data_processor import tokenizers
from src.model.schema_graph import SchemaGraphs
from src.data_processor.schema_loader import load_schema_graphs
from src.data_processor.processors.data_processor_spider import preprocess_example
import src.data_processor.processor_utils as data_utils
import os
import time
import torch
import pickle




def demo_preprocess(args, example, vocabs=None, schema_graph=None):
    text_tokenize, program_tokenize, post_process, tu = tokenizers.get_tokenizers(args)
    if not schema_graph:
        schema_graphs = load_schema_graphs(args)
        schema_graph = schema_graphs.get_schema(example.db_id)
    schema_graph.lexicalize_graph(tokenize=text_tokenize, normalized=(args.model_id == 2))
    preprocess_example('test', example, args, {}, text_tokenize, program_tokenize, post_process, tu, schema_graph, vocabs)

class Text2SQLWrapper(object):
    def __init__(self, args, schema, sps):
        self.args = args
        self.text_tokenize, _, _, self.tu = tokenizers.get_tokenizers(args)

        # Vocabulary
        self.vocabs = data_loader.load_vocabs(args)

        # Text-to-SQL model
        self.semantic_parsers = []
        self.model_ensemble = None

        
        checkpoint_path = os.path.join(args.model_dir, 'model-best.{}.tar'.format(16))
        sps.model.schema_graphs = SchemaGraphs()
        sps.model.eval()
        sps.model.training = False
        sps.model.load_checkpoint(checkpoint_path)
        sps.model.cuda()
        self.semantic_parsers = sps
        self.model_ensemble = sps

        if schema is not None:
            self.add_schema(schema)

        # When generating SQL in execution order, cache reordered SQLs to save time
        if args.process_sql_in_execution_order:
            self.pred_restored_cache = self.load_pred_restored_cache()
        else:
            self.pred_restored_cache = None

    def translate(self, example):
        """
        :param text: natural language question
        :return: SQL query corresponding to the input question
        """
        start_time = time.time()
        output = self.semantic_parsers.inference([example], restore_clause_order=self.args.process_sql_in_execution_order,
                                                    pred_restored_cache=self.pred_restored_cache,
                                                    model_ensemble=self.model_ensemble, verbose=False)

        if len(output['pred_decoded'][0]) > 1:
            pred_sql = output['pred_decoded'][0][0]
        else:
            pred_sql = None
        print('inference time: {:.2f}s'.format(time.time() - start_time))
        return pred_sql

    def process(self, text, schema_name, verbose=False):
        schema = self.semantic_parsers.model.schema_graphs[schema_name]
        start_time = time.time()
        example = data_utils.Text2SQLExample(data_utils.OTHERS, schema.name,
                                             db_id=self.semantic_parsers.model.schema_graphs.get_db_id(schema.name))
        example.text = text
        demo_preprocess(self.args, example, self.vocabs, schema)
        print('data processing time: {:.2f}s'.format(time.time() - start_time))

        translatable, confuse_span, replace_span = True, None, None

        sql_query = None
        if translatable:
            print('Translatable!')
            sql_query = self.translate(example)
            if verbose:
                print('Text: {}'.format(text))
                print('SQL: {}'.format(sql_query))
                print()
        else:
            print('Untranslatable!')

        output = dict()
        output['translatable'] = translatable
        output['sql_query'] = sql_query
        output['confuse_span'] = confuse_span
        output['replace_span'] = replace_span
        return output

    def add_schema(self, schema):
        schema.lexicalize_graph(tokenize=self.text_tokenize)
        if schema.name not in self.semantic_parsers.model.schema_graphs.db_index:
            self.semantic_parsers.model.schema_graphs.index_schema_graph(schema)

    def schema_exists(self, schema_name):
        return schema_name in self.semantic_parsers.model.schema_graphs.db_index

    def load_pred_restored_cache(self):
        split = 'test' if self.args.test else 'dev'
        pred_restored_cache_path = os.path.join(
            self.args.model_dir, '{}.eo.pred.restored.pkl'.format(split))
        if os.path.exists(pred_restored_cache_path):
            with open(pred_restored_cache_path, 'rb') as f:
                pred_restored_cache = pickle.load(f)
                pred_restored_cache_size = sum([len(pred_restored_cache[k]) for k in pred_restored_cache])
                print('{} pre-computed prediction order reconstruction cached'.format(pred_restored_cache_size))
            return pred_restored_cache
        else:
            return dict()