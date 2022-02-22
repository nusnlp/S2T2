import logging
import os
from argparse import ArgumentParser
import json
import re
from tqdm import tqdm, trange
from pprint import pprint, pformat
import time
from datetime import timedelta
from shutil import copyfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
from model import PretrainedEmbedder, SMLP

from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModel, AdamW

SEED = 62
random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

def write_json(data, dsave):
	outf = open(dsave, 'w', encoding='utf-8')
	json.dump(data, outf, indent=2, ensure_ascii=False)
	outf.close()
	print('>>> write to {}'.format(dsave))

def write_jsonl(data, dsave):
	outf = open(dsave, 'w')
	for d in data:
		json.dump(d, outf)
		outf.write('\n')
	outf.close()
	print('\n+++ write to {}'.format(dsave))

def custom_tokenize(text, tokenizer):
    encoded = tokenizer.encode_plus(text,
                                    add_special_tokens=True,
                                    return_token_type_ids=True)
    return encoded['input_ids']

def check_label_scores(label_scores):
    """
    avoid return of wrong argmax when multiple max values
    """
    max_v = max(label_scores)
    for i in range(3):
        if label_scores[i] == max_v:
            scores_hard = [0, 0, 0]
            scores_hard[i] = 1
            return i, scores_hard

def summary_printout(logger, content):
    for k, v in content.items():
        logger.critical('-- {}: {}'.format(k, v))
    logger.critical('\n------------------------------\n')

def format_numbers(n, decimal=4):
    d = decimal
    ns = '{:.{}f}'.format(n, d)
    return float(ns)

def format_time(seconds):
    return str(timedelta(seconds=seconds)).split('.')[0]

def save_configs(dsave, task_config):
    write_json(task_config, dsave + '/config.json')

class Trainer():
    def __init__(self, args):
        self.args = args
        self.task_config = read_json(args.config)
        ckpt_dsave = './saved/' + 'inferenced'

        assert not os.path.exists(ckpt_dsave)
        os.makedirs(ckpt_dsave)
        self.ckpt_dsave = ckpt_dsave
        print('{} created'.format(ckpt_dsave))
        self.test_dsave = ckpt_dsave + '/eval_pred.jsonl'
        self.test_ii_dsave = ckpt_dsave + '/test_ii_pred.jsonl'
        self.test_iii_dsave = ckpt_dsave + '/test_iii_pred.jsonl'

        self.device = torch.device('cuda:{}'.format(args.cuda))

        save_configs(ckpt_dsave, self.task_config)

        ckpt_saved = self.args.model

        self.loaded = torch.load(ckpt_saved, map_location='cuda:{}'.format(args.cuda))

        self.model_name = 'roberta-large'

        self.lmodel = PretrainedEmbedder(self.model_name)
        emb_dim = self.lmodel.model.config.hidden_size
        self.layer_cls = SMLP(emb_dim, 2)

        # load from saved checkpoint
        self.lmodel.load_state_dict(self.loaded['lmodel'])
        self.layer_cls.load_state_dict(self.loaded['layer_cls'])

        del self.loaded

        self.vsoftmax = torch.nn.Softmax(dim=1)

        bsz = 128

        self.optimizer = None

        bsz_test = 64

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id

        train_data = []
        dev_data = self.load_data(read_json(self.task_config['dev_data_path'])['data'], tokenizer)
        test_data = self.load_data(read_json(self.task_config['test_i_path'])['data'], tokenizer)
        test_ii_data = self.load_data(read_json(self.task_config['test_ii_path'])['data'], tokenizer)
        test_iii_data = self.load_data(read_json(self.task_config['test_iii_path'])['data'], tokenizer)

        if args.debug:
            train_data = train_data[-100:]
            dev_data = dev_data[-100:]
            test_data = test_data[-100:]
            self.epochs = 5
        self.dev_data = [dev_data[i:i + bsz_test] for i in range(0, len(dev_data), bsz_test)]
        self.test_data = [test_data[i:i + bsz_test] for i in range(0, len(test_data), bsz_test)]
        self.test_ii_data = [test_ii_data[i:i + bsz_test] for i in range(0, len(test_ii_data), bsz_test)]
        self.test_iii_data = [test_iii_data[i:i + bsz_test] for i in range(0, len(test_iii_data), bsz_test)]

    def load_data(self, data, tokenizer, train=False):
        print('... loading data ... train={}'.format(train))
        datap = []

        for inst in data:
            uid = inst['uid']
            prev_turns = inst['prev_turns']
            label_scores = inst['label_scores']
            label_scores_hard = label_scores
            label_gold = 0
            if label_scores == [0, 1]:
                label_gold = 1

            metadata = {}
            metadata['uid'] = uid
            metadata['context_id'] = inst['context_id']
            metadata['context'] = inst['context']
            metadata['prev_turns'] = prev_turns
            metadata['utterance'] = inst['utterance']

            context_turns_selected = inst['context']
            c_text = ' '.join(context_turns_selected)
            # t_text = inst['utterance'].strip()
            t_text = inst['utterance']

            t_text_length = len(tokenizer.encode(t_text)) - 1

            prev_turns_selected = prev_turns
            h_text = ' '.join(prev_turns_selected)

            combined_text = c_text + ' ' + h_text + ' ' + tokenizer.sep_token + ' ' + t_text

            max_seq_len = 256

            if 'id' in inst.keys():
                combined_tokens = inst['input_ids']
            else:
                while len(combined_text.split()) > max_seq_len:
                    if len(context_turns_selected) > 0:
                        context_turns_selected = context_turns_selected[:-1]
                    else:
                        prev_turns_selected = prev_turns_selected[1:]
                    c_text = ' '.join(context_turns_selected)
                    h_text = ' '.join(prev_turns_selected)
                    combined_text = c_text + ' ' + h_text + ' ' + tokenizer.sep_token + ' ' + t_text
                combined_tokens = custom_tokenize(combined_text, tokenizer)

                while len(combined_tokens) > max_seq_len:
                    if len(context_turns_selected) > 0:
                        context_turns_selected = context_turns_selected[:-1]
                    else:
                        prev_turns_selected = prev_turns_selected[1:]
                    c_text = ' '.join(context_turns_selected)
                    h_text = ' '.join(prev_turns_selected)
                    combined_text = c_text + ' ' + h_text + ' ' + tokenizer.sep_token + ' ' + t_text
                    combined_tokens = custom_tokenize(combined_text, tokenizer)
                # print('{} > {}'.format(len(combined_tokens), max_seq_len))

            # print('{} {}'.format(uid, len(combined_tokens)))

            datap_inst = {'combined_tokens': combined_tokens,
                          'label_scores': label_scores,
                          'label_scores_hard': label_scores_hard,
                          'label_gold': label_gold,
                          't_text_length': t_text_length,
                          'metadata': metadata
                          }
            datap.append(datap_inst)
        if train:
            datap = random.sample(datap, len(datap))
        return datap

    def output_test_results(self, test_data):
        test_samples = [d['label_probs'][1] for d in test_data]
        test_labels = [d['label_gold'] for d in test_data]
        test_preds = [1 if score > 0.5 else 0 for score in test_samples]
        report = classification_report(test_labels, test_preds, labels=[0, 1], digits=3, output_dict=True)
        V_P = report['1']['precision']
        V_R = report['1']['recall']
        V_f1 = report['1']['f1-score']
        macro_f1 = report['macro avg']['f1-score']

        V_P = '{:.1f}'.format(V_P * 100)
        V_R = '{:.1f}'.format(V_R * 100)
        V_f1 = '{:.1f}'.format(V_f1 * 100)
        macro_f1 = '{:.1f}'.format(macro_f1 * 100)

        return V_P, V_R, V_f1, macro_f1

    def log_test_results(self, test_data, test_name):
        V_P, V_R, V_f1, macro_f1 = self.output_test_results(test_data)
        self.logger.critical('{:<9}  {} / {} / {} / {}'.format(test_name, V_P, V_R, V_f1, macro_f1))

    def run_process(self):
        global_start_time = time.time()
        ckpt_dsave = self.ckpt_dsave
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(ckpt_dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)
        self.logger = logger
        logger.critical('------ task config ------')
        logger.critical('-- cuda: {}'.format(self.args.cuda))
        summary_printout(logger, self.task_config)

        self.lmodel = self.lmodel.to(self.device)
        self.layer_cls = self.layer_cls.to(self.device)

        test_out, test_acc = self.run_subprocess(self.test_data, test=True)
        test_ii_out, test_ii_acc = self.run_subprocess(self.test_ii_data, test=True)
        test_iii_out, test_iii_acc = self.run_subprocess(self.test_iii_data, test=True)

        logger.critical('[Set] / V_P / V_R / V_f1 / Macro_F1')
        self.log_test_results(test_out, 'Test-I')
        self.log_test_results(test_ii_out, 'Test-II')
        self.log_test_results(test_iii_out, 'Test-III')

        write_jsonl(test_out, self.test_dsave)
        write_jsonl(test_ii_out, self.test_ii_dsave)
        write_jsonl(test_iii_out, self.test_iii_dsave)

    def run_subprocess(self, data, train=False, dev=False, test=False, gen=False):
        self.lmodel.eval()
        self.layer_cls.eval()

        device = self.device
        run_correct = 0
        seen = 0
        inst_count = 0

        acc = 0.0
        # train
        pred_out = []
        val_iteration = len(data)
        self.dataiter = iter(data)

        iters = trange(val_iteration)
        for batch_idx in iters:
            batch = next(self.dataiter)

            combined_tokens_b = pad_sequence([torch.LongTensor(d['combined_tokens']) for d in batch], batch_first=True,
                                             padding_value=self.pad_token_id).to(device)
            attn_mask_b = pad_sequence([torch.FloatTensor([1]*len(d['combined_tokens'])) for d in batch], batch_first=True,
                                        padding_value=0).to(device)

            label_scores = torch.FloatTensor([d['label_scores'] for d in batch]).to(device)
            label_scores_hard = torch.FloatTensor([d['label_scores_hard'] for d in batch])
            labels = label_scores_hard.argmax(dim=1).to(device)

            label_gold = [d['label_gold'] for d in batch]

            with torch.no_grad():
                combined_out = self.lmodel(combined_tokens_b, attn_mask_b)[0][:, 0, :]
                x = self.layer_cls(combined_out)
                output = self.vsoftmax(x)
                _, pred_label = output.max(1)

            seen += len(batch)

            run_correct += (output.detach().cpu().argmax(dim=1) == label_scores_hard.argmax(dim=1)).sum().item()
            acc = run_correct/seen
            iters.set_description('[test] acc_cls:{:.4f}'.format(acc), refresh=False)
            assert len(batch) == output.shape[0] == pred_label.shape[0] == \
                   label_scores.shape[0] == len(label_gold)
            for d, dist, cls, lp_gold, lgold_hard in zip(batch, output, pred_label, label_scores, label_gold):
                pred_out.append({'id': 'eval_' + str(inst_count),
                                 'uid': d['metadata']['uid'],
                                 'context_id': d['metadata']['context_id'],
                                 'label_p_gold': lp_gold.tolist(),
                                 'label_gold': lgold_hard,
                                 'label_probs': dist.tolist(),
                                 'label_pred': int(cls)
                                 })

                inst_count += 1

        return pred_out, acc

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--model', default='./model.pt.tar')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run_process()
