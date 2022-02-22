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
        ckpt_dsave = './saved/' + 'btrain'

        if self.args.prefix != '':
            ckpt_dsave = './saved/' + \
                         self.args.prefix + '_' + \
                         'btrain'

        assert not os.path.exists(ckpt_dsave)
        os.makedirs(ckpt_dsave)
        self.ckpt_dsave = ckpt_dsave
        print('{} created'.format(ckpt_dsave))
        self.test_dsave = ckpt_dsave + '/eval_pred.jsonl'
        self.test_ii_dsave = ckpt_dsave + '/test_ii_pred.jsonl'
        self.test_iii_dsave = ckpt_dsave + '/test_iii_pred.jsonl'

        self.device = torch.device('cuda:{}'.format(args.cuda))

        self.task_config['tokenized_udata'] = './saved/' + \
                                              'inf_gref/' \
                                              'self_train_generated.json'

        save_configs(ckpt_dsave, self.task_config)

        ckpt_path = './saved/' + \
                    'gref/'

        ckpt_saved = ckpt_path + 'model.pt.tar'

        self.loaded = torch.load(ckpt_saved, map_location='cuda:{}'.format(args.cuda))

        self.model_name = 'roberta-large'

        self.lmodel = PretrainedEmbedder(self.model_name)
        emb_dim = self.lmodel.model.config.hidden_size
        self.layer_cls = SMLP(emb_dim, 2)

        # load from saved checkpoint
        self.lmodel.load_state_dict(self.loaded['lmodel'])

        del self.loaded

        self.vdropout = nn.Dropout(p=0.3)
        self.mse_loss = torch.nn.MSELoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy = torch.nn.CrossEntropyLoss()
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

        unlabelled_data = self.load_tokenized_unlabeled_data(read_json(self.task_config['tokenized_udata'])['data'])

        if args.debug:
            train_data = train_data[-100:]
            dev_data = dev_data[-100:]
            test_data = test_data[-100:]
            self.epochs = 5

        self.unlabelled_data = [unlabelled_data[i:i + bsz_test] for i in range(0, len(unlabelled_data), bsz_test)]
        self.dev_data = [dev_data[i:i + bsz_test] for i in range(0, len(dev_data), bsz_test)]
        self.test_data = [test_data[i:i + bsz_test] for i in range(0, len(test_data), bsz_test)]

        self.test_ii_data = [test_ii_data[i:i + bsz_test] for i in range(0, len(test_ii_data), bsz_test)]
        self.test_iii_data = [test_iii_data[i:i + bsz_test] for i in range(0, len(test_iii_data), bsz_test)]

        masked_ref_data_path = './saved/' + \
                               'inf_mref/' + \
                               'self_train_generated.json'

        self.masked_ref_data = read_json(masked_ref_data_path)['data']
        try:
            os.remove('./saved/gref/model.pt.tar')
        except:
            pass
        try:
            os.remove('./saved/mref/model.pt.tar')
        except:
            pass


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

    def replace_with_mask_tokens(self, seq, p=0):
        if len(seq) <= 2:
            return seq
        seq = torch.LongTensor(seq)
        seq_mask = torch.rand(seq.shape[0]-2).lt(p)
        seq[1:-1][seq_mask==True] = self.tokenizer.mask_token_id
        seq = seq.tolist()
        return seq

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

    def load_tokenized_unlabeled_data(self, data):
        datap = []
        for inst in data:
            uid = inst['uid']
            prev_turns = inst['prev_turns']
            # label_scores = torch.FloatTensor(inst['label_scores'])
            label_scores = inst['label_scores']
            label_gold = 0
            if label_scores == [0, 1]:
                label_gold = 1

            combined_tokens = inst['input_ids']

            metadata = {}
            metadata['uid'] = uid
            metadata['context_id'] = inst['context_id']
            metadata['context'] = inst['context_id']
            metadata['prev_turns'] = prev_turns
            metadata['utterance'] = inst['utterance']

            metadata['input_ids'] = combined_tokens

            t_text = inst['utterance']
            t_text_length = len(self.tokenizer.encode(t_text)) - 1

            datap_inst = {'combined_tokens': combined_tokens,
                          'label_scores': label_scores,
                          'label_gold': label_gold,
                          't_text_length': t_text_length,
                          'metadata': metadata
                          }

            datap.append(datap_inst)

        return datap

    def gen_labels(self, data):
        gen_out, _ = self.run_subprocess(data, gen=True)
        gen_out_wrapper = {'version': '0.0.1', 'data': gen_out}
        write_json(gen_out_wrapper, self.ckpt_dsave + '/self_train_generated.json')

        train_data_path = self.task_config['train_data_path']
        combined_out_wrapper = read_json(train_data_path)
        combined_out_wrapper['data'] += gen_out
        write_json(combined_out_wrapper, self.ckpt_dsave + '/self_train_combined.json')

        return gen_out

    def merge_label_scores(self, list1, list2, iter=1, total_iter=5):
        import copy
        assert len(list1) == len(list2)
        datap = []
        for d1, d2 in zip(list1, list2):
            assert d1['input_ids'] == d2['input_ids']
            s1 = d1['label_scores']
            s2 = d2['label_scores']
            ss = [0.5 * (1 - (iter/total_iter)) * s1[i] + 0.5 * (1 + (iter/total_iter)) * s2[i] for i in range(len(s1))]
            ssum = sum(ss)
            ss_merged = [s / ssum for s in ss]

            dd = copy.deepcopy(d2)
            dd['label_scores'] = ss_merged
            datap.append(dd)

        labeled_data_path = self.task_config['train_data_path']
        combined_out_data = read_json(labeled_data_path)['data']
        combined_out_data += datap

        return combined_out_data

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

        param_groups = [{'params': self.layer_cls.parameters()}]

        lr = 2e-6

        ITERATIONS = 5
        epochs = 5
        patience = 0
        patience_max = 10

        self.optimizer = AdamW(params=param_groups, lr=lr, weight_decay=0.01)

        ckpt_best = None
        dev_acc_best = 0.0
        epoch_best = 0
        iter_best = 0
        self.gen_out = []

        for iter in trange(ITERATIONS, desc='ITER'):
            if iter >= 0:
                if iter > 0:
                    self.lmodel.load_state_dict(ckpt_best['lmodel'])
                    self.layer_cls.load_state_dict(ckpt_best['layer_cls'])
                    self.lmodel = self.lmodel.to(self.device)
                    self.layer_cls = self.layer_cls.to(self.device)

                    self.gen_out = self.gen_labels(self.unlabelled_data)

                bsz = 128
                if self.gen_out == []:
                    self.gen_out = read_json(self.task_config['tokenized_udata'])['data']
                train_data_out = \
                    self.merge_label_scores(self.masked_ref_data, self.gen_out, iter=iter, total_iter=ITERATIONS)
                train_data = self.load_data(train_data_out, self.tokenizer, train=True)
                self.train_data = [train_data[i:i + bsz] for i in range(0, len(train_data), bsz)]

            summary = []

            for epoch in trange(epochs, desc='epoch'):
                epoch_start_time = time.time()

                _, train_acc, loss_terms = self.run_subprocess(self.train_data, train=True)
                dev_out, dev_acc, valid_loss_terms = self.run_subprocess(self.dev_data, dev=True)

                test_out, test_acc = self.run_subprocess(self.test_data, test=True)
                test_ii_out, test_ii_acc = self.run_subprocess(self.test_ii_data, test=True)
                test_iii_out, test_iii_acc = self.run_subprocess(self.test_iii_data, test=True)

                V_P, V_R, V_f1, macro_f1 = self.output_test_results(dev_out)
                dev_acc = float(macro_f1)

                test_save_local = self.ckpt_dsave + '/LOCAL_eval_{}_{}.jsonl'.format(iter, epoch)
                write_jsonl(test_out, test_save_local)

                logger.critical('----- iter {} epoch {} -----'.format(iter, epoch))
                logger.critical('[Set] / V_P / V_R / V_f1 / Macro_F1')
                self.log_test_results(dev_out, 'Dev')

                if dev_acc > dev_acc_best:
                    dev_acc_best = dev_acc
                    iter_best = iter
                    epoch_best = str(iter) + '_' + str(epoch)
                    ckpt_best = {'iter': iter,
                                 'epoch': epoch_best,
                                 'lmodel': self.keep_state_dict(self.lmodel.cpu().state_dict()),
                                 'layer_cls': self.keep_state_dict(self.layer_cls.cpu().state_dict()),
                                 'train_acc': train_acc,
                                 'dev_acc': dev_acc_best,
                                 'test_acc': test_acc
                                 }

                    self.lmodel = self.lmodel.to(self.device)
                    self.layer_cls = self.layer_cls.to(self.device)

                    write_jsonl(test_out, self.test_dsave)
                    write_jsonl(test_ii_out, self.test_ii_dsave)
                    write_jsonl(test_iii_out, self.test_iii_dsave)

                    patience = 0
                else:
                    if iter == 0 and epoch == 0:
                        patience = 0
                    else:
                        patience += 1

                summary.append({'iter': iter,
                                'epoch': epoch,
                                'epoch_best': epoch_best,
                                'train_acc': train_acc,
                                'dev_acc': dev_acc,
                                'test_acc': test_acc,
                                'epoch_time': format_time(time.time() - epoch_start_time)
                                })
                logger.critical('\n\n------ summary: iter {} epoch {} ----\n'.format(iter, epoch))
                logger.critical('-- train loss ---> {}'.format(loss_terms))
                logger.critical('-- valid loss ---> {}'.format(valid_loss_terms))
                summary_printout(logger, summary[-1])

                if patience == patience_max or (iter == ITERATIONS - 1 and epoch == epochs - 1):
                    torch.save(ckpt_best, ckpt_dsave + '/model.pt.tar')
                    if patience == patience_max:
                        logger.critical('best epoch: {}. patience {} reached.'.format(epoch_best, patience_max))
                    else:
                        logger.critical('best epoch: {}.'.format(epoch_best))
                    logger.critical('------ training summary ------')
                    logger.critical('total epochs: {}'.format(epoch))
                    logger.critical('total time: {}'.format(format_time(time.time() - global_start_time)))
                    logger.critical('model directory: {}'.format(ckpt_dsave))
                    logger.critical('------------------------------')
                    logger.critical('best ckpt saved: model.pt.tar')
                    break
            if patience == patience_max:
                break

    def keep_state_dict(self, state_dict):
        import copy
        return copy.deepcopy(state_dict)

    def run_subprocess(self, data, train=False, dev=False, test=False, gen=False):
        if train:
            self.lmodel.eval()
            self.layer_cls.train()
        else:
            self.lmodel.eval()
            self.layer_cls.eval()

        device = self.device
        train_loss = 0.0
        valid_loss_CE = 0.0
        run_correct = 0
        seen = 0
        inst_count = 0
        loss_terms = {}

        sum_CE = 0.0
        sum_SCL = 0.0

        acc = 0.0
        # train
        pred_out = []
        val_iteration = 600
        if train == False:
            val_iteration = len(data)
            self.dataiter = iter(data)

        iters = trange(val_iteration)
        for batch_idx in iters:
            if train:
                try:
                    batch = next(self.trainiter)
                except:
                    self.trainiter = iter(data)
                    batch = next(self.trainiter)
            else:
                batch = next(self.dataiter)

            if train:
                self.optimizer.zero_grad()
            combined_tokens_b = pad_sequence([torch.LongTensor(d['combined_tokens']) for d in batch], batch_first=True,
                                             padding_value=self.pad_token_id).to(device)
            attn_mask_b = pad_sequence([torch.FloatTensor([1]*len(d['combined_tokens'])) for d in batch], batch_first=True,
                                        padding_value=0).to(device)

            if not gen:
                label_scores = torch.FloatTensor([d['label_scores'] for d in batch]).to(device)
                label_scores_hard = torch.FloatTensor([d['label_scores_hard'] for d in batch])
                labels = label_scores_hard.argmax(dim=1).to(device)

                label_gold = [d['label_gold'] for d in batch]

            t_len_b = [d['t_text_length'] for d in batch]


            if not train:
                with torch.no_grad():
                    combined_out = self.lmodel(combined_tokens_b, attn_mask_b)[0][:, 0, :]
                    x = self.layer_cls(combined_out)
                    output = self.vsoftmax(x)
                    _, pred_label = output.max(1)
            if train:
                with torch.no_grad():
                    combined_out = self.lmodel(combined_tokens_b, attn_mask_b)[0][:, 0, :]

                x = self.layer_cls(combined_out)
                output = self.vsoftmax(x)

                L_SCL = self.SCL_loss(F.normalize(combined_out, dim=1), labels)

            seen += len(batch)
            if train:
                p1 = 1.0
                p2 = 0.1
                L_CE = self.cross_entropy(x, labels)

                sum_CE += L_CE.item() * len(batch)
                sum_SCL += L_SCL.item() * len(batch)

                avg_CE = sum_CE / seen
                avg_SCL = sum_SCL / seen

                loss = p1 * L_CE + p2 * L_SCL

                loss.backward()
                train_loss += (loss.item() * len(batch))
                avg_loss = train_loss / seen
                self.optimizer.step()

                loss_terms = {'CE': format_numbers(avg_CE),
                              'SCL': format_numbers(avg_SCL),
                              'AggAvg': format_numbers(avg_loss)}

            if dev:
                with torch.no_grad():
                    L_CE = self.cross_entropy(x, labels)

                    valid_loss_CE += L_CE.item() * len(batch)

                    valid_loss_CE_avg = valid_loss_CE / seen

                    loss_terms = {'CE': format_numbers(valid_loss_CE_avg)}

            if not gen:
                run_correct += (output.detach().cpu().argmax(dim=1) == label_scores_hard.argmax(dim=1)).sum().item()
                acc = run_correct/seen

            if train:
                iters.set_description('CE:{:.3f}|SCL:{:.3f}|Acc:{:.4f}'.
                                          format(avg_CE, avg_SCL, acc), refresh=False)
            if dev:
                iters.set_description('[dev] acc_cls:{:.4f}'.format(acc), refresh=False)
            if test:
                iters.set_description('[test] acc_cls:{:.4f}'.format(acc), refresh=False)


            if dev or test:
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

            if gen:
                assert len(batch) == output.shape[0]
                for d, dist in zip(batch, output):
                    pred_out.append({'id': 'gen_' + str(inst_count),
                                     'uid': d['metadata']['uid'],
                                     'context_id': d['metadata']['context_id'],
                                     'context': d['metadata']['context'],
                                     'prev_turns': d['metadata']['prev_turns'],
                                     'utterance': d['metadata']['utterance'],
                                     'label_scores': dist.tolist(),
                                     'input_ids': d['metadata']['input_ids']
                                     })

                    inst_count += 1

        if train or dev:
            return pred_out, acc, loss_terms
        else:
            return pred_out, acc

    def SCL_loss(self, features, labels, tau=1.0, tau_base=1.0):
        labels = labels.contiguous().view(-1, 1)

        mask_s = torch.eq(labels, labels.T).float().fill_diagonal_(0)
        mask_s = mask_s.to(self.device)

        sims = torch.div(features.mm(features.T), tau)

        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()

        logits_mask = torch.ones_like(mask_s).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        denom = mask_s.sum(1)
        denom[denom==0] = 1

        mean_log_prob_pos = (mask_s * log_prob).sum(1) / denom

        loss = - (tau/tau_base) * mean_log_prob_pos

        loss = loss.mean()

        return loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run_process()
