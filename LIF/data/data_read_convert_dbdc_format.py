import json
import os
import argparse
import glob
import math
import copy
from nltk import tokenize

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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default='dev',
						help='train/dev/test_i/test_ii/test_iii')

	args = parser.parse_args()
	jname = args.mode
	json_path = 'dataset/{}.json'.format(jname)
	dsave_json = './{}.json'.format(jname)

	data = read_json(json_path)['data']

	CONTEXT_ID = 0
	INSTANCE_ID = 0

	insts = []
	for d in data:
		p = d['paragraphs'][0]
		context = copy.deepcopy(p['context'])
		context = tokenize.sent_tokenize(context)
		for qas in p['qas']:
			inst = {}
			inst['uid'] = INSTANCE_ID
			INSTANCE_ID += 1
			inst['context_id'] = CONTEXT_ID
			inst['context'] = context
			inst['prev_turns'] = []
			assert len(qas['prev_qs']) == len(qas['prev_ans'])
			for q, a in zip(qas['prev_qs'], qas['prev_ans']):
				inst['prev_turns'].append(q)
				inst['prev_turns'].append(a['text'])
			inst['utterance'] = copy.deepcopy(qas['candidate'])
			inst['label_scores'] = [1, 0]
			if qas['label'] == 1:
				inst['label_scores'] = [0, 1]
			insts.append(inst)
		CONTEXT_ID += 1

	insts_json = {'version': '0.0.0',
				  'data': insts}
	write_json(insts_json, dsave_json)
	# if 'dev' in jname or 'test' in jname:
	# 	write_jsonl(insts, dsave_jsonl)

	print('{} instances'.format(len(insts)))






if __name__ == '__main__':
	main()

		
