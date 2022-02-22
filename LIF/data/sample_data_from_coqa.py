import json
import os
import argparse
import glob
import math
import copy
import random
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

	json_path = './coqa-train-v1.0.json'
	data = read_json(json_path)['data']

	context_list = []
	filename_list = []
	source_list = []
	context_id_list = []

	datap = []

	for d in data:
		context_id_list.append(d['id'])
		source_list.append(d['source'])
		filename_list.append(d['filename'])
		context_list.append(tokenize.sent_tokenize(d['story']))

		q_len = len(d['questions'])
		a_len = len(d['answers'])

		assert(q_len == a_len)

		qas_hist = []
		for i in range(q_len - 1):
			inst = {}
			qas_hist.append(d['questions'][i]['input_text'])
			qas_hist.append(d['answers'][i]['input_text'])
			inst['uid'] = d['id'] + '_' + str(i)
			inst['context_i'] = len(context_list) - 1
			inst['prev_turns'] = qas_hist[:]
			inst['utterance'] = d['questions'][i+1]['input_text']

			datap.append(inst)

	print(len(datap))

	sampled = datap

	insts = []
	for s in sampled:
		inst = {}
		ctx_id = s['context_i']
		inst['uid'] = s['uid']
		inst['context_id'] = context_id_list[ctx_id]
		inst['source'] = source_list[ctx_id]
		inst['filename'] = filename_list[ctx_id]
		inst['context'] = context_list[ctx_id]
		inst['prev_turns'] = s['prev_turns']
		inst['utterance'] = s['utterance']
		inst['label_scores'] = [1, 0]

		insts.append(inst)

	d_samples= {'version': '1.0', 'data': insts[:]}
	dsave_sample = 'coqa_data.json'


	dsave = './'
	assert not os.path.exists(dsave + dsave_sample)
	write_json(d_samples, dsave + dsave_sample)




if __name__ == '__main__':
	main()

		
