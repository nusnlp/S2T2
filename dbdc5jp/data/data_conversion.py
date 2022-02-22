import json
import os
import argparse


label2idx = {'O': 0, 'T': 1, 'X': 2}
eval_mode = 'eval'

# O = NB  "Not a breakdown"
# T = PB  "Possible breakdonw"
# X = B   "Breakdown"

def get_label_scores(label_list):
	label0_count = 0
	label1_count = 0
	label2_count = 0

	for label in label_list:
		if label2idx[label] == 0:
			label0_count += 1
		elif label2idx[label] == 1:
			label1_count += 1
		else:
			label2_count += 1

	total_labels = len(label_list)
	label0_score = label0_count*1.0/total_labels
	label1_score = label1_count*1.0/total_labels
	label2_score = label2_count*1.0/total_labels

	return [label0_score, label1_score, label2_score]

def read_json(json_dir, mode, inst_count):
	f = open(json_dir, 'r', encoding='utf-8')
	data = json.load(f)
	f.close()
	dialogue_id = data['dialogue-id']
	prev_turns = []
	json_instances = []
	jsonl_instances = []
	for turn in data['turns']:
		annotations = turn['annotations']
		if turn['speaker'] == 'S' and annotations != [] and (mode == eval_mode or annotations != []):
			inst = {}
			inst['dialogue_id'] = dialogue_id
			inst['turn_index'] = turn['turn-index']
			inst['prev_turns'] = [t for t in prev_turns]
			inst['utterance'] = turn['utterance']
			if annotations == []:
				inst['label_scores'] = [0.0, 0.0, 0.0]
			else:
				label_list = [a['breakdown'] for a in annotations]
				inst['label_scores'] = get_label_scores(label_list)

			json_instances += [inst]

		prev_turns += [turn['utterance']]

	for inst in json_instances:
		jsonl_inst = {}
		jsonl_inst['id'] = mode + '_' + str(inst_count)
		jsonl_inst['dialogue_id'] = inst['dialogue_id']
		jsonl_inst['turn_index'] = inst['turn_index']
		jsonl_inst['prev_turns'] = inst['prev_turns']
		jsonl_inst['utterance'] = inst['utterance']
		jsonl_inst['label_scores'] = inst['label_scores']

		jsonl_instances += [jsonl_inst]
		inst_count += 1


	return json_instances, jsonl_instances, dialogue_id, inst_count

def read_json_expand(json_dir):
	f = open(json_dir, 'r', encoding='utf-8')
	data = json.load(f)
	f.close()
	dialogue_id = data['dialogue-id']
	prev_turns = []
	json_instances = []

	for turn in data['turns']:
		annotations = turn['annotations']
		if annotations != []:
			for a in annotations:
				inst = {}
				inst['dialogue_id'] = dialogue_id
				inst['turn_index'] = turn['turn-index']
				inst['prev_turns'] = [t for t in prev_turns]
				inst['utterance'] = turn['utterance']
				inst['label_scores'] = [0.0, 0.0, 0.0]
				label_index = label2idx[a['breakdown']]
				inst['label_scores'][label_index] = 1.0

				json_instances += [inst]

		prev_turns += [turn['utterance']]

	return json_instances, dialogue_id




def save_json(file_to_save, out_dir):
	outf = open(out_dir, 'w', encoding='utf-8')
	json.dump(file_to_save, outf, indent=2, ensure_ascii=False)

def save_jsonl(file_to_save, out_dir):
	outf = open(out_dir, 'w', encoding='utf-8')
	for line in file_to_save:
		json.dump(line, outf, ensure_ascii=False)
		outf.write('\n')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default='test',
						help='train/dev/eval')
	parser.add_argument("--lang", type=str, default='en',
						help='en/jp')
	args = parser.parse_args()

	mode = args.mode

	lang = args.lang

	ddata = './'

	if lang == 'en':
		path_to_json = ddata + 'en/' + mode + '_all/'
	elif lang == 'jp':
		path_to_json = ddata + 'jp/' + mode + '_all/'

	else:
		print('language can only be { \'en\', \'jp\' }')
		return

	json_out_dir = ddata + '{}/'.format(lang) + mode + '.json'
	jsonl_out_dir = ddata + '{}/'.format(lang) + mode + '.jsonl'

	json_files = [path_to_json + pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
	json_save = {}
	json_save['version'] = '0.0.0'
	json_save['data'] = []

	jsonl_save = []

	inst_count = 0

	for json_dir in json_files:
		if args.mode == 'train':
			json_instances, _, _, inst_count = read_json(json_dir, mode, inst_count)
		else:
			json_instances, jsonl_instances, _, inst_count = read_json(json_dir, mode, inst_count)
			jsonl_save += jsonl_instances
		json_save['data'] += json_instances

	save_json(json_save, json_out_dir)

	if mode == 'eval' or mode == 'dev':
		save_jsonl(jsonl_save, jsonl_out_dir)


if __name__ == '__main__':
	main()

		
