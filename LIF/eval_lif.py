import json
from argparse import ArgumentParser
from sklearn.metrics import classification_report

def load_jsonl_data(data_dir):
        data = []
        with open(data_dir, 'r', encoding='utf-8') as lf:
                for line in lf:
                        data.append(json.loads(line))

        lf.close()
        return data

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--pred', default='./saved/combined/test_i_pred.jsonl')

    args = parser.parse_args()
    data_ref = read_json(read_json(args.config)['test_i_path'])['data']
    labels_ref = [d['label_scores'][1] for d in data_ref]

    data_pred = load_jsonl_data(args.pred)
    labels_pred = [d['label_pred'] for d in data_pred]

    assert len(labels_ref) == len(labels_pred)

    report = classification_report(labels_ref, labels_pred, labels=[0, 1], digits=3, output_dict=True)
    V_P = report['1']['precision']
    V_R = report['1']['recall']
    V_f1 = report['1']['f1-score']
    macro_f1 = report['macro avg']['f1-score']

    V_P = '{:.1f}'.format(V_P * 100)
    V_R = '{:.1f}'.format(V_R * 100)
    V_f1 = '{:.1f}'.format(V_f1 * 100)
    macro_f1 = '{:.1f}'.format(macro_f1 * 100)

    print('Test / V_P / V_R / V_f1 / Macro_F1')
    print('{:<9}  {} / {} / {} / {}'.format('test', V_P, V_R, V_f1, macro_f1))


