import json
import os
import argparse

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

def write_json(data, dsave):
	outf = open(dsave, 'w', encoding='utf-8')
	json.dump(data, outf, indent=2, ensure_ascii=False)
	outf.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/data/xx/xx/xx.log.json',
                        help='path to json files')
    args = parser.parse_args()

    path_to_json = args.path + '/'

    json_files = [path_to_json + pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    for json_dir in json_files:
        if 'LIVE' in json_dir:
            d = read_json(json_dir)
            sp = json_dir.index('LIVE')
            ep = json_dir.index('.log.json')
            renamed = json_dir[sp:ep]
            d['dialogue-id'] = renamed
            write_json(d, json_dir)


if __name__ == '__main__':
    main()