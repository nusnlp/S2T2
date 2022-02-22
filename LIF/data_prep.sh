#!/usr/bin/env bash

DATA_DIR="data"
cd $DATA_DIR
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun" -O lif_v1.zip && rm -rf /tmp/cookies.txt
unzip lif_v1.zip
rm lif_v1.zip
python data_read_convert_dbdc_format.py --mode train
python data_read_convert_dbdc_format.py --mode dev
python data_read_convert_dbdc_format.py --mode test_i
python data_read_convert_dbdc_format.py --mode test_ii
python data_read_convert_dbdc_format.py --mode test_iii
rm -rf dataset
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
python sample_data_from_coqa.py
rm coqa-train-v1.0.json
cd ..