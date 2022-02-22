#!/usr/bin/env bash

GID=0
python gref.py --config config.json --cuda $GID
python mref.py --config config.json --cuda $GID
python inference.py --config config.json --path gref --cuda $GID
python inference.py --config config.json --path mref --cuda $GID
python btrain.py --config config.json --cuda $GID

