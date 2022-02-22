### Data Downloading & Preprocessing  ###

Data (en+jp) can be retrieved from following sites:

[DBDC1](https://dbd-challenge.github.io/dbdc3/datasets),
[DBDC2](https://dbd-challenge.github.io/dbdc3/datasets),
[DBDC3](https://dbd-challenge.github.io/dbdc3/data/),
[DBDC4](https://sites.google.com/site/dialoguebreakdowndetection4/datasets?authuser=0),
[DBDC5](http://workshop.colips.org/wochat/@iwsds2020/shared.html)

Go to `data` directory and download the required data files.

Run the script of data downloading and preprocessing:
```bash
bash get_data.sh
```
The processed data files will be located at `/dbdc5en/data/en` and `/dbdc5jp/data/jp` for English track and Japanese track, respectively.

### Training ###
Use the training script to run training process
```bash
bash run.sh
```

### Prediction ###
To run prediction from a trained model (skip this if running `Training`)
```bash
python inference_t.py --config config.json --model [path_to_trained_model]
```

### Evaluation ###
First, convert prediction file to separate json files:
```bash
cd evaluation
python convert_predictions_to_files.py --eval_file ../saved/btrain/eval_pred.jsonl
```
Then run the evaluation script:
```bash
python2 evaluation/eval.py -t 0.0 -p ../data/jp/eval_all/ -o pred_label_files/labels_btrain