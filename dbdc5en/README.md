### Data ###
Refer to data processing in `/dbdc5jp` directory.

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
Convert prediction file to separate json files:

```bash
cd evaluation
python convert_predictions_to_files.py --eval_file ../saved/btrain/eval_pred.jsonl --lang en
```
Then run the evaluation script:
```bash
python2 evaluation/eval.py -t 0.0 -p ../data/en/eval_all/ -o pred_label_files/labels_btrain