### Data  ###
Run the script of data downloading and preprocessing:
```bash
bash data_prep.sh
```

### Training ###
Use the training script to run training process:
```bash
bash run.sh
```

### Prediction ###
To run prediction from a trained model
```bash
python inference_t.py --config config.json --model [path_to_trained_model]
```

### Evaluation ###
After training, run `eval_lif.py` for evaluation
```bash
bash eval_lif.py --config config.json --pred ./saved/btrain/test_i_pred.jsonl
```
