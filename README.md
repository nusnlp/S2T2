## A Semi-Supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues
This repository contains codes of the paper A Semi-Supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues.

### Requirements ###
Check dependencies in `requirements.txt`, install required packages (with python 3.6.8):
```
pip install -r requirements.txt
```

### Data ###

Refer to `README` in each `[dataset]` sub-directory for instructions of data retrieval and preprocessing.

### Training ###

Follow the commands of training in `[dataset]/run.sh`.

Trained models can be specified and downloaded by running `bash get_trained_models.sh`.

### Evaluation ###

Refer to `README` in each `[dataset]` sub-directory for evaluation steps.

### Publication ###
If you use the source code or models from this work, please cite our paper:
```
@inproceedings{lin2022a,
  author    = "Lin, Qian and Ng, Hwee Tou",
  title     = "A Semi-Supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues",
  booktitle = "Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence",
  year      = "2022",
}
```
### License ###

The source code is licensed under GNU GPL 3.0 (see LICENSE) for non-commercial use. For commercial use of this code, separate commercial licensing is also available. Please contact Prof. Hwee Tou Ng ([nght@comp.nus.edu.sg](mailto:nght@comp.nus.edu.sg)).