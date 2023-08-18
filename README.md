## A Semi-supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues
This repository contains codes of the paper "A Semi-supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues" published in AAAI 2022.

[![paperlink](https://img.shields.io/badge/paper-AAAI-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/21349)
[![paperlink](https://img.shields.io/badge/paper-arXiv-b31b1b)](https://arxiv.org/abs/2202.10948)

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
@inproceedings{lin2022semi,
  author    = "Lin, Qian and Ng, Hwee Tou",
  title     = "A Semi-supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues",
  booktitle = "Proceedings of the AAAI Conference on Artificial Intelligence",
  year      = "2022",
  pages     = "11011--11019",
}
```
### License ###

The source code and models in this repository are licensed under GNU GPL 3.0 (see [LICENSE](LICENSE)) for non-commercial use. For commercial use of this code, separate commercial licensing is also available. Please contact Prof. Hwee Tou Ng ([nght@comp.nus.edu.sg](mailto:nght@comp.nus.edu.sg)).
