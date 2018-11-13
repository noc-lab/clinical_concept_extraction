# Clinical Concept Extraction with Contextual Word Embedding

This repository contains codes and models clinical concept extraction described in our paper [https://arxiv.org/abs/1810.10566](https://arxiv.org/abs/1810.10566). It is designed for a clinical concept extraction task such as the 2010 i2b2/VA shared task.

## Install package

The package is tested in Python 3.6. To begin with, install `tensorflow` according to [this instruction](https://www.tensorflow.org/install/) and
```bash
pip install git+https://github.com/noc-lab/clinical_concept_extraction.git
```

Next, create a folder, say  `cce_assets`, and set an environment variable `CCE_ASSETS` to the path of the folder. Download the pretrained ELMo model [here](https://github.com/noc-lab/clinical_concept_extraction/releases/download/latest/elmo.tar.gz) and unzip files to the folder. Currently, we don't provide the pretrained LSTM model due to i2b2 license. But you can either train a model according to the [instruction](https://github.com/noc-lab/clinical_concept_extraction/blob/master/training_scripts/README.md) or send us an email to henghuiz@bu.edu with a proof that you can access to the [i2b2 NLP data sets](https://www.i2b2.org/NLP/DataSets/Main.php). (We are currently working on building a model from silver data which will be released soon.) The files should be structured as follows:
```text
cce_assets
├── blstm
│   ├── checkpoint
│   ├── model.data-00000-of-00001
│   └── model.index
└── elmo
    ├── mimic_wiki.hdf5
    ├── options.json
    └── vocab.txt
```

## Usages

An example of how to use the package is shown [here](https://nbviewer.jupyter.org/github/noc-lab/clinical_concept_extraction/blob/master/example.ipynb)

## Citation

If you use the code, please cite this paper:

```text
@article{zhu2018clinical,
  title={Clinical Concept Extraction with Contextual Word Embedding},
  author={Zhu, Henghui and Paschalidis, Ioannis Ch and Tahmasebi, Amir},
  journal={arXiv preprint arXiv:1810.10566},
  year={2018}
}
```

