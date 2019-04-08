# Clinical Concept Extraction with Contextual Word Embedding

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

This repository contains codes and models clinical concept extraction described in our paper [https://arxiv.org/abs/1810.10566](https://arxiv.org/abs/1810.10566). It is designed for a clinical concept extraction task such as the 2010 i2b2/VA shared task.

## Install package

The package is tested in Python 3.6. To begin with, install `tensorflow` according to [this instruction](https://www.tensorflow.org/install/) and
```bash
pip install git+https://github.com/noc-lab/clinical_concept_extraction.git
```

Next, create a folder, say `cce_assets`, and set an environment variable `CCE_ASSETS` to the path of the folder. Download the pretrained ELMo model [here](https://github.com/noc-lab/clinical_concept_extraction/releases/download/latest/elmo.tar.gz) and unzip files to the folder. Currently, we don't provide the pretrained LSTM model using I2B2 data due to i2b2 license. But we provide a silver model [here](https://github.com/noc-lab/clinical_concept_extraction/releases/download/latest/blstm.tar.gz). We use the gold model trained using all training and test data in 2010 i2b2/VA shared task to generate silver annotations for 2000 discharge summaries in MIMIC-III. Then we fit the these data and get the silver model. Finally, the files should be structured as follows:
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

