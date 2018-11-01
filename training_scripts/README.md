# Training Clinical Concept Extraction Models

## Environment Setup

The training scripts are tested in Python 3.6. 
To begin with, install `tensorflow` according to 
[this instruction](https://www.tensorflow.org/install/) and the
 bilm packages and our package by
```bash
pip install git+https://github.com/allenai/bilm-tf.git
pip install git+https://github.com/noc-lab/clinical_concept_extraction.git
```

Next, set up the environments for running the codes. Create a folder, say 
`cce_assets`, and set an environment variables `CCE_ASSETS` to be the path of the folder. Download the pretrained ELMo model [here](https://github.com/noc-lab/clinical_concept_extraction/releases/download/latest/elmo.tar.gz)
and unzip files to the folder. The files should be structured as follows:
```text
cce_assets
└── elmo
    ├── mimic_wiki.hdf5
    ├── options.json
    └── vocab.txt
``` 

Finally, download the I2B2-2010 training and test files from [here](https://www.i2b2.org/NLP/DataSets/Main.php).
Create a folders `data` and place the files there.  The files should be structured as follows:
```text
clinical_concept_extraction
├── data
│    └── raw
│       ├── concept_assertion_relation_training_data.tar.gz
│       ├── reference_standard_for_test_data.tar.gz
│       └── test_data.tar.gz
├── training_scripts
│    ├── dump_models.py
│        ...  
├── clinical_concept_extraction
    ...
``` 

## Preprocess and write tfrecords

To preprocess the data, run
```bash
python preprocess_data.py
```
The scripts will unzip the files, fix the typos in the data, and parse the training and test data.

Then, run
```bash
python write_tfrecords.py
```
It will convert the parsed data to tfrecords. We run the ELMo model and write the embeddings to the tfrecords. It will
take a large space in the hard drive but can save time in the training.

## Train models

To train the NER models starting with 10 different random seeds, run
```bash
for i in {0..9}
do
    python training.py --train=True --random_seed=${i} --save_model_dir=../ckpt/bilstm_crf_concept/model_${i}/
done
``` 

## Evaluate models

After obtaining 10 models using the training scripts, run
```bash
for i in {0..9}
do
    python training.py --train=False --random_seed=${i} --save_model_dir=../ckpt/bilstm_crf_concept/model_${i}/
done
``` 
the annotations of the test files will be generated temporally. Finally, run
```bash
python evaluate.py --ensemble=True --full_report=True
```
This script will convert the annotations into the I2B2 format and run the I2B2 evaluate scripts. It will have a similar
result as our paper reported. 

## Dump a model checkpoint for the package

Finally, run the 
```bash
python dump_models
```
for building a compacted checkpoint file for the package. The files are generated at the folder `ckpt/ensemble`.
Copy the model files to `cce_assets` folder and organize the files as follows
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
Now you can use our provided package to generated annotations for clinical notes. Please see examples in the `examples` 
folder for detailed usage.
