# MNMT with Semantic Image Regions Quickstart

## Step 0: Download the semantic image region features for the Multi30k data set.

GDrive: [Features_hdf5_file of train/vaild/test for Multi30k data set](https://drive.google.com/drive/folders/1LpOBlCfsFkmDq_b614fuXtncxGSTLAbH?usp=sharing)

These semantic image region features are extracted by the object-detection based method [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), and then pro-processed with .hdf5 file format. The amount of regions per image is 100, and each region feature is representated as a 2,048-dim vector.

## Step 1: Preprocess the data

1. Downloaded the [Multi30k data set](http://www.statmt.org/wmt16/multimodal-task.html) and extracted the sentences in its training, validation and test sets. 
2. After pre-processing them ([Moses tokenizer](https://github.com/moses-smt/mosesdecoder/tree/RELEASE-2.1.1) and [BPE model](https://github.com/rsennrich/subword-nmt)), feed the training and validation sets to the `preprocess.py` script, as below.

```bash
python preprocess.py -train_src multi30k/train.norm.tok.lc.bpe10000.en -train_tgt multi30k/train.norm.tok.lc.bpe10000.de -valid_src multi30k/val.norm.tok.lc.bpe10000.en -valid_tgt multi30k/val.norm.tok.lc.bpe10000.de -save_data data/m30k
```

*Our processed bpe file is saved in 'data/bpe/' and data file is saved in 'data/ende/'.

### Step 2: Train the model

```bash
python train_mm.py -data data/ende/m30k -save_model model-ende/NMT-src-img_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats semantic_regions/local_obj36_train_2016.hdf5 -path_to_valid_img_feats semantic_regions/local_obj36_val_2016.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection -decoder_type doubly-attentive-rnn --multimodal_model_type src+img
```

### Step 3: Translate new sentences

```bash
MODEL_SNAPSHOT=model-ende/NMT-src-img_ADAM_acc_65.83_ppl_7.57_e16.pt                                                                                                    
python translate_mm.py -gpu 0 -src multi30k/test2016.norm.tok.lc.bpe10000.en -model ${MODEL_SNAPSHOT} -path_to_test_img_feats semantic_regions/local_feats_test.hdf5 -output ${MODEL_SNAPSHOT}.translation.de
```

## Citation

If you use any part of this repository, please consider citing the following papers:

```
@inproceedings{zhao-etal-2020-double,
    title = "Double Attention-based Multimodal Neural Machine Translation with Semantic Image Regions",
    author = "Zhao, Yuting  and
      Komachi, Mamoru  and
      Kajiwara, Tomoyuki  and
      Chu, Chenhui",
    booktitle = "Proceedings of the 22nd Annual Conference of the European Association for Machine Translation",
    month = nov,
    year = "2020",
    address = "Lisboa, Portugal",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2020.eamt-1.12",
    pages = "105--114",
}
```

This work is based on [Multi-modal Neural Machine Translation](https://github.com/iacercalixto/MultimodalNMT#multi-modal-neural-machine-translation)

```
@InProceedings{CalixtoLiu2017EMNLP,
  Title                    = {{Incorporating Global Visual Features into Attention-Based Neural Machine Translation}},
  Author                   = {Iacer Calixto and Qun Liu},
  Booktitle                = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  Year                     = {2017},
  Address                  = {Copenhagen, Denmark},
  Url                      = {http://aclweb.org/anthology/D17-1105}
}
```

```
@InProceedings{CalixtoLiuCampbell2017ACL,
  author    = {Calixto, Iacer  and  Liu, Qun  and  Campbell, Nick},
  title     = {{Doubly-Attentive Decoder for Multi-modal Neural Machine Translation}},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1913--1924},
  url       = {http://aclweb.org/anthology/P17-1175}
}
```

If you use OpenNMT, please cite as below.

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
