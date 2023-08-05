The code for ACL2023 paper: 《DualGATs: Dual Graph Attention Networks for Emotion Recognition in Conversations》


## Requirements

- Python 3.6.13
- PyTorch 1.7.1+cu110


With Anaconda, we can create the environment with the provided `environment.yml`:

```bash
conda env create --file environment.yml 
conda activate MMERC
```

The code has been tested on Ubuntu 16.04 using a single GPU.
<br>

## Run Steps

1. Please download the four ERC datasets (including pre-processed discourse graphs and RoBERTa utterance feature) and put them in the data folder. Here we utilize the data and codes from [here](https://github.com/shizhouxing/DialogueDiscourseParsing) to pre-train a conversation discourse parser and use that parser to extract discourse graphs in the four ERC datasets. And we utilize the codes from [here](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC) to extract utterance feature.
2. Run our model:

```bash
# For IEMOCAP:
CUDA_VISIBLE_DEVICES=0 python main.py --dataset IEMOCAP --lr 1e-4 --dropout 0.2 --batch_size 16 --gnn_layers 2
# For MELD:
CUDA_VISIBLE_DEVICES=0 python main.py --dataset MELD --lr 1e-4 --dropout 0.3 --batch_size 32 --gnn_layers 2
# For EmoryNLP:
CUDA_VISIBLE_DEVICES=0 python main.py --dataset EmoryNLP --lr 1e-4 --dropout 0.1 --batch_size 32 --gnn_layers 2
# For DailyDialog:
CUDA_VISIBLE_DEVICES=0 python main.py --dataset DailyDialog --lr 5e-5 --dropout 0.4 --batch_size 64 --gnn_layers 3
```

## Citation

```
@inproceedings{zhang2023dualgats,
  title={DualGATs: Dual Graph Attention Networks for Emotion Recognition in Conversations},
  author={Zhang, Duzhen and Chen, Feilong and Chen, Xiuyi},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7395--7408},
  year={2023}
}
```

