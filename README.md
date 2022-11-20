<h3 align="center">
<p>GHR
<a href="https://github.com/jaytsien/GHR/blob/main/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
</h3>
<div align="center">
    <p>Capturing Conversational Interaction for Question Answering via <b>G</b>lobal <b>H</b>istory <b>R</b>easoning
    <p>NAACL Findings 2022
</div>

<div align="center">
  <img alt="GHR Overview" src="https://github.com/jaytsien/GHR/blob/main/utils/GHR_model.png" width="800px">
</div> 	


We used the adapter-transformer architecture for the Conversational Question Answering task. In order to model dialogue history, the GHR model was utilized as described in [paper](https://aclanthology.org/2022.findings-naacl.159.pdf). 
To reproduce our results, adapter-transformers from https://github.com/PantelisSfak/adapter-transformers.git have to be cloned and used as adapter_transformers inside the GHR folder.


## Setup

```bash
$ pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install transformers==4.22.2
```

### Datasets

We use the [QuAC (Choi et al., 2018)](https://quac.ai/) dataset.
```bash
mkdir -p datasets
wget https://s3.amazonaws.com/my89public/quac/train.json -O datasets/train.json
wget https://s3.amazonaws.com/my89public/quac/val.json -O datasets/dev.json
```

## Train



```bash

CUDA_VISIBLE_DEVICES=0 python3 roberta_run_quac.py \
	--model_type roberta \
	--model_name_or_path roberta-large \
	--do_train \
	--do_eval \
  	--data_dir ./datasets/ \
	--train_file train.json \
	--predict_file dev.json \
	--output_dir ./tmp/model \
	--per_gpu_train_batch_size 6 \
	--num_train_epochs 2 \
	--learning_rate 2e-4 \
	--weight_decay 0.01 \
	--threads 20 \
	--do_lower_case \
	--fp16 --fp16_opt_level "O2" \
	--evaluate_during_training \
	--max_answer_length 50 \
  	--cache_prefix roberta-large \
	--logging_steps 3000 \
  	--adapter_train bottleneck_adapter 
```

There is the option to change the adapter type. Adapters can also be removed and fine tune the transformer without freezing the weights of its layers. In order to achieve better results when fine-tuning the whole model, it is recomended to decrease the learning rate (lr=2e-5 could be used). Also apex `--fp16` is employed in order to accelerate training and prediction. To reproduce bert the bert_run_quac.py should be used.

## Evaluation

The example below, is for the evaluation of the development set of QUAC. It produces the file "pred.json". This file is used along with (scorer.py found in https://quac.ai/) for the official evaluation.

```bash
!CUDA_VISIBLE_DEVICES=0 python3 roberta_run_quac.py \
	--model_type roberta  \
	--model_name_or_path ./tmp/model \
	--do_eval \
  	--data_dir ./datasets/ \
	--train_file train.json \
	--predict_file dev.json \
	--output_dir ./tmp \
	--per_gpu_train_batch_size 12 \
	--num_train_epochs 2 \
	--learning_rate 2e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--do_lower_case \
	--fp16 \
 	--fp16_opt_level "O2" \
	--evaluate_during_training \
	--max_answer_length 50 \
  	--cache_prefix roberta-large \
	--adapter_train bottleneck_adapter \
	--output_pred_file \
  	--for_eval_only
```

### Result

The results for training with Bert and Roberta models, with and without the use of bottleneck adapters are displayed below. When bottleneck adapters were not used, the learning rate was set to 2e-5.

<div align="center">
  <img alt="GHR_results" src="https://github.com/PantelisSfak/GHR/blob/main/utils/results.png" width="600px">
</div> 	


## Citation

```bibtex
@inproceedings{qian2022capturing,
  title={Capturing Conversational Interaction for Question Answering via Global History Reasoning},
  author={Qian, Jin and Zou, Bowei and Dong, Mengxing and Li, Xiao and Aw, Aiti and Hong, Yu},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  pages={2071--2078},
  year={2022}
}
```
