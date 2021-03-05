## AWESOME: Aligning Word Embedding Spaces of Multilingual Encoders

`awesome-align` is a tool that can extract word alignments from multilingual BERT (mBERT) [[Demo]](https://colab.research.google.com/drive/1205ubqebM0OsZa1nRgbGJBtitgHqIVv6?usp=sharing) and allows you to fine-tune mBERT on parallel corpora for better alignment quality (see [our paper](https://arxiv.org/abs/2101.08231) for more details).

### Dependencies

First, you need to install the dependencies:

```bash
pip install -r requirements.txt
python setup.py install
```

### Input format

Inputs should be *tokenized* and each line is a source language sentence and its target language translation, separated by (` ||| `). You can see some examples in the `examples` folder.

### Extracting alignments

Here is an example of extracting word alignments from multilingual BERT:

```bash
DATA_FILE=/path/to/data/file
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
OUTPUT_FILE=/path/to/output/file

CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32 \
```

This produces outputs in the `i-j` Pharaoh format. A pair `i-j` indicates that the <i>i</i>th word (zero-indexed) of the source sentence is aligned to the <i>j</i>th word of the target sentence.

You can also set `MODEL_NAME_OR_PATH` to the path of your fine-tuned model as shown below.

### Fine-tuning on parallel data

If there is parallel data available, you can fine-tune embedding models on that data.

Here is an example of fine-tuning mBERT that balances well between efficiency and effectiveness:

```bash
TRAIN_FILE=/path/to/train/file
EVAL_FILE=/path/to/eval/file
OUTPUT_DIR=/path/to/output/directory

CUDA_VISIBLE_DEVICES=0 awesome-train \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=bert-base-multilingual-cased \
    --extraction 'softmax' \
    --do_train \
    --train_tlm \
    --train_so \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --save_steps 4000 \
    --max_steps 20000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
```

You can also fine-tune the model a bit longer with more training objectives for better quality:

```bash
TRAIN_FILE=/path/to/train/file
EVAL_FILE=/path/to/eval/file
OUTPUT_DIR=/path/to/output/directory

CUDA_VISIBLE_DEVICES=0 awesome-train \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=bert-base-multilingual-cased \
    --extraction 'softmax' \
    --do_train \
    --train_mlm \
    --train_tlm \
    --train_tlm_full \
    --train_so \
    --train_psi \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --save_steps 10000 \
    --max_steps 40000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
```

If you want high alignment recalls, you can turn on the `--train_co` option, but note that the alignment precisions may drop.

### Model performance

The following table shows the alignment error rates (AERs) of our models and popular statistical word aligners on five language pairs. The De-En, Fr-En, Ro-En datasets can be obtained following [this repo](https://github.com/lilt/alignment-scripts), the Ja-En data is from [this link](http://www.phontron.com/kftt/) and the Zh-En data is available at [this link](http://nlp.csai.tsinghua.edu.cn/~ly/systems/TsinghuaAligner/TsinghuaAligner.html). The best scores are in **bold**.

|            | De-En | Fr-En | Ro-En | Ja-En | Zh-En |
| -| ------- | ------- | ------- | ------- | ------- | 
| [fast\_align](https://github.com/clab/fast_align) | 27.0 | 10.5 | 32.1 | 51.1 | 38.1 |
| [eflomal](https://github.com/robertostling/eflomal) | 22.6 | 8.2 | 25.1 | 47.5 | 28.7 |
| [Mgiza](https://github.com/moses-smt/mgiza)    | 20.6 | 5.9 | 26.4 | 48.0 | 35.1 |
| Ours (w/o fine-tuning, softmax) | 17.4 | 5.6 | 27.9 | 45.6 | 18.1 |
| Ours (multilingually fine-tuned <br/>  w/o `--train_co`, softmax) [[Download]](https://drive.google.com/file/d/1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ/view?usp=sharing) | 15.2 | **4.1** | 22.6 | **37.4** | **13.4** |
| Ours (multilingually fine-tuned <br/>  w/ `--train_co`, softmax) [[Download]](https://drive.google.com/file/d/1IluQED1jb0rjITJtyj4lNMPmaRFyMslg/view?usp=sharing) |  **15.1** | 4.5 | **20.7** | 38.4 | 14.5 |


### Citation

If you use our tool, we'd appreciate if you cite the following paper:

```
@inproceedings{dou2021word,
  title={Word Alignment by Fine-tuning Embeddings on Parallel Corpora},
  author={Dou, Zi-Yi and Neubig, Graham},
  booktitle={Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2021}
}
```


### Acknowledgements

Some of the code is borrowed from [HuggingFace Transformers](https://github.com/huggingface/transformers) licensed under [Apache 2.0](https://github.com/huggingface/transformers/blob/master/LICENSE) and the entmax implementation is from [this repo](https://github.com/deep-spin/entmax).
