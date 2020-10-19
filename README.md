# NeuAligner

NeuAligner is a tool that can extract word alignments from contextualized word embeddings and allows you to fine-tune contextualized embeddings on parallel corpora for word alignment.


## Input format

Inputs should be *tokenized* and each line is a source language sentence and its target language translation, separated by (` ||| `). You can see some examples in the `examples` folder.

## Fine-tuning on parallel data

If there is some parallel data available, you can fine-tune your contextualized embedding model. An example for fine-tuning multilingual BERT:

```bash
TRAIN_FILE=/path/to/train/file
EVAL_FILE=/path/to/eval/file
OUTPUT_DIR=/path/to/output/directory

CUDA_VISIBLE_DEVICES=0 python run_train.py \
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
    --save_steps 2000 \
    --max_steps 20000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --overwrite_output_dir \
```

## Extracting alignments

Here is an example of extracting word alignments from multilingual BERT:

```bash
DATA_FILE=/path/to/data/file
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
OUTPUT_FILE=/path/to/output/file

python run_align.py \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
```

This produces outputs in the `i-j` Pharaoh format. A pair `i-j` indicates that the <i>i</i>th word (zero-indexed) of the source sentence is aligned to the <i>j</i>th word of the target sentence.

You can also set `MODEL_NAME_OR_PATH` to the path of your fine-tuned model.

## Acknowledgements

Some of the code is borrowed from [HuggingFace Transformers](https://github.com/huggingface/transformers).
