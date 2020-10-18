# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import random
import itertools

import numpy as np
import torch
from tqdm import tqdm

from configuration_bert import BertConfig
from modeling import BertForMaskedLM
from tokenization_bert import BertTokenizer
from tokenization_utils import PreTrainedTokenizer
from modeling_utils import PreTrainedModel



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    model.to(args.device)
    model.eval()
    with open(args.data_file, encoding="utf-8") as f:
        with open(args.output_file, 'w') as writer:
            for line in tqdm(f.readlines()):
                if len(line) > 0 and not line.isspace() and len(line.split(' ||| ')) == 2:
                    try:
                        src, tgt = line.split(' ||| ')
                        if src.rstrip() == '' or tgt.rstrip() == '':
                            writer.write('ERROR\n')
                            continue
                    except:
                        writer.write('ERROR\n')
                        continue
                    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
                    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
                    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt')['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt')['input_ids']
                    ids_src, ids_tgt = ids_src.to(args.device), ids_tgt.to(args.device)
                    

                    bpe2word_map_src = []
                    for i, word_list in enumerate(token_src):
                        bpe2word_map_src += [i for x in word_list]
                    bpe2word_map_tgt = []
                    for i, word_list in enumerate(token_tgt):
                        bpe2word_map_tgt += [i for x in word_list]

                    word_aligns = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, tokenizer, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True)
                    output_str = []
                    for word_align in word_aligns:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                    writer.write(' '.join(output_str)+'\n')
                else:
                    writer.write('ERROR\n')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file", default=None, type=str, required=True, help="The input data file (a text file)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, help='softmax or entmax15'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--cache_dir",
        default='cache_dir',
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        model = model_class(config=config)

    word_align(args, model, tokenizer)

if __name__ == "__main__":
    main()
