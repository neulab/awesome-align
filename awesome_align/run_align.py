# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
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
import os

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel



def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path):
        assert os.path.isfile(file_path)
        print('Loading the dataset...')
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
                    raise ValueError(f'Line {idx+1} is not in the correct format!')
                
                src, tgt = line.split(' ||| ')
                if src.rstrip() == '' or tgt.rstrip() == '':
                    raise ValueError(f'Line {idx+1} is not in the correct format!')
            
                sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
                wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

                ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=tokenizer.max_len)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=tokenizer.max_len)['input_ids']
                if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
                    raise ValueError(f'Line {idx+1} is not in the correct format!')

                bpe2word_map_src = []
                for i, word_list in enumerate(token_src):
                    bpe2word_map_src += [i for x in word_list]
                bpe2word_map_tgt = []
                for i, word_list in enumerate(token_tgt):
                    bpe2word_map_tgt += [i for x in word_list]

                self.examples.append( (ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt) )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):

    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt

    dataset = LineByLineTextDataset(tokenizer, args, file_path=args.data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(dataset.__len__(), desc="Extracting")
    if args.output_prob_file is not None:
        prob_writer = open(args.output_prob_file, 'w')
    with open(args.output_file, 'w') as writer:
        for batch in dataloader:
            with torch.no_grad():
                ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = batch
                word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=(args.output_prob_file is not None))
                for word_aligns in word_aligns_list:
                    output_str = []
                    if args.output_prob_file is not None:
                        output_prob_str = []
                    for word_align in word_aligns:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                        if args.output_prob_file is not None:
                            output_prob_str.append(f'{word_aligns[word_align]}')
                    writer.write(' '.join(output_str)+'\n')
                    if args.output_prob_file is not None:
                        prob_writer.write(' '.join(output_prob_str)+'\n')
                tqdm_iterator.update(len(ids_src))
    if args.output_prob_file is not None:
        prob_writer.close()


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
        help="The output file."
    )
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, help='softmax or entmax15'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--output_prob_file", default=None, type=str, help='The output probability file.'
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
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
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

    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id

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
