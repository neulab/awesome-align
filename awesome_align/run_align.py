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

import time
import argparse
import random
import itertools
import os
from multiprocessing import Pool, Process, SimpleQueue, cpu_count, Pipe
import copy


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

def takeFirst(elem):
    return elem[0]

def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

def insert_sentence_in_batch(idxs, sent_src, sent_tgt, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sentence):
    idxs.append(sentence[0])
    sent_src.append(sentence[1])
    sent_tgt.append(sentence[2])
    ids_src.append(sentence[3])
    ids_tgt.append(sentence[4])
    bpe2word_map_src.append(sentence[5])
    bpe2word_map_tgt.append(sentence[6])


def process_encoding(q_data: SimpleQueue, q_preprocessed : SimpleQueue, tokenizer: PreTrainedTokenizer, number_line, nb_preprocess):
    End = False
    while End != True:
        idx, line = q_data.get()
        if idx >= (number_line-nb_preprocess):
            #print("C'est la fin de  process encoding")
            End = True
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
        q_preprocessed.put(  (idx, sent_src, sent_tgt, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt) )
    time.sleep(10)

def feed_data(tokenizer: PreTrainedTokenizer, args, file_path, q_data : SimpleQueue):
    assert os.path.isfile(file_path)
    print('Loading the dataset...')

    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
                raise ValueError('Line {idx+1} is not in the correct format!')
            q_data.put( (idx, line) )

def data_batch(q_preprocessed : SimpleQueue, q_batch, args, tokenizer : PreTrainedTokenizer, number_line):
    number_batch = args.batch_size
    current_row = 0
    End = False
    tmp_info = [] # Store wrong order sentences
    while End != True:
        idxs, sent_src, sent_tgt, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = ([] for i in range(7))
        i = 0 
        while i < args.batch_size:
            if tmp_info == [] or tmp_info[-1][0] != current_row:
                sentence = q_preprocessed.get()
                if sentence[0] != current_row:
                    tmp_info.append( sentence )
                    tmp_info.sort(key=takeFirst, reverse=True)
                else:
                    i += 1
                    current_row += 1
                    insert_sentence_in_batch(idxs, sent_src, sent_tgt, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sentence)
            if tmp_info != []:
                while i < args.batch_size and tmp_info != [] and tmp_info[-1][0] == current_row:

                    if tmp_info[-1][0] == current_row:
                        sentence = tmp_info.pop()
                        current_row += 1
                        i += 1
                        insert_sentence_in_batch(idxs, sent_src, sent_tgt, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sentence)

            if current_row == number_line:
                End = True
                break                    

        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        q_batch.send( (idxs, sent_src, sent_tgt, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt) )
    time.sleep(15)

def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, q_batch, number_line, pipe_recovering):
    End = False
    model.to(args.device)
    model.eval()
    with open(args.output_file, 'w') as writer:
        with torch.no_grad():
            while End != True:
                batch = q_batch.recv()
                idxs, sent_src, sent_tgt, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = batch
                word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=(args.output_prob_file is not None))
                pipe_recovering.send( (idxs, sent_src, sent_tgt, word_aligns_list) )
                if idxs[-1] == number_line-1:
                    End = True

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def alignement_recovering(p, args, number_line):
    end = False
    tqdm_iterator = trange(number_line, desc="Extracting")
    if args.output_prob_file is not None:
        prob_writer = open(args.output_prob_file, 'w')
    with open(args.output_file, 'w') as writer:
        while end != True:
            idxs, sent_src, sent_tgt, word_aligns_list = p.recv()
            output_str = []
            for idx, word_aligns in enumerate(word_aligns_list):
                output_str = []
                if args.output_prob_file is not None:
                    output_prob_str = []
                for word_align in word_aligns:
                    if args.word_output:
                        output_str.append(f'{sent_src[idx][word_align[0]]}  {sent_tgt[idx][word_align[1]]}')
                    else:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                    if args.output_prob_file is not None:
                        output_prob_str.append(f'{word_aligns[word_align]}')
                if args.word_output:
                    writer.write('\n'.join(output_str)+'\n')
                else :
                    writer.write(' '.join(output_str)+'\n')
                if args.output_prob_file is not None:
                    prob_writer.write(' '.join(output_prob_str)+'\n')
            tqdm_iterator.update(len(idxs))
            if idxs[-1] == number_line-1:
                end = True

            


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
    parser.add_argument("--nb_preprocess", type=int, default=int(cpu_count()/2), help="Number of process that run preprocessing, default = cpu_count / 2")
    parser.add_argument("--word_output", action="store_true", help="Write align-word in the output")
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


    number_line = file_len(args.data_file)
    jobs = []
    q_data = SimpleQueue()
    q_batch = Pipe()
    q_preprocessed = SimpleQueue()

    # Creation of preprocessing process 
    for i in range(args.nb_preprocess):
        p = Process(target=process_encoding, args=(q_data, q_preprocessed, copy.deepcopy(tokenizer), number_line, args.nb_preprocess))
        p.start()
        jobs.append(p)

    # Creation of feed_data process (could change that to a thread but it's not faster)
    p = Process(target=feed_data, args=(tokenizer, args, args.data_file, q_data))
    p.start()
    jobs.append(p)
    p = Process(target=data_batch, args=(q_preprocessed, q_batch[1], args, copy.deepcopy(tokenizer), number_line))
    p.start()
    jobs.append(p)
    #alignement_recovering, (could change that to a thread but it's not faster)
    pipe = Pipe()
    p = Process(target=alignement_recovering, args=(pipe[0], args, number_line))
    p.start()
    jobs.append(p)

    #Run word, alignement
    word_align(args, model, tokenizer, q_batch[0], number_line, pipe[1])


    for process in jobs:
        process.join()

if __name__ == "__main__":
    main()
