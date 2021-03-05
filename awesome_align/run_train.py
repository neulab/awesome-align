# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020, Zi-Yi Dou
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
import glob
import logging
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from awesome_align import modeling
from awesome_align.train_utils import _sorted_checkpoints, _rotate_checkpoints, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel



logger = logging.getLogger(__name__)

import itertools

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        cache_fn = f'{file_path}.cache'
        if args.cache_data and os.path.isfile(cache_fn) and not args.overwrite_cache:
            logger.info("Loading cached data from %s", cache_fn)
            self.examples = torch.load(cache_fn)
        else:
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                for line in f.readlines():
                    if len(line) > 0 and not line.isspace() and len(line.split(' ||| ')) == 2:
                        try:
                            src, tgt = line.split(' ||| ')
                            if src.rstrip() == '' or tgt.rstrip() == '':
                                logger.info("Skipping instance %s", line)
                                continue
                        except:
                            logger.info("Skipping instance %s", line)
                            continue
                        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                        token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
                        wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

                        ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=tokenizer.max_len)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=tokenizer.max_len)['input_ids']
                        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
                            logger.info("Skipping instance %s", line)
                            continue

                        bpe2word_map_src = []
                        for i, word_list in enumerate(token_src):
                            bpe2word_map_src += [i for x in word_list]
                        bpe2word_map_tgt = []
                        for i, word_list in enumerate(token_tgt):
                            bpe2word_map_tgt += [i for x in word_list]

                        self.examples.append( (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt) )

            if args.cache_data:
                logger.info("Saving cached data to %s", cache_fn)
                torch.save(self.examples, cache_fn)



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        neg_i = random.randint(0, len(self.examples)-1)
        while neg_i == i:
            neg_i = random.randint(0, len(self.examples)-1)
        return tuple(list(self.examples[i]) + list(self.examples[neg_i][:2] ) )


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return LineByLineTextDataset(tokenizer, args, file_path=file_path)


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, langid_mask=None, lang_id=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.uint8), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    if langid_mask is not None:
        padding_mask = langid_mask.eq(lang_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
    masked_indices = torch.bernoulli(probability_matrix).byte()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).byte() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).byte() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    def collate(examples):
        model.eval()
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        for example in examples:
            end_id = example[0][0][-1].view(-1)

            src_id = example[0][0][:args.block_size]
            src_id = torch.cat([src_id[:-1], end_id])
            tgt_id = example[1][0][:args.block_size]
            tgt_id = torch.cat([tgt_id[:-1], end_id])

            half_block_size = int(args.block_size/2)
            half_src_id = example[0][0][:half_block_size]
            half_src_id = torch.cat([half_src_id[:-1], end_id])
            half_tgt_id = example[1][0][:half_block_size]
            half_tgt_id = torch.cat([half_tgt_id[:-1], end_id])

            examples_src.append(src_id)
            examples_tgt.append(tgt_id)
            src_len = max(src_len, len(src_id))
            tgt_len = max(tgt_len, len(tgt_id))

            srctgt = torch.cat( [half_src_id, half_tgt_id] )
            langid = torch.cat([ torch.ones_like(half_src_id), torch.ones_like(half_tgt_id)*2] )
            examples_srctgt.append(srctgt)
            langid_srctgt.append(langid)

            tgtsrc = torch.cat( [half_tgt_id, half_src_id])
            langid = torch.cat([ torch.ones_like(half_tgt_id), torch.ones_like(half_src_id)*2] )
            examples_tgtsrc.append(tgtsrc)
            langid_tgtsrc.append(langid)

            # [neg, neg] pair
            neg_half_src_id = example[-2][0][:half_block_size]
            neg_half_src_id = torch.cat([neg_half_src_id[:-1], end_id])
            neg_half_tgt_id = example[-1][0][:half_block_size]
            neg_half_tgt_id = torch.cat([neg_half_tgt_id[:-1], end_id])
            if random.random()> 0.5:
                neg_srctgt = torch.cat( [neg_half_src_id, neg_half_tgt_id] )
            else:
                neg_srctgt = torch.cat( [neg_half_tgt_id, neg_half_src_id] )
            psi_examples_srctgt.append(neg_srctgt)
            psi_labels.append(1)
                
            # [pos, neg] pair
            rd = random.random()
            if rd> 0.75:
                neg_srctgt = torch.cat([half_src_id, neg_half_tgt_id])
            elif rd > 0.5:
                neg_srctgt = torch.cat([neg_half_src_id, half_tgt_id])
            elif rd > 0.25:
                neg_srctgt = torch.cat([half_tgt_id, neg_half_src_id])
            else:
                neg_srctgt = torch.cat([neg_half_tgt_id, half_src_id])
            psi_examples_srctgt.append(neg_srctgt)
            psi_labels.append(0)

            bpe2word_map_src.append(example[2])
            bpe2word_map_tgt.append(example[3])
            
        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_srctgt = pad_sequence(examples_srctgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        langid_srctgt = pad_sequence(langid_srctgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgtsrc = pad_sequence(examples_tgtsrc, batch_first=True, padding_value=tokenizer.pad_token_id)
        langid_tgtsrc = pad_sequence(langid_tgtsrc, batch_first=True, padding_value=tokenizer.pad_token_id)
        psi_examples_srctgt = pad_sequence(psi_examples_srctgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        psi_labels = torch.tensor(psi_labels)
        guides = model.get_aligned_word(examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, src_len, tgt_len, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold)
        return examples_src, examples_tgt, guides, examples_srctgt, langid_srctgt, examples_tgtsrc, langid_tgtsrc, psi_examples_srctgt, psi_labels


    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.max_steps > 0 and args.max_steps < t_total:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if ( not (any(nd in n for nd in no_decay)) )],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if ( (any(nd in n for nd in no_decay)) )], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    # Check if continuing training from a checkpoint
    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility

    def backward_loss(loss, tot_loss):
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        tot_loss += loss.item()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return tot_loss

    tqdm_iterator = trange(int(t_total), desc="Iteration", disable=args.local_rank not in [-1, 0])
    for _ in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):

            model.train()

            if args.train_so or args.train_co:
                inputs_src, inputs_tgt = batch[0].clone(), batch[1].clone()
                inputs_src, inputs_tgt = inputs_src.to(args.device), inputs_tgt.to(args.device)
                attention_mask_src, attention_mask_tgt = (inputs_src!=0), (inputs_tgt!=0)
                guide = batch[2].to(args.device)
                loss = model(inputs_src=inputs_src, inputs_tgt=inputs_tgt, attention_mask_src=attention_mask_src, attention_mask_tgt=attention_mask_tgt, guide=guide, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, train_so=args.train_so, train_co=args.train_co)
                tr_loss = backward_loss(loss, tr_loss)

            if args.train_mlm:
                inputs_src, labels_src = mask_tokens(batch[0], tokenizer, args)
                inputs_tgt, labels_tgt = mask_tokens(batch[1], tokenizer, args)
                inputs_src, inputs_tgt = inputs_src.to(args.device), inputs_tgt.to(args.device)
                labels_src, labels_tgt = labels_src.to(args.device), labels_tgt.to(args.device)
                loss = model(inputs_src=inputs_src, labels_src=labels_src)
                tr_loss = backward_loss(loss, tr_loss)

                loss = model(inputs_src=inputs_tgt, labels_src=labels_tgt)
                tr_loss = backward_loss(loss, tr_loss)

            if args.train_tlm:
                rand_ids = [0, 1]
                if not args.train_tlm_full:
                    rand_ids = [int(random.random() > 0.5)]
                for rand_id in rand_ids:
                    select_srctgt = batch[int(3+rand_id*2)]
                    select_langid = batch[int(4+rand_id*2)]
                    for lang_id in [1, 2]:
                        inputs_srctgt, labels_srctgt = mask_tokens(select_srctgt, tokenizer, args, select_langid, lang_id)
                        inputs_srctgt, labels_srctgt = inputs_srctgt.to(args.device), labels_srctgt.to(args.device)
                        loss = model(inputs_src=inputs_srctgt, labels_src=labels_srctgt)
                        tr_loss = backward_loss(loss, tr_loss)

            if args.train_psi:
                loss = model(inputs_src=batch[7].to(args.device), labels_psi=batch[8].to(args.device), align_layer=args.align_layer+1)
                tr_loss = backward_loss(loss, tr_loss)


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                tqdm_iterator.update()

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("  Step %s. Training loss = %s", str(global_step), str((tr_loss-logging_loss)/args.logging_steps))
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if global_step > t_total:
                break
        if global_step > t_total:
            break

    return global_step, tr_loss / global_step

def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    def collate(examples):
        model.eval()
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        for example in examples:
            end_id = example[0][0][-1].view(-1)

            src_id = example[0][0][:args.block_size]
            src_id = torch.cat([src_id[:-1], end_id])
            tgt_id = example[1][0][:args.block_size]
            tgt_id = torch.cat([tgt_id[:-1], end_id])

            half_block_size = int(args.block_size/2)
            half_src_id = example[0][0][:half_block_size]
            half_src_id = torch.cat([half_src_id[:-1], end_id])
            half_tgt_id = example[1][0][:half_block_size]
            half_tgt_id = torch.cat([half_tgt_id[:-1], end_id])

            examples_src.append(src_id)
            examples_tgt.append(tgt_id)
            src_len = max(src_len, len(src_id))
            tgt_len = max(tgt_len, len(tgt_id))

            srctgt = torch.cat( [half_src_id, half_tgt_id] )
            langid = torch.cat([ torch.ones_like(half_src_id), torch.ones_like(half_tgt_id)*2] )
            examples_srctgt.append(srctgt)
            langid_srctgt.append(langid)

            tgtsrc = torch.cat( [half_tgt_id, half_src_id] )
            langid = torch.cat([ torch.ones_like(half_tgt_id), torch.ones_like(half_src_id)*2] )
            examples_tgtsrc.append(tgtsrc)
            langid_tgtsrc.append(langid)

            # [neg, neg] pair
            neg_half_src_id = example[-2][0][:half_block_size]
            neg_half_src_id = torch.cat([neg_half_src_id[:-1], end_id])
            neg_half_tgt_id = example[-1][0][:half_block_size]
            neg_half_tgt_id = torch.cat([neg_half_tgt_id[:-1], end_id])
            neg_srctgt = torch.cat( [neg_half_src_id, neg_half_tgt_id] )
            psi_examples_srctgt.append(neg_srctgt)
            psi_labels.append(1)
                
            # [pos, neg] pair
            neg_srctgt = torch.cat([half_src_id, neg_half_tgt_id])
            psi_examples_srctgt.append(neg_srctgt)
            psi_labels.append(0)

            bpe2word_map_src.append(example[2])
            bpe2word_map_tgt.append(example[3])

            
        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_srctgt = pad_sequence(examples_srctgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        langid_srctgt = pad_sequence(langid_srctgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgtsrc = pad_sequence(examples_tgtsrc, batch_first=True, padding_value=tokenizer.pad_token_id)
        langid_tgtsrc = pad_sequence(langid_tgtsrc, batch_first=True, padding_value=tokenizer.pad_token_id)
        psi_examples_srctgt = pad_sequence(psi_examples_srctgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        psi_labels = torch.tensor(psi_labels)
        guides = model.get_aligned_word(examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, src_len, tgt_len, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold)
        return examples_src, examples_tgt, guides, examples_srctgt, langid_srctgt, examples_tgtsrc, langid_tgtsrc, psi_examples_srctgt, psi_labels

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    set_seed(args)  # Added here for reproducibility

    def post_loss(loss, tot_loss): 
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        tot_loss += loss.item()
        return tot_loss

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            if args.train_so or args.train_co:
                inputs_src, inputs_tgt = batch[0].clone(), batch[1].clone()
                inputs_src, inputs_tgt = inputs_src.to(args.device), inputs_tgt.to(args.device)
                attention_mask_src, attention_mask_tgt = (inputs_src!=0), (inputs_tgt!=0)
                guide = batch[2].to(args.device)
                loss = model(inputs_src=inputs_src, inputs_tgt=inputs_tgt, attention_mask_src=attention_mask_src, attention_mask_tgt=attention_mask_tgt, guide=guide, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, train_so=args.train_so, train_co=args.train_co)
                eval_loss = post_loss(loss, eval_loss)

            if args.train_mlm:
                inputs_src, labels_src = mask_tokens(batch[0], tokenizer, args)
                inputs_tgt, labels_tgt = mask_tokens(batch[1], tokenizer, args)
                inputs_src, inputs_tgt = inputs_src.to(args.device), inputs_tgt.to(args.device)
                labels_src, labels_tgt = labels_src.to(args.device), labels_tgt.to(args.device)
                loss = model(inputs_src=inputs_src, labels_src=labels_src)
                eval_loss = post_loss(loss, eval_loss)

                loss = model(inputs_src=inputs_tgt, labels_src=labels_tgt)
                eval_loss = post_loss(loss, eval_loss)

            if args.train_tlm:
                select_ids = [0, 1]
                if not args.train_tlm_full:
                    select_ids = [0]
                for select_id in select_ids:
                    for lang_id in [1, 2]:
                        inputs_srctgt, labels_srctgt = mask_tokens(batch[3+select_id*2], tokenizer, args, batch[4+select_id*2], lang_id)
                        inputs_srctgt, labels_srctgt = inputs_srctgt.to(args.device), labels_srctgt.to(args.device)
                        loss = model(inputs_src=inputs_srctgt, labels_src=labels_srctgt)
                        eval_loss = post_loss(loss, eval_loss)

            if args.train_psi:
                loss = model(inputs_src=batch[7].to(args.device), labels_psi=batch[8].to(args.device))
                eval_loss = post_loss(loss, eval_loss)

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Training objectives
    parser.add_argument("--train_mlm", action="store_true")
    parser.add_argument("--train_tlm", action="store_true")
    parser.add_argument("--train_tlm_full", action="store_true")
    parser.add_argument("--train_so", action="store_true")
    parser.add_argument("--train_psi", action="store_true")
    parser.add_argument("--train_co", action="store_true")
    # Other parameters
    parser.add_argument("--cache_data", action="store_true", help='if cache the dataset')
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, choices=['softmax', 'entmax'], help='softmax or entmax'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--eval_data_file", default=None, type=str, help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
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
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=2, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

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
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("Saving trained model checkpoint to %s", args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
