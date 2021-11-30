# coding=utf-8
# copyright (C) Ziyi Dou
# Modifications copyright (C) 2021 Jun Cao

import itertools
import random
from typing import List

import numpy as np
import torch
from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Dataset(IterableDataset):
    def __init__(self, tokenizer, examples):
        self.examples = examples
        self.tokenizer = tokenizer

    def process_line(self, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None

        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [
            self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = \
            self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                             max_length=self.tokenizer.max_len)['input_ids'], \
            self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                             max_length=self.tokenizer.max_len)['input_ids']
        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt

    def __iter__(self):
        for example in self.examples:
            processed = self.process_line(example)
            if processed is None:
                empty_tensor = torch.tensor(
                    [self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                empty_sent = ''
                processed = empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent
                yield processed
            else:
                yield processed


class AwesomeAlign(object):

    def __init__(self, model_name_or_path, seed=42, no_cuda=False):
        set_seed(seed)
        config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
        config = config_class.from_pretrained(model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        self.model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=config)
        modeling.PAD_ID = self.tokenizer.pad_token_id
        modeling.CLS_ID = self.tokenizer.cls_token_id
        modeling.SEP_ID = self.tokenizer.sep_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

    def get_align(self,
                  lines: List[str],
                  batch_size=32, align_layer=8, extraction='softmax', threshold=0.001):

        def collate(examples):
            ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(
                *examples)
            ids_src = pad_sequence(ids_src, batch_first=True,
                                   padding_value=self.tokenizer.pad_token_id)
            ids_tgt = pad_sequence(ids_tgt, batch_first=True,
                                   padding_value=self.tokenizer.pad_token_id)
            return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

        dataset = Dataset(tokenizer=self.tokenizer, examples=lines)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate
        )

        self.model.to(self.device)
        self.model.eval()

        res = []
        for batch in dataloader:
            with torch.no_grad():
                ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
                word_aligns_list = self.model.get_aligned_word(ids_src,
                                                               ids_tgt,
                                                               bpe2word_map_src,
                                                               bpe2word_map_tgt, self.device, 0, 0,
                                                               align_layer=align_layer,
                                                               extraction=extraction,
                                                               softmax_threshold=threshold,
                                                               test=True, output_prob=False)
                for word_aligns, sent_src, sent_tgt in zip(word_aligns_list, sents_src, sents_tgt):
                    word_aligns = list(sorted(word_aligns, key=lambda w: (w[0], w[1])))
                    res.append((word_aligns, sent_src, sent_tgt))
        return res

    def get_align_score(self,
                        examples: List[str],
                        batch_size=32, align_layer=8, extraction='softmax', threshold=0.001):
        res = self.get_align(examples, batch_size, align_layer, extraction, threshold)
        scores = list()
        for word_aligns, src_token, trg_token in res:
            src_aligns_len = len(set([item[0] for item in word_aligns]))
            trg_aligns_len = len(set([item[1] for item in word_aligns]))
            if not src_aligns_len or not trg_aligns_len:
                score = 0
            else:
		# calculate score by harmonic mean of both sides align rate (tokens aligned / total)
                score = 2 / (len(src_token) / src_aligns_len + len(trg_token) / trg_aligns_len)
            scores.append(score)
        return scores


if __name__ == '__main__':
    aa = AwesomeAlign(model_name_or_path="bert-base-multilingual-cased")
    lines = [
                "输入 匹配 参数 命令 为了 指定 兴趣 流量 在 类 映射 上 ||| Configure the class map and specify the match parameter for the",
                "有 接口 的 一个 路由器 该 支持 FR 封装 。 ||| To enable Frame Relay encapsulation on the interface ."
            ] * 20
    d = aa.get_align(lines)
    print(d)
    print(len(d))

