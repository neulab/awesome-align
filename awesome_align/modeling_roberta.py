
import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from awesome_align.configuration_roberta import RobertaConfig
from awesome_align.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from awesome_align.modeling import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu, BertPSIHead, BertGuideHead, PAD_ID, CLS_ID, SEP_ID

from awesome_align.sparsemax import sparsemax, entmax15  


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx
class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.psi_cls = BertPSIHead(config)
        self.guide_layer = BertGuideHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        inputs_src,
        inputs_tgt=None,
        labels_src=None,
        labels_tgt=None,
        attention_mask_src=None,
        attention_mask_tgt=None,
        align_layer=-1,
        guide=None,
        extraction='softmax', softmax_threshold=0.001,
        position_ids1=None,
        position_ids2=None,
        labels_psi=None,
        train_so=True,
        train_co=False,
    ):

        loss_fct=CrossEntropyLoss(reduction='none')
        batch_size = inputs_src.size(0)

        outputs_src = self.roberta(
            inputs_src,
            attention_mask=attention_mask_src,
            align_layer=align_layer,
            position_ids=position_ids1,
        )

        if labels_psi is not None:
            prediction_scores_psi = self.psi_cls(outputs_src)
            psi_loss = loss_fct(prediction_scores_psi.view(-1, 2), labels_psi.view(-1))
            psi_loss = torch.mean(psi_loss)
            return psi_loss

        if inputs_tgt is None:
            prediction_scores_src = self.lm_head(outputs_src)
            masked_lm_loss = loss_fct(prediction_scores_src.view(-1, self.config.vocab_size), labels_src.view(-1))
            masked_lm_loss = torch.sum(masked_lm_loss.view(batch_size, -1), -1) / (torch.sum(labels_src.view(batch_size, -1)!=-100, -1).float()+1e-9) #ignore_index=-100
            masked_lm_loss = torch.mean(masked_lm_loss)
            return masked_lm_loss

        if guide is None:
            raise ValueError('must specify labels for the self-trianing objective')

        outputs_tgt = self.roberta(
            inputs_tgt,
            attention_mask=attention_mask_tgt,
            align_layer=align_layer,
            position_ids=position_ids2,
        )

        sco_loss = self.guide_layer(outputs_src, outputs_tgt, inputs_src, inputs_tgt, guide=guide, extraction=extraction, softmax_threshold=softmax_threshold, train_so=train_so, train_co=train_co)
        return sco_loss

    def get_aligned_word(self, inputs_src, inputs_tgt, bpe2word_map_src, bpe2word_map_tgt, device, src_len, tgt_len, align_layer=8, extraction='softmax', softmax_threshold=0.001, test=False, output_prob=False, word_aligns=None):
        batch_size = inputs_src.size(0)
        bpelen_src, bpelen_tgt = inputs_src.size(1)-2, inputs_tgt.size(1)-2
        if word_aligns is None:
            inputs_src = inputs_src.to(dtype=torch.long, device=device).clone()
            inputs_tgt = inputs_tgt.to(dtype=torch.long, device=device).clone()

            with torch.no_grad():
                outputs_src = self.roberta(
                    inputs_src,
                    align_layer=align_layer,
                    attention_mask=(inputs_src!=PAD_ID),
                )
                outputs_tgt = self.roberta(
                    inputs_tgt,
                    align_layer=align_layer,
                    attention_mask=(inputs_tgt!=PAD_ID),
                )

                attention_probs_inter = self.guide_layer(outputs_src, outputs_tgt, inputs_src, inputs_tgt, extraction=extraction, softmax_threshold=softmax_threshold, output_prob=output_prob)
                if output_prob:
                    attention_probs_inter, alignment_probs = attention_probs_inter
                    alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]
                attention_probs_inter = attention_probs_inter.float()
                
            word_aligns = []
            attention_probs_inter = attention_probs_inter[:, 0, 1:-1, 1:-1]

            for idx, (attention, b2w_src, b2w_tgt) in enumerate(zip(attention_probs_inter, bpe2word_map_src, bpe2word_map_tgt)):
                aligns = set() if not output_prob else dict()
                non_zeros = torch.nonzero(attention)
                for i, j in non_zeros:
                    word_pair = (b2w_src[i], b2w_tgt[j])
                    if output_prob:
                        prob = alignment_probs[idx, i, j] 
                        if not word_pair in aligns:
                            aligns[word_pair] = prob
                        else:
                            aligns[word_pair] = max(aligns[word_pair], prob)
                    else:
                        aligns.add(word_pair)
                word_aligns.append(aligns)

        if test:
            return word_aligns

        guide = torch.zeros(batch_size, 1, src_len, tgt_len)
        for idx, (word_align, b2w_src, b2w_tgt) in enumerate(zip(word_aligns, bpe2word_map_src, bpe2word_map_tgt)):
            len_src = min(bpelen_src, len(b2w_src))
            len_tgt = min(bpelen_tgt, len(b2w_tgt))
            for i in range(len_src):
                for j in range(len_tgt):
                    if (b2w_src[i], b2w_tgt[j]) in word_align:
                        guide[idx, 0, i+1, j+1] = 1.0
        return guide
