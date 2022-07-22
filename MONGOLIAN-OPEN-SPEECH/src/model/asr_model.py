# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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

from collections import defaultdict
from typing import List, Optional, Tuple
import numpy as np

import torch

from transformer.cmvn import GlobalCMVN
from transformer.ctc import CTC
from transformer.decoder import (TransformerDecoder,
                                 BiTransformerDecoder)
from transformer.encoder import ConformerEncoder
from transformer.encoder import TransformerEncoder
from transformer.label_smoothing_loss import LabelSmoothingLoss
from utils.cmvn import load_cmvn
from utils.common import (IGNORE_ID, add_sos_eos, log_add,
                          remove_duplicates_and_blank, th_accuracy,
                          reverse_pad_list)
from model.base_model import BaseASRModel


class ASRModel(BaseASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            vocab_size: int,
            encoder: TransformerEncoder,
            decoder: TransformerDecoder,
            ctc: CTC,
            ctc_weight: float = 0.5,
            ignore_id: int = IGNORE_ID,
            reverse_weight: float = 0.0,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
    ):
        super().__init__(vocab_size=vocab_size, encoder=encoder, decoder=decoder,
                         ctc=ctc, ctc_weight=ctc_weight, ignore_id=ignore_id,
                         reverse_weight=reverse_weight, lsm_weight=lsm_weight,
                         length_normalized_loss=length_normalized_loss)

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return loss, loss_att, loss_ctc

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_mask: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)

        isnan = np.isnan(decoder_out.cpu().detach().numpy())
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        if True in isnan:
            print('!!!!!!!!!!!!!!!!!!!!')
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
                1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att
