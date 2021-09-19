from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions import Categorical
from joblib import Parallel, delayed

from data.target_data import RING_ID_END, RING_ID_START
from data.target_data import Data as TargetData
from data.target_data import MAX_LEN as MAX_TARGET_LEN
from data.target_data import TOKENS as TARGET_TOKENS
from data.target_data import get_id as get_target_id
from data.target_data import PAD_TOKEN as TARGET_PAD_TOKEN

from data.source_data import MAX_LEN as MAX_SOURCE_LEN
from data.source_data import TOKENS as SOURCE_TOKENS
from data.target_data import get_id as get_source_id
from data.target_data import PAD_TOKEN as SOURCE_PAD_TOKEN

from model.generator import TokenEmbedding, EdgeLogitLayer


class BaseEncoder(nn.Module):
    def __init__(self, num_layers, emb_size, nhead, dim_feedforward, dropout):
        super(BaseEncoder, self).__init__()
        self.nhead = nhead
        self.tok_emb = TokenEmbedding(len(SOURCE_TOKENS), emb_size)

        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.distance_embedding_layer = nn.Embedding(MAX_SOURCE_LEN, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        sequences, distance_squares = src
        sequence_len = sequences.size(1)

        #
        out = self.tok_emb(sequences)
        out = self.input_dropout(out)
        out = out.transpose(0, 1)

        #
        mask = self.distance_embedding_layer(distance_squares)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, sequence_len, sequence_len)

        #
        key_padding_mask = sequences == get_source_id(SOURCE_PAD_TOKEN)

        #
        memory = self.transformer(out, mask, key_padding_mask)

        return memory, key_padding_mask


class BaseDecoder(nn.Module):
    def __init__(self, num_layers, emb_size, nhead, dim_feedforward, dropout):
        super(BaseDecoder, self).__init__()
        self.nhead = nhead

        self.token_embedding_layer = TokenEmbedding(len(TARGET_TOKENS), emb_size)
        self.branch_embedding_layer = TokenEmbedding(MAX_TARGET_LEN, emb_size)

        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.distance_embedding_layer = nn.Embedding(MAX_TARGET_LEN + 1, nhead)

        self.up_loc_embedding_layer = nn.Embedding(MAX_TARGET_LEN + 1, nhead)
        self.down_loc_embedding_layer = nn.Embedding(MAX_TARGET_LEN + 1, nhead)
        self.right_loc_embedding_layer = nn.Embedding(MAX_TARGET_LEN + 1, nhead)

        #
        encoder_layer = nn.TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(TARGET_TOKENS) - (RING_ID_END - RING_ID_START))
        self.ring_generator = EdgeLogitLayer(emb_size=emb_size, hidden_dim=emb_size)

    def forward(self, tgt, memory, memory_key_padding_mask):
        (
            sequences,
            branch_sequences,
            distance_squares,
            up_loc_squares,
            down_loc_squares,
            right_loc_squares,
            pred_masks,
        ) = tgt
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)

        #
        out = self.token_embedding_layer(sequences)
        out += self.branch_embedding_layer(branch_sequences)

        out = self.input_dropout(out)

        #
        mask = self.distance_embedding_layer(distance_squares)
        mask += self.up_loc_embedding_layer(up_loc_squares)
        mask += self.down_loc_embedding_layer(down_loc_squares)
        mask += self.right_loc_embedding_layer(right_loc_squares)

        mask = mask.permute(0, 3, 1, 2)

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        key_padding_mask = sequences == get_target_id(TARGET_PAD_TOKEN)

        out = out.transpose(0, 1)
        out = self.transformer(
            out,
            memory,
            tgt_mask=mask,
            memory_mask=None,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = out.transpose(0, 1)

        logits0 = self.generator(out)
        logits1 = self.ring_generator(out, sequences)
        logits = torch.cat([logits0, logits1], dim=2)
        logits = logits.masked_fill(pred_masks, float("-inf"))

        return logits


class BaseTranslator(nn.Module):
    def __init__(self, num_layers, emb_size, nhead, dim_feedforward, dropout):
        super(BaseTranslator, self).__init__()
        self.encoder = BaseEncoder(num_layers, emb_size, nhead, dim_feedforward, dropout)
        self.decoder = BaseDecoder(num_layers, emb_size, nhead, dim_feedforward, dropout)

    def forward(self, src, tgt):
        memory, memory_key_padding_mask = self.encoder(src)
        logits = self.decoder(tgt, memory, memory_key_padding_mask)
        return logits

    def decode(self, src, max_len, device):
        num_samples = src[0].size(0)
        memory, memory_key_padding_mask = self.encoder(src)

        data_list = [TargetData() for _ in range(num_samples)]
        data_idx_list = list(range(num_samples))
        ended_data_list = []
        parallel = Parallel(n_jobs=8)

        def _update_data(inp):
            data, id = inp
            data.update(id)
            return data

        for _ in (range(max_len)):
            if len(data_list) == 0:
                break

            tgt = TargetData.collate([data.featurize() for data in data_list])
            tgt = [tsr.to(device) for tsr in tgt]
            memory_ = memory[:, torch.tensor(data_idx_list)]
            memory_key_padding_mask_ = memory_key_padding_mask[torch.tensor(data_idx_list)]
            logits = self.decoder(tgt, memory_, memory_key_padding_mask_)
            preds = Categorical(logits=logits[:, -1]).sample()
            data_list = parallel(delayed(_update_data)(pair) for pair in zip(data_list, preds.tolist()))

            ended_data_list += [data for data in data_list if data.ended]
            data_idx_list, data_list = map(
                list, zip(*[(idx, data) for idx, data in zip(data_idx_list, data_list) if not data.ended])
            )

        data_list = data_list + ended_data_list

        return data_list
