import torch
import torch.nn as nn
from layers.Layers_pyraformer import EncoderLayer, Decoder, Predictor
from layers.Layers_pyraformer import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.Layers_pyraformer import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from layers.embed_pyraformer import DataEmbedding, CustomEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, arg):
        super().__init__()

        self.d_model = arg.d_model
        # self.model_type = arg.model
        self.window_size = arg.window_size
        self.truncate = arg.truncate
        if arg.decoder == 'attention':
            self.mask, self.all_size = get_mask(arg.seq_len, arg.window_size, arg.inner_size, arg.device)
        else:
            self.mask, self.all_size = get_mask(arg.seq_len, arg.window_size, arg.inner_size, arg.device)
        self.decoder_type = arg.decoder
        if arg.decoder == 'FC':
            self.indexes = refer_points(self.all_size, arg.window_size, arg.device)

        if arg.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if arg.decoder == 'FC' else 0
            q_k_mask = get_q_k(arg.seq_len + padding, arg.inner_size, arg.window_size[0], arg.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(arg.d_model, arg.d_inner_hid, arg.n_heads, arg.d_k, arg.d_v, dropout=arg.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(arg.n_layer)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(arg.d_model, arg.d_inner_hid, arg.n_heads, arg.d_k, arg.d_v, dropout=arg.dropout, \
                    normalize_before=False) for i in range(arg.n_layer)
                ])

        # if arg.embed_type == 'CustomEmbedding':
        #     self.enc_embedding = CustomEmbedding(arg.d_feature, arg.d_model, arg.covariate_size, arg.seq_num, arg.dropout)
        # else:
        self.enc_embedding = DataEmbedding(arg.d_feature, arg.d_model, arg.dropout)

        self.conv_layers = eval(arg.CSCM)(arg.d_model, arg.window_size, arg.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Pyraformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, arg):
        super().__init__()

        self.pred_len = arg.pred_len
        self.d_model = arg.d_model
        self.input_size = arg.seq_len
        self.decoder_type = arg.decoder
        # self.channels = arg.d_feature

        self.encoder = Encoder(arg)
        if arg.decoder == 'attention':
            mask = get_subsequent_mask(arg.seq_len, arg.window_size, arg.pred_len, arg.truncate)
            self.decoder = Decoder(arg, mask)
            self.predictor = Predictor(arg.d_model, arg.d_feature)
        elif arg.decoder == 'FC':
            self.predictor = Predictor(4 * arg.d_model, arg.pred_len * arg.d_feature)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.seq_len], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.pred_len, -1)

        return pred

