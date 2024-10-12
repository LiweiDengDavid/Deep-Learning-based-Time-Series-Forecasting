import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        # pe.requires_grad = False
        self.register_buffer('pe', pe)
        self.initial = None

    def forward(self, x):
        if self.initial == None:
            self.initial = True
            batchsize, dim = x.shape[0], x.shape[1]
            self.pe = torch.cat(batchsize * dim * [self.pe], dim=0).view(batchsize, dim, self.pe.shape[0], -1)

        x += self.pe[:, :, :x.shape[-2], :]
        return x


class SSD(nn.Module):
    def __init__(self, args):
        super(SSD, self).__init__()

        self.seqlen = args.seq_len
        self.prelen = args.pred_len
        self.d_feature = args.d_feature
        self.d_model = args.d_model - args.d_model%args.n_heads  # Preventing dimensions from not dividing heads
        self.s = args.d_dimension
        self.device = args.device
        self.past_dimension = args.d_mark
        self.future_dimension = args.d_mark-1
        self.dropout = args.dropout

        self.input_fc = nn.Linear(self.past_dimension, self.d_model)
        self.co_embedding = nn.Linear(self.future_dimension, self.d_model)

        self.pos_encoder_emb = PositionalEncoding(self.d_model)
        self.pos_decoder_emb = PositionalEncoding(self.d_model)

        self.encoder_layer = [nn.TransformerEncoderLayer(d_model=self.d_model, nhead=args.n_heads,
                                                         dim_feedforward=args.d_ff, batch_first=True,
                                                         dropout=self.dropout, device=self.device) for _ in
                              range(self.d_feature)]
        self.decoder_layer = [nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=args.n_heads,
            dropout=self.dropout, dim_feedforward=args.d_ff,
            batch_first=True, device=self.device
        ) for _ in range(self.d_feature)]
        self.encoder = [torch.nn.TransformerEncoder(self.encoder_layer[i], num_layers=args.e_layers) for i in range(self.d_feature)]
        self.encoder_linear = [nn.Linear(self.seqlen, self.prelen).to(self.device) for _ in range(self.d_feature)]
        self.decoder = [torch.nn.TransformerDecoder(self.decoder_layer[i], num_layers=args.d_layers) for i in range(self.d_feature)]

        self.g_c = nn.Sequential(
            nn.Linear(self.d_model, self.s),
            nn.Hardsigmoid()
        ).to(self.device)

        self.g_s = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softplus()
        ).to(self.device)

        self.tao = torch.ones((self.s, self.s)).to(self.device)
        self.tao[:, 0] = 0
        self.tao[:, -1] = 0
        self.tao[1] *= -1
        self.tao[0] = 0
        self.tao[0, 0] = 1
        self.tao[2:, 1:-1] = torch.eye(self.s - 2)

        self.z = torch.zeros((1, self.s))
        self.z[:, :2] = 1

        self.batch_proceeded = None

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        # model input: bs,seqlen,dim,5 ,bs,prelen,dim,4 model output bs prelen,dim
        # x shape:bs,seqlen,dim,5 5 means that one dimension is historical data and the other dimensions are time-dependent variables.
        # covariable shape :bs,prelen,dim,4
        for i in range(batch_x.shape[-1]):
            if i == 0:
                batch_x_mark_concat = batch_x_mark[:, -self.seqlen:].unsqueeze(-1)
            else:
                batch_x_mark_concat = torch.cat((batch_x_mark_concat, batch_x_mark[:, -self.seqlen:].unsqueeze(-1)),
                                                dim=-1)
        batch_x_mark_concat = batch_x_mark_concat.permute(0, 1, 3, 2)
        history_input = torch.cat((batch_x.unsqueeze(-1), batch_x_mark_concat), dim=-1)

        for j in range(batch_y.shape[-1]):
            if j == 0:
                future_input = batch_y_mark[:, -self.prelen:].unsqueeze(-2)
            else:
                future_input = torch.cat((future_input, batch_y_mark[:, -self.prelen:].unsqueeze(-2)), dim=-2)

        history_input = history_input.permute(0, 2, 1, 3)  # bs,dim,seqlen,5
        future_input = future_input.permute(0, 2, 1, 3)  # bs,dim prelen,1

        input_embedding = self.input_fc(history_input)
        covariable_embedding = self.co_embedding(future_input)

        position_input_embedding = self.pos_encoder_emb(input_embedding)
        covariable_embedding = self.pos_decoder_emb(covariable_embedding)

        total_decoder_output = torch.zeros_like(covariable_embedding)
        for single in range(input_embedding.shape[1]):  # 每个对象单独建模
            encoder_input = position_input_embedding[:, single]  # bs,seqlen,hidden_dimension
            covariable_embedding_single = covariable_embedding[:, single]  # bs,prelen,hidden_dimension
            encoder_output = self.encoder[single](encoder_input)
            decoder_memory = self.encoder_linear[single](encoder_output.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_output = self.decoder[single](tgt=covariable_embedding_single, memory=decoder_memory,
                                                  tgt_mask=nn.Transformer().generate_square_subsequent_mask(
                                                      self.prelen).to(self.device))
            total_decoder_output[:, single] = decoder_output

        prediction = torch.zeros(size=(batch_y_mark.shape[0], self.prelen, batch_x.shape[2])).squeeze(-1).to(
            self.device)
        for day in range(self.prelen):
            if day == 0:
                a = self.g_c(total_decoder_output[:, :, day]).unsqueeze(-1).to(self.device)

            if self.batch_proceeded == None:
                self.batch_proceeded = True
                self.tao = torch.cat(a.shape[0] * a.shape[1] * [self.tao], dim=0) \
                    .view(a.shape[0], a.shape[1], self.s, self.s).to(self.device)  # Batch processing for subsequent matrix multiplication
                self.z = torch.cat(a.shape[0] * a.shape[1] * [self.z], dim=0) \
                    .view(a.shape[0], a.shape[1], self.s, 1).to(self.device)

            c = self.g_c(total_decoder_output[:, :, day]).unsqueeze(-1)
            a = torch.matmul(self.tao, a) + c
            sigmoid = self.g_s(total_decoder_output[:, :, day]).squeeze(-1)
            y = torch.matmul(self.z.permute(0, 1, 3, 2), a).squeeze(-1).squeeze(-1)
            prediction[:, day] = y + sigmoid

        return prediction