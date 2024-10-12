import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.linear_layer import LinearLayer
from layers.static_combine_and_mask import StaticCombineAndMask
from layers.add_and_norm import AddAndNorm
from layers.gated_residual_network import GatedResidualNetwork
from layers.gated_linear_unit import GLU
from layers.linear_layer import LinearLayer
from layers.lstm_combine_and_mask import LSTMCombineAndMask
from layers.interpretable_multi_head_attention import InterpretableMultiHeadAttention

class TFT(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, args):
        super(TFT, self).__init__()

        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.outputfc = nn.Linear(1,160)
        self.input_size = args.d_feature
        self.hidden_layer_size = 160
        self.batch_first = True
        self.dropout_rate = args.dropout
        self.num_heads = args.n_heads
        self.output_size = args.c_out


        self.category_counts = [0,1,2,3,4,5]
        self._static_input_loc = [1]
        self._input_obs_loc = [1]
        self._known_regular_input_idx = [0]
        self._known_categorical_input_idx = [0]
        self.quantiles = [0.1,0.5,0.9]

        self.history_embeddings = nn.ModuleList()
        for i in range(self.input_size):
            # embedding = nn.Embedding(self.input_size,self.hidden_layer_size)
            embedding = nn.Linear(1,160)
            self.history_embeddings.append(embedding)

        self.future_embeddings = nn.ModuleList()
        for i in range(4):
            embedding = nn.Linear(1,160)
            self.future_embeddings.append(embedding)

        self.static_input_layer = nn.Linear(self.input_size+4, self.hidden_layer_size)
        self.time_varying_embedding_layer = LinearLayer(input_size=1, size=self.hidden_layer_size,
                                                        use_time_distributed=True, batch_first=self.batch_first)

        self.static_combine_and_mask = StaticCombineAndMask(
            input_size=self.input_size,
            num_static=1,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            additional_context=None,
            use_time_distributed=False,
            batch_first=self.batch_first)

        self.static_context_variable_selection_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)
        self.static_context_enrichment_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)
        self.static_context_state_h_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)
        self.static_context_state_c_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)

        self.historical_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.label_len,
                num_inputs=self.input_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.future_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.label_len,
                num_inputs=4,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size,
                                    batch_first=self.batch_first)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size,
                                    batch_first=self.batch_first)

        self.lstm_glu = GLU(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            batch_first=self.batch_first)

        self.lstm_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.static_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=True,
                batch_first=self.batch_first)

        self.self_attn_layer = InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size,
                                                               dropout_rate=self.dropout_rate)

        self.self_attention_glu = GLU(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            batch_first=self.batch_first)

        self.self_attention_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.decoder_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            return_gate=False,
            batch_first=self.batch_first)

        self.final_glu = GLU(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            batch_first=self.batch_first)

        self.final_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.output_layer = LinearLayer(
            input_size=self.hidden_layer_size,
            size=self.output_size * len(self.quantiles),
            use_time_distributed=True,
            batch_first=self.batch_first)

    def get_tft_embeddings(self, all_inputs):

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
              all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[:,:, i].long())
            for i in range(num_categorical_variables)
        ]

        if self._static_input_loc:
            static_inputs = []


            emb_inputs = []
            for i in range(num_categorical_variables):
                if i in self._static_input_loc:
                    # [64,192,160] --> [64,1,160]
                    emb_inputs.append(embedded_inputs[i][:, 0, :])

            static_inputs += emb_inputs
            static_inputs = torch.stack(static_inputs, dim=1)

        else:
            static_inputs = None

        obs_inputs = torch.stack([
            self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float())
            for i in self._input_obs_loc
        ], dim=-1)

        unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = []
        for i in self._known_regular_input_idx:
            if i not in self._static_input_loc:
                known_regular_inputs.append(
                    self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float()))

        known_categorical_inputs = []
        for i in self._known_categorical_input_idx:
            if i + num_regular_variables not in self._static_input_loc:
                known_categorical_inputs.append(embedded_inputs[i])

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, dim=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.

        Args:
          self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1] # 192
        bs = self_attn_inputs.shape[:1][0] # [64]
        # create batch_size identity matrices
        mask = torch.cumsum(torch.eye(len_s).reshape((1, len_s, len_s)).repeat(bs, 1, 1), 1)
        return mask




    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        all_inputs = torch.cat([x_mark_dec,x_dec],dim=-1)
        encoder_steps = self.label_len

        historical_inputs = []
        future_inputs = []

        for i in range(self.input_size):
            historical_inputs.append(self.history_embeddings[i](all_inputs[:,:encoder_steps,i+4].unsqueeze(-1)))
        historical_inputs = torch.stack(historical_inputs, dim=-1)

        for i in range(4):
            future_inputs.append(self.future_embeddings[i](all_inputs[:,encoder_steps:,i].unsqueeze(-1)))
        future_inputs = torch.stack(future_inputs, dim=-1)



        static_inputs = self.static_input_layer(all_inputs[:,0,:]).unsqueeze(1)

        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)

        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)

        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)

        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)


        historical_features, historical_flags, _ = self.historical_lstm_combine_and_mask(historical_inputs,
                                                                                         static_context_variable_selection)
        future_features, future_flags, _ = self.future_lstm_combine_and_mask(future_inputs,
                                                                             static_context_variable_selection)


        history_lstm, (state_h, state_c) = self.lstm_encoder(historical_features, (
        static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)))

        future_lstm, _ = self.lstm_decoder(future_features, (state_h, state_c))

        lstm_layer = torch.cat([history_lstm, future_lstm], dim=1)

        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        lstm_layer, _ = self.lstm_glu(lstm_layer)

        temporal_feature_layer = self.lstm_glu_add_and_norm(lstm_layer, input_embeddings)

        expanded_static_context = static_context_enrichment.unsqueeze(1)

        enriched, _ = self.static_enrichment_grn(temporal_feature_layer, expanded_static_context)

        mask = self.get_decoder_mask(enriched)

        x, self_att = self.self_attn_layer(enriched, enriched, enriched, mask)

        x, _ = self.self_attention_glu(x)

        x = self.self_attention_glu_add_and_norm(x, enriched)

        decoder = self.decoder_grn(x)

        decoder, _ = self.final_glu(decoder)

        transformer_layer = self.final_glu_add_and_norm(decoder, temporal_feature_layer)

        outputs = self.output_layer(transformer_layer[:, -self.pred_len:, :]).unsqueeze(-1)


        outputs = outputs.view(-1,self.pred_len,self.output_size,len(self.quantiles))

        return outputs




























