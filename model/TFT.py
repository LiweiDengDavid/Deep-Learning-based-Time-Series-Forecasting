from torch import nn
import torch

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self,input_dimension):
        super(GLU, self).__init__()
        self.fc1=nn.Linear(input_dimension,input_dimension)
        self.fc2=nn.Linear(input_dimension,input_dimension)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        sig=self.sigmoid(self.fc1(x))
        x=self.fc2(x)
        return torch.mul(sig,x)


class GRN(nn.Module):
    def __init__(self,input_dimension,hidden_dimension,
                 output_dimension,context_dimension=None,dropout=0.2):
        super(GRN, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.context_dimension = context_dimension
        self.dropout = dropout

        self.dense1 = nn.Linear(input_dimension,hidden_dimension)
        self.ELU = nn.ELU()
        self.dense12 = nn.Linear(hidden_dimension,output_dimension)

        if input_dimension != output_dimension:
            self.skip_layer = nn.Linear(input_dimension,output_dimension)

        if self.context_dimension!=None:
            self.context_change = nn.Linear(context_dimension,hidden_dimension)

        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(input_dimension=output_dimension)

        self.ln = nn.LayerNorm(output_dimension)

    def forward(self,x,context=None):

        if self.input_dimension != self.output_dimension:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.dense1(x)

        if context != None:
            x += self.context_change(context)

        x = self.ELU(x)
        x = self.dense12(x)
        x = self.dropout(x)
        x = self.glu(x)
        x = self.ln(x+residual)

        return x


class VSN(nn.Module):
    # Variable Selection Network
    def __init__(self, input_dimension, hidden_dimension,
                 output_dimension,feature_dimension,
                 drop_out=0.2,device='cpu',context=None):
        super(VSN, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.drop_out = drop_out
        self.context = context
        self.feature_dimension = feature_dimension
        self.device = device

        self.softmax = nn.Softmax(dim=-1)
        self.flatten_grn = GRN(input_dimension=int(feature_dimension*input_dimension),
                               hidden_dimension=hidden_dimension,context_dimension=hidden_dimension,
                               output_dimension=output_dimension)
        self.normal_grn = ([GRN(input_dimension=input_dimension,hidden_dimension=hidden_dimension,output_dimension=hidden_dimension,context_dimension=hidden_dimension).to(self.device) for _ in range(self.feature_dimension)])

    def forward(self,x,context):
        flatten_x = x.view(x.shape[0],x.shape[1],x.shape[2],-1) #bs,seqlen,dim,feature_dimension*hidden_dimension
        context = context.view(x.shape[0],x.shape[1],x.shape[2],-1)
        weight = self.flatten_grn(flatten_x,context) #bs,seqlen,dim,feature_dimension*hidden_dimension
        weight_softmax = self.softmax(weight)

        grn_total = torch.zeros(size=(x.shape[0],x.shape[1],x.shape[2],
                                      self.hidden_dimension,self.feature_dimension)).to(self.device)
        for i in range(self.feature_dimension):
            a = x[:,:,:,i]
            grn_total[:,:,:,:,i] = self.normal_grn[i](x[:,:,:,i])

        a,b = grn_total,weight_softmax.unsqueeze(-1)
        output = torch.matmul(grn_total , weight_softmax.unsqueeze(-1))

        return output.squeeze(-1)



class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0,device='cpu'):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        self.v_layer = nn.Linear(d_model, self.d_v)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q).to(self.device) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k).to(self.device) for _ in range(self.n_head)]
        )
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)


    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=-1)
        attn = torch.stack(attns, dim=-1)

        outputs = torch.mean(head, dim=-1)
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn



class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=0)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.permute(0,1,3,2))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            mask = torch.cat((attn.shape[0]*attn.shape[1]*[mask]),dim=0).view(attn.shape[0],attn.shape[1],attn.shape[2],attn.shape[3])
            attn = attn.masked_fill(mask.byte().bool(), -1e9)
        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn





class TFT(nn.Module):
    def __init__(self,args):
        super(TFT, self).__init__()

        self.seqlen = args.seq_len
        self.prelen = args.pred_len
        self.static_dimension = 1
        self.past_dimension = 1+args.d_mark
        self.future_dimension = args.d_mark
        self.hidden_dimension = args.d_dimension
        self.quantiles = len(args.quantiles)
        self.prediction_number = args.d_feature
        self.device= args.device


        self.static_embedding = [nn.Linear(1,self.hidden_dimension).to(self.device) for _ in range(self.static_dimension)]
        self.past_embedding = [nn.Linear(1, self.hidden_dimension).to(self.device) for _ in range(self.past_dimension)]
        self.future_embedding = [nn.Linear(1,self.hidden_dimension).to(self.device) for _ in range(self.future_dimension)]

        self.vsn_past = VSN(input_dimension=self.hidden_dimension,
                            hidden_dimension=self.hidden_dimension,
                            output_dimension=self.past_dimension,
                            feature_dimension=self.past_dimension,
                            device=self.device)
        self.vsn_future = VSN(input_dimension=self.hidden_dimension,
                              hidden_dimension=self.hidden_dimension,
                              output_dimension=self.future_dimension,
                              feature_dimension=self.future_dimension,
                              device=self.device)

        self.lstm_encoder = [nn.LSTM(input_size=self.hidden_dimension,
                                    hidden_size=self.hidden_dimension,
                                    batch_first=True).to(self.device) for _ in range(self.prediction_number)]
        self.lstm_decoder = [nn.LSTM(input_size=self.hidden_dimension,
                                    hidden_size=self.hidden_dimension,
                                    batch_first=True).to(self.device) for _ in range(self.prediction_number)]

        self.glu_before_add = GLU(input_dimension=self.hidden_dimension)
        self.glu_after_add = GLU(input_dimension=self.hidden_dimension)
        self.ln = nn.LayerNorm(self.hidden_dimension)

        self.static_enrichment = GRN(input_dimension=self.hidden_dimension,
                                     hidden_dimension=self.hidden_dimension,
                                     output_dimension=self.hidden_dimension,
                                     context_dimension=self.hidden_dimension)

        self.masked_multive_attention = InterpretableMultiHeadAttention(n_head=4,
                                                                        d_model=self.hidden_dimension,
                                                                        device=self.device)

        self.glu2 = GLU(input_dimension=self.hidden_dimension)

        self.po_wise_feed = GRN(input_dimension=self.hidden_dimension,
                                hidden_dimension=int(self.hidden_dimension*1.2),
                                output_dimension=self.hidden_dimension)

        self.glu3 = GLU(input_dimension=self.hidden_dimension)

        self.dense = nn.Linear(self.hidden_dimension,self.quantiles)



    def encoder(self,x):
        # x shape: bs,seqlen,dim,hidden_dimension
        output_record,h_record,c_record = [],[],[]
        for i in range(x.shape[-2]):
            input = x[:,:,i]
            output,(h,c) = self.lstm_encoder[i](input)
            output_record.append(output.unsqueeze(1))
            h_record.append(h.unsqueeze(1))
            c_record.append(c.unsqueeze(1))
        output_record = torch.cat(output_record,dim=1).to(self.device)
        h_record = torch.cat(h_record,dim=1)
        c_record = torch.cat(c_record,dim=1)

        return output_record,(h_record,c_record)

    def decoder(self,x,h_past=None):
        # x shape: bs,prelen,dim,hidden_dimension
        h_record, c_record = h_past


class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self,input_dimension):
        super(GLU, self).__init__()
        self.fc1=nn.Linear(input_dimension,input_dimension)
        self.fc2=nn.Linear(input_dimension,input_dimension)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        sig=self.sigmoid(self.fc1(x))
        x=self.fc2(x)
        return torch.mul(sig,x)


class GRN(nn.Module):
    def __init__(self,input_dimension,hidden_dimension,
                 output_dimension,context_dimension=None,dropout=0.2):
        super(GRN, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.context_dimension = context_dimension
        self.dropout = dropout

        self.dense1 = nn.Linear(input_dimension,hidden_dimension)
        self.ELU = nn.ELU()
        self.dense12 = nn.Linear(hidden_dimension,output_dimension)

        if input_dimension != output_dimension:
            self.skip_layer = nn.Linear(input_dimension,output_dimension)

        if self.context_dimension!=None:
            self.context_change = nn.Linear(context_dimension,hidden_dimension)
            self.context_change2 = nn.Linear(96,64)
            self.context_change3 = nn.Linear(96, 96)

        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(input_dimension=output_dimension)

        self.ln = nn.LayerNorm(output_dimension)

    def forward(self,x,context=None):

        if self.input_dimension != self.output_dimension:
            try:
                residual = self.skip_layer(x)
            except:
                residual = self.skip_layer(x)
        else:
            residual = x

        x = self.dense1(x)

        if context != None:
            x += self.context_change(context)



        x = self.ELU(x)
        x = self.dense12(x)
        x = self.dropout(x)
        x = self.glu(x)
        x = self.ln(x+residual)

        return x


class VSN(nn.Module):
    # Variable Selection Network
    def __init__(self, input_dimension, hidden_dimension,
                 output_dimension,feature_dimension,
                 drop_out=0.2,device='cpu',context=None):
        super(VSN, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.drop_out = drop_out
        self.context = context
        self.feature_dimension = feature_dimension
        self.device = device

        self.softmax = nn.Softmax(dim=-1)
        self.flatten_grn = GRN(input_dimension=int(feature_dimension*input_dimension),
                               hidden_dimension=hidden_dimension,context_dimension=hidden_dimension,
                               output_dimension=output_dimension)
        self.normal_grn = ([GRN(input_dimension=input_dimension,hidden_dimension=hidden_dimension,output_dimension=hidden_dimension,context_dimension=hidden_dimension).to(self.device) for _ in range(self.feature_dimension)])

    def forward(self,x,context):
        flatten_x = x.view(x.shape[0],x.shape[1],x.shape[2],-1) #bs,seqlen,dim,feature_dimension*hidden_dimension
        length,dim = context.shape[1],context.shape[2]
        context = context.contiguous().view(x.shape[0],length,dim,-1)
        weight = self.flatten_grn(flatten_x,context) #bs,seqlen,dim,feature_dimension*hidden_dimension
        weight_softmax = self.softmax(weight)

        grn_total = torch.zeros(size=(x.shape[0],x.shape[1],x.shape[2],
                                      self.hidden_dimension,self.feature_dimension)).to(self.device)
        for i in range(self.feature_dimension):
            a = x[:,:,:,i]
            grn_total[:,:,:,:,i] = self.normal_grn[i](x[:,:,:,i])

        a,b = grn_total,weight_softmax.unsqueeze(-1)
        output = torch.matmul(grn_total , weight_softmax.unsqueeze(-1))

        return output.squeeze(-1)



class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0,device='cpu'):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        self.v_layer = nn.Linear(d_model, self.d_v)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q).to(self.device) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k).to(self.device) for _ in range(self.n_head)]
        )
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)


    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=-1)
        attn = torch.stack(attns, dim=-1)

        outputs = torch.mean(head, dim=-1)
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn



class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=0)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.permute(0,1,3,2))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            mask = torch.cat((attn.shape[0]*attn.shape[1]*[mask]),dim=0).view(attn.shape[0],attn.shape[1],attn.shape[2],attn.shape[3])
            attn = attn.masked_fill(mask.byte().bool(), -1e9)
        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn





class TFT(nn.Module):
    def __init__(self,args):
        super(TFT, self).__init__()

        self.seqlen = args.seq_len
        self.prelen = args.pred_len
        self.static_dimension = 1
        self.past_dimension = args.d_mark+1
        self.future_dimension = args.d_mark
        self.hidden_dimension = args.d_dimension
        self.quantiles = len(args.quantiles)
        self.prediction_number = args.d_feature
        self.device= args.device


        self.static_embedding = [nn.Linear(1,self.hidden_dimension).to(self.device) for _ in range(self.static_dimension)]
        self.past_embedding = [nn.Linear(1, self.hidden_dimension).to(self.device) for _ in range(self.past_dimension)]
        self.future_embedding = [nn.Linear(1,self.hidden_dimension).to(self.device) for _ in range(self.future_dimension)]

        self.vsn_past = VSN(input_dimension=self.hidden_dimension,
                            hidden_dimension=self.hidden_dimension,
                            output_dimension=self.past_dimension,
                            feature_dimension=self.past_dimension,
                            device=self.device)
        self.vsn_future = VSN(input_dimension=self.hidden_dimension,
                              hidden_dimension=self.hidden_dimension,
                              output_dimension=self.future_dimension,
                              feature_dimension=self.future_dimension,
                              device=self.device)

        self.lstm_encoder = [nn.LSTM(input_size=self.hidden_dimension,
                                    hidden_size=self.hidden_dimension,
                                    batch_first=True).to(self.device) for _ in range(self.prediction_number)]
        self.lstm_decoder = [nn.LSTM(input_size=self.hidden_dimension,
                                    hidden_size=self.hidden_dimension,
                                    batch_first=True).to(self.device) for _ in range(self.prediction_number)]

        self.glu_before_add = GLU(input_dimension=self.hidden_dimension)
        self.glu_after_add = GLU(input_dimension=self.hidden_dimension)
        self.ln = nn.LayerNorm(self.hidden_dimension)

        self.static_enrichment = GRN(input_dimension=self.hidden_dimension,
                                     hidden_dimension=self.hidden_dimension,
                                     output_dimension=self.hidden_dimension,
                                     context_dimension=self.hidden_dimension)

        self.masked_multive_attention = InterpretableMultiHeadAttention(n_head=4,
                                                                        d_model=self.hidden_dimension,
                                                                        device=self.device)

        self.glu2 = GLU(input_dimension=self.hidden_dimension)

        self.po_wise_feed = GRN(input_dimension=self.hidden_dimension,
                                hidden_dimension=int(self.hidden_dimension*1.2),
                                output_dimension=self.hidden_dimension)

        self.glu3 = GLU(input_dimension=self.hidden_dimension)

        self.dense = nn.Linear(self.hidden_dimension,self.quantiles)



    def encoder(self,x):
        # x shape: bs,seqlen,dim,hidden_dimension
        output_record,h_record,c_record = [],[],[]
        for i in range(x.shape[-2]):
            input = x[:,:,i]
            output,(h,c) = self.lstm_encoder[i](input)
            output_record.append(output.unsqueeze(1))
            h_record.append(h.unsqueeze(1))
            c_record.append(c.unsqueeze(1))
        output_record = torch.cat(output_record,dim=1).to(self.device)
        h_record = torch.cat(h_record,dim=1)
        c_record = torch.cat(c_record,dim=1)

        return output_record,(h_record,c_record)

    def decoder(self,x,h_past=None):
        # x shape: bs,prelen,dim,hidden_dimension
        h_record, c_record = h_past
        output_record = []
        for i in range(x.shape[-2]):
            input = x[:,:,i]
            h,c = h_record[:,i],c_record[:,i]
            output,(h,c) = self.lstm_encoder[i](input,(h,c))
            output_record.append(output.unsqueeze(1))

        output_record = torch.cat(output_record,dim=1)

        return output_record

    def forward(self,batch_x, batch_y, batch_x_mark, batch_y_mark,category):

        # model inputs include historical data, historical time, future time and features of the predicted object (here the features are represented as the mean category of the dataset)

        # batch_x:bs,seqlen,dim
        # batch_x_mark:bs,seqlen,time_dimension
        # batch_y_mark:bs,seqlen+label_len,time_dimension
        # category:bs,seqlen,dim
        static = category[:,0].unsqueeze(-1)
        for i in range(batch_x.shape[-1]):
            if i == 0:
                batch_x_mark_concat = batch_x_mark.unsqueeze(-1)
            else:
                batch_x_mark_concat = torch.cat((batch_x_mark_concat,batch_x_mark.unsqueeze(-1)),dim=-1)
        batch_x_mark_concat = batch_x_mark_concat.permute(0,1,3,2)
        past = torch.cat((batch_x.unsqueeze(-1),batch_x_mark_concat),dim=-1)
        future_time = batch_y_mark[:,-self.prelen:]
        for i in range(batch_x.shape[-1]):
            if i == 0:
                future_input = future_time.unsqueeze(-1)
            else:
                future_input = torch.cat((future_input,future_time.unsqueeze(-1)),dim=-1)
        future = future_input.permute(0,1,3,2)


        # input shape: static(bs,dim,static_dimension)
        # past:(bs,seqlen,dim,past_feature_dimension)
        # future(bs,prelen,dim,future_feature_dimension)
        static_proceeded = torch.zeros(size=(static.shape[0],static.shape[1],
                                             self.static_dimension,
                                             self.hidden_dimension)).to(self.device)

        for i in range(self.static_dimension):
            static_proceeded[:,:,i,:] = self.static_embedding[i](static[:,:,i].unsqueeze(-1))

        for i in range(past.shape[1]):
            if i == 0:
                static_proceeded_batch = static_proceeded.unsqueeze(-1)
            else:
                static_proceeded_batch = torch.cat((static_proceeded_batch,static_proceeded.unsqueeze(-1)),dim=-1)
        static_proceeded = static_proceeded_batch.permute(0,4,1,2,3)

        past_proceeded = torch.zeros(size=(past.shape[0],
                                           past.shape[1],
                                           past.shape[2],
                                           past.shape[3],
                                           self.hidden_dimension)).to(self.device)
        for i in range(self.past_dimension):
            past_proceeded[:,:,:,i] += self.past_embedding[i](past[:, :, :, i].unsqueeze(-1))

        future_proceeded = torch.zeros(size=(future.shape[0],future.shape[1],
                                             future.shape[2],future.shape[3],
                                             self.hidden_dimension)).to(self.device)
        for i in range(self.future_dimension):
            future_proceeded[:,:,:,i]= self.future_embedding[i](future[:,:,:,i].unsqueeze(-1))

        # Feature Screening
        past_variable_selected = self.vsn_past(past_proceeded,static_proceeded).to(self.device)  # bs,seqlen,dim,hidden_dimension
        future_variable_selected = self.vsn_future(future_proceeded,static_proceeded).to(self.device) # bs,prelen,dim,hidden_dimension


        past_lstm, past_hidden = self.encoder(past_variable_selected)
        future_lstm= self.decoder(future_variable_selected,past_hidden )
        lstm_output = torch.cat((past_lstm,future_lstm),dim=2).permute(0,2,1,3)
        lstm_input = torch.cat((past_variable_selected,future_variable_selected),dim=1)
        lstm_proceeded = self.ln(self.glu_after_add(self.glu_before_add(lstm_output) + lstm_input))

        # static enrichment
        static_embedding_enrich_context = torch.cat((static_proceeded.squeeze(-2),static_proceeded.squeeze(-2)),dim=1)
        enriched_past_future = self.static_enrichment(lstm_proceeded,static_embedding_enrich_context)

        enriched_past_future = enriched_past_future.permute(0, 2, 1, 3) # bs,seqlen+prelen,dim,hidden_dimension ----- bs,dim,seqlen+prelen,hidden_dimension
        mask = torch.tril(torch.ones(enriched_past_future.shape[2], enriched_past_future.shape[2]).to(self.device)) - \
               torch.eye(enriched_past_future.shape[2], enriched_past_future.shape[2]).to(self.device)
        mask = mask.bool()
        attention_output_future,attention_weight = self.masked_multive_attention(q=enriched_past_future,
                                                                k=enriched_past_future,
                                                                v=enriched_past_future,
                                                                mask=mask)
        attention_output_future_add_and_norm = self.ln(self.glu2(attention_output_future[:,:,-self.prelen:]) +
                                                       enriched_past_future[:,:,-self.prelen:])

        future_position_wise_feed_forward = self.po_wise_feed(attention_output_future_add_and_norm).permute(0,2,1,3)

        future_position_wise_feed_forward_add_and_norm = self.ln(self.glu3(future_position_wise_feed_forward) + \
                                                         lstm_proceeded[:,-self.prelen:])

        prediction = self.dense(future_position_wise_feed_forward_add_and_norm)

        return prediction

















        output_record = []
        for i in range(x.shape[-2]): # 对每个股票单独建模
            input = x[:,:,i]
            h,c = h_record[:,i],c_record[:,i]
            output,(h,c) = self.lstm_encoder[i](input,(h,c))
            output_record.append(output.unsqueeze(1))

        output_record = torch.cat(output_record,dim=1)

        return output_record

    def forward(self,static,past,future):
        # input shape: static(bs,dim,static_dimension)
        # past:(bs,seqlen,dim,past_feature_dimension)
        # future(bs,prelen,dim,future_feature_dimension)

        # 对静态变量的embedding
        static_proceeded = torch.zeros(size=(static.shape[0],static.shape[1],
                                             self.static_dimension,
                                             self.hidden_dimension)).to(self.device)

        for i in range(self.static_dimension):
            static_proceeded[:,:,i,:] = self.static_embedding[i](static[:,:,i].unsqueeze(-1))

        for i in range(max(past.shape[1],self.prelen)):# 将static_proceeded重复batchsize次组成矩阵，便于后续批量化处理
            if i == 0:
                static_proceeded_batch = static_proceeded.unsqueeze(-1)
            else:
                static_proceeded_batch = torch.cat((static_proceeded_batch,static_proceeded.unsqueeze(-1)),dim=-1)
        static_proceeded = static_proceeded_batch.permute(0,4,1,2,3)

        # 对动态时变量的embedding
        past_proceeded = torch.zeros(size=(past.shape[0],
                                           past.shape[1],
                                           past.shape[2],
                                           past.shape[3],
                                           self.hidden_dimension)).to(self.device)
        for i in range(self.past_dimension):
            past_proceeded[:,:,:,i] += self.past_embedding[i](past[:, :, :, i].unsqueeze(-1))

        # 对动态时不变量的embedding
        ####### 一般这里都是一维的特征，也就是下面的for循环只会执行一次
        future_proceeded = torch.zeros(size=(future.shape[0],future.shape[1],
                                             future.shape[2],future.shape[3],
                                             self.hidden_dimension)).to(self.device)
        for i in range(self.future_dimension):
            future_proceeded[:,:,:,i]= self.future_embedding[i](future[:,:,:,i].unsqueeze(-1))

        # 特征筛选
        past_variable_selected = self.vsn_past(past_proceeded,static_proceeded[:,:self.seqlen]).to(self.device)  # bs,seqlen,dim,hidden_dimension
        future_variable_selected = self.vsn_future(future_proceeded,static_proceeded[:,:self.prelen]).to(self.device) # bs,prelen,dim,hidden_dimension

        # 输入lstm，分为encoder和decoder
        past_lstm, past_hidden = self.encoder(past_variable_selected)
        future_lstm= self.decoder(future_variable_selected,past_hidden )
        lstm_output = torch.cat((past_lstm,future_lstm),dim=2).permute(0,2,1,3) # 将seqlen和prelen拼接起来
        lstm_input = torch.cat((past_variable_selected,future_variable_selected),dim=1) # 后面用于残差相加
        lstm_proceeded = self.ln(self.glu_after_add(self.glu_before_add(lstm_output) + lstm_input))

        # static enrichment
        static_embedding_enrich_context = torch.cat((static_proceeded.squeeze(-2),static_proceeded.squeeze(-2)),dim=1)  # 处理一下static_embedding,便于后续批量操作
        enriched_past_future = self.static_enrichment(lstm_proceeded,static_embedding_enrich_context[:,:lstm_proceeded.shape[1]])

        enriched_past_future = enriched_past_future.permute(0, 2, 1, 3) # bs,seqlen+prelen,dim,hidden_dimension ----- bs,dim,seqlen+prelen,hidden_dimension
        mask = torch.tril(torch.ones(enriched_past_future.shape[2], enriched_past_future.shape[2]).to(self.device)) - \
               torch.eye(enriched_past_future.shape[2], enriched_past_future.shape[2]).to(self.device)  # 右上三角矩阵，用于mask，使得未来的特征只收到之前时间的影响，不受更未来特征的影响
        mask = mask.bool()
        attention_output_future,attention_weight = self.masked_multive_attention(q=enriched_past_future,
                                                                k=enriched_past_future,
                                                                v=enriched_past_future,
                                                                mask=mask)
        attention_output_future_add_and_norm = self.ln(self.glu2(attention_output_future[:,:,-self.prelen:]) +
                                                       enriched_past_future[:,:,-self.prelen:])

        future_position_wise_feed_forward = self.po_wise_feed(attention_output_future_add_and_norm).permute(0,2,1,3)

        future_position_wise_feed_forward_add_and_norm = self.ln(self.glu3(future_position_wise_feed_forward) + \
                                                         lstm_proceeded[:,-self.prelen:])

        prediction = self.dense(future_position_wise_feed_forward_add_and_norm)

        return prediction
















