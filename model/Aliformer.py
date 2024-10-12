import torch
import torch.nn as nn
import math
import random
from layers.Embed import DataEmbedding_time_token,TimeEmbedding



class AliAttention(nn.Module):
    def __init__(self,h=8,d_feature=7,d_mark=4,d_model=64,d_ff=128,drop_out=0.1):
        assert d_ff %h==0
        super().__init__()
        self.h=h
        self.d_feature=d_feature
        self.d_mark=d_mark
        self.d_model=d_model
        self.d_ff=d_ff
        self.drop_out=nn.Dropout(p=drop_out)
        self.dk=self.d_ff//self.h # Dimension of each head in the multiple attention mechanism

        # x -->q,k,v
        # （batch_size,seq_len+pred_len,d_model)-->（batch_size,seq_len+pred_len,d_ff)
        self.fc_x_to_v = nn.Linear(self.d_model, self.d_ff)
        self.fc_x_to_k = nn.Linear(self.d_model, self.d_ff)
        self.fc_x_to_q=nn.Linear(self.d_model,self.d_ff)

        #（batch_size,seq_len+pred_len,d_model)-->（batch_size,seq_len+pred_len,d_ff)
        self.fc_knowledge_x_to_k=nn.Linear(self.d_model,self.d_ff)
        self.fc_knowledge_x_to_q = nn.Linear(self.d_model, self.d_ff)

        # （batch_size,seq_len+pred_len,d_ff)-->（batch_size,seq_len+pred_len,d_ff)
        self.fc_q=nn.Linear(self.d_ff,self.d_ff)
        self.fc_k=nn.Linear(self.d_ff,self.d_ff)
        self.fc_k_fei=nn.Linear(self.d_ff,self.d_ff)
        self.fc_q_fei=nn.Linear(self.d_ff,self.d_ff)

        # （batch_size,seq_len+pred_len,d_ff)-->（batch_size,seq_len+pred_len,d_model)
        # Change the dimensions of the data back to the dimensions entered into attention
        self.fc_out=nn.Linear(self.d_ff,self.d_model)

    def attention(self, q, k, v, q_fei, k_fei):
        '''
        Args:
        :param q: Combined information of q,shape(batch_size,seq_len+pred_len,d_ff)
        :param k: Combined information of k,shape(batch_size,seq_len+pred_len,d_ff)
        :param v: Combined information of v,shape(batch_size,seq_len+pred_len,d_ff)
        :param q_fei: Knowledge information of q,shape(batch_size,seq_len+pred_len,d_ff)
        :param v_fei: Knowledge information of v,shape(batch_size,seq_len+pred_len,d_ff)
        :return: out,shape(batch_size,seq_len+pred_len,d_model)
        '''

        # The shape of the polytope attention q,k,q_fei,k_fei becomes
        # (batch_size,h,seq_len+pred_len,dk) ,dk*h=d_ff
        q=self.fc_q(q).reshape(q.shape[0],self.h,q.shape[1],-1)
        k = self.fc_k(k).reshape(k.shape[0],self.h,k.shape[1],-1)
        q_fei = self.fc_q_fei(q_fei).reshape(q_fei.shape[0],self.h ,q_fei.shape[1],-1)
        k_fei = self.fc_k_fei(k_fei).reshape(k_fei.shape[0], self.h, k_fei.shape[1],-1)

        d = q.shape[-1]
        d_fei = q_fei.shape[-1] # In order to be used as a scaling factor in the attentions calculation, it's dk.

        # (batch_size,h,seq_len+pred_len,dk)-->((batch_size,h,dk,seq_len+pred_len),To be able to multiply matrices
        # att and att_fei shape(batch_size,h,seq_len+pred_len,seq_len+pred_len),
        # which represents the multiplication of the attentional relationship between each time point
        att = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(2 * d))
        att_fei = torch.matmul(q_fei, k_fei.transpose(-1, -2)) / (math.sqrt(2 * d_fei))
        att_final = att + att_fei
        score = torch.softmax(att_final, dim=-1)  # score'shape(batch_size,h,seq_len+pred_len,dk)
        score=self.drop_out(score)
        # In order to make it possible to matrix multiply V with score, therefore reshape V into (batch_size,h,seq_len+pred_len,dk)
        v=v.reshape(v.shape[0],self.h,v.shape[1],-1) # (batch_size,seq_len+pred_len,d_ff)-->(batch_size,h,seq_len+pred_len,dk)
        out = torch.matmul(score, v) # out shape(batch_size,seq_len+pred_len,dk)
        out=out.reshape(out.shape[0],out.shape[2],-1) # reshape：(batch_size,seq_len+pred_len,d_ff)
        out=self.fc_out(out) #(batch_size,seq_len+pred_len,d_ff)-->（batch_size,seq_len+pred_len,d_model）
        return out

    def forward(self,x,x_knowledge):
        '''
        Args:
        :param x: shape:(batch_size,seq_len+pred_len,d_model)
        :param x_knowledge: shape:(batch_size,seq_len+pred_len,mark),Knowledge information, time dimension
        :return:out: shape(batch_size,seq_len+pred_len,d_model)
        '''
        # v,k,q：(batch_size, seq_len + pred_len, d_model)-->（batch_size, seq_len + pred_len, d_ff)
        v=self.fc_x_to_v(x)
        k=self.fc_x_to_k(x)
        q=self.fc_x_to_q(x)
        # Get the knowledge information q, k, called q_fei,k_fei
        # k_fei,q_fei（batch_size, seq_len + pred_len, d_model)-->（batch_size, seq_len + pred_len, d_ff)
        k_fei=self.fc_knowledge_x_to_k(x_knowledge)
        q_fei=self.fc_knowledge_x_to_q(x_knowledge)

        # Input into attentions to get output out as (batch_size,seq_len+pred_len,d_model)
        out=self.attention(q,k,v,q_fei,k_fei)
        return out

class Aliformer(nn.Module):
    def __init__(self,args,h=8,p2=0.5,loss_fn=torch.nn.MSELoss()):
        '''
        Args:
        :param h:Number of heads Number of heads in multi-attention networks
        :param d_feature:The last dimension of the input data dim
        :param d_mark:Dimensions of knowledge information
        :param d_model:Dimension after embedding
        :param d_ff:Dimension after Dimension Upgrade
        d_ff % h==0
        :param p2: The probability of using a spanning mask during training is p2
        :param loss_fn The loss function used
        '''
        super().__init__()
        self.h=h
        self.d_feature=args.d_feature
        self.d_mark=args.d_mark
        self.d_model=args.d_model
        self.d_ff=args.d_ff
        self.p2=p2

        # embedding
        self.time_embeded = TimeEmbedding(d_mark=self.d_mark, d_model=self.d_model)
        self.embeded = DataEmbedding_time_token(d_feature=self.d_feature, d_mark=self.d_mark,
                                     d_model=self.d_model)

        self.aliattention=AliAttention(h=self.h,d_feature=self.d_feature,
                                       d_mark=self.d_mark,d_model=self.d_model,d_ff=self.d_ff)
        self.pred_len=args.pred_len
        self.label_len=args.label_len
        self.seq_len=args.seq_len

        self.out_dim=nn.Linear(self.d_model,self.d_feature)
        self.out_time=nn.Linear((self.seq_len+self.pred_len),self.pred_len)

        self.choice=torch.zeros((100))
        self.choice[:int(p2*100)]=1
        self.loss_fn=loss_fn

    def forward(self, enc_x, enc_mark, y, y_mark,mode):
        '''
        Args:
        :param enc_x: (batch_size,seq_len,dim)
        :param enc_mark: (batch_size,seq_len,d_mark)
        :param y: (batch_size,label_len+pred_len,dim)
        :param y_mark: (batch_size,label_len+pred_len,d_mark)
        :param  mode: Determine if you are training
        :return:
        '''

        x = torch.zeros(enc_x.shape[0], enc_x.shape[1] + self.pred_len, enc_x.shape[2],device=enc_x.device) # x（batch_size,seq_len+pred_len,dim)
        x_knowledge = torch.zeros(enc_mark.shape[0], enc_mark.shape[1] + self.pred_len, enc_mark.shape[2],device=enc_x.device) # x_knowledge（batch_size,seq_len+pred_len,d_mark)

        if mode == 'train':
            # Spell all the time data from seq_len+pred_len onto x
            x[:,:self.seq_len,:]=x[:,:self.seq_len,:]+enc_x
            x[:,self.seq_len:,:]=x[:,self.seq_len:,:]+y[:,self.label_len:,:] # Splicing in the data from pred_len

            # 将seq_len+pred_len的时间数据都拼接到x_knowledge上
            x_knowledge[:, :self.seq_len, :] = x_knowledge[:, :self.seq_len, :] + enc_mark
            x_knowledge[:, self.seq_len:, :] = x_knowledge[:, self.seq_len:, :] + y_mark[:, self.label_len:, :]

            choice=random.choice(self.choice)
            if choice==0: #Normal forecast
                label=x[:,-self.pred_len:,:].clone() # Separate the labels.
                x[:,-self.pred_len:,:]=0 # Set the corresponding part of the label to 0
                star=self.seq_len # The starting index of the part to be predicted.

            else: # span masking
                star=random.choice(range(self.seq_len)) #Initial point randomly selected from 0~seq_len
                label=x[:,star:star+self.pred_len,:].clone() # label is the strar-star+pred_len part
                x[:,star:star+self.pred_len,:]=0 # Set the corresponding part of the label to 0

        else: # If you are testing, then just mask off the pred_len part of the data and set it to 0 points set it to 0
            x[:, :self.seq_len, :] = x[:, :self.seq_len, :] + enc_x # Assign the data in the seq_len section to x
            label = y[:, self.label_len:, :] # The label is the data corresponding to the pred_len part.
            x_knowledge[:, :self.seq_len, :] = x_knowledge[:, :self.seq_len, :] + enc_mark # Assign the time data in the seq_len section to x_knowledge
            x_knowledge[:, self.seq_len:, :] = x_knowledge[:, self.seq_len:, :] + y_mark[:, self.label_len:,:]
            star=self.seq_len


        x=self.embeded(x,x_knowledge) # (batch_size,seq_len+pred_len，d_model）
        x_knowledge=self.time_embeded(x_knowledge) # x_knowledge shape：（batch_size,seq_len+pred_len,d_model）

        x=self.aliattention(x,x_knowledge)
        x_new=x.clone()

        pred=self.out_time(x.permute(0,2,1)).permute(0,2,1)
        x_new[:,star:star+self.pred_len,:]=pred # Splice back the values predicted using AliAttention


        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new,x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new,x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new,x_knowledge)
        x_new = self.aliattention(x_new, x_knowledge)
        x_new = self.aliattention(x_new,x_knowledge)

        # (batch_size,seq_len+pred_len,d_model)-->(batch_size,seq_len+pred_len,d_feature)
        x_new = self.out_dim(x_new)
        # (batch_size,seq_len+pred_len,d_feature)-->(batch_size,pred_len,d_feature)
        y_hat=(self.out_time(x_new.permute(0,2,1))).permute(0,2,1)

        loss=self.loss_fn(y_hat,label) # calculate loss
        return y_hat,loss


