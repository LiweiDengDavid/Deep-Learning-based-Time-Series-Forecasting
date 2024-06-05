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
        self.dk=self.d_ff//self.h # 多头注意力机制中每一个头的维度

        # x -->q,k,v
        # 输入为以下三个输入输出的维度都是相同的（batch_size,seq_len+pred_len,d_model)-->输出为（batch_size,seq_len+pred_len,d_ff)
        self.fc_x_to_v = nn.Linear(self.d_model, self.d_ff)
        self.fc_x_to_k = nn.Linear(self.d_model, self.d_ff)
        self.fc_x_to_q=nn.Linear(self.d_model,self.d_ff)

        # 知识信息x的非-->k的非，Q的非
        # 以下两个的输入输出的维度都是相同的（batch_size,seq_len+pred_len,d_model)-->输出为（batch_size,seq_len+pred_len,d_ff)
        self.fc_knowledge_x_to_k=nn.Linear(self.d_model,self.d_ff)
        self.fc_knowledge_x_to_q = nn.Linear(self.d_model, self.d_ff)

        # 多头注意力机制,同维度变换，然后再reshape为，h,dk,(d_ff=dk*h)，就相当于经过h个全连接层在拼接起来
        # 以下四个的输入输出的shape都是相同的 batch_size,seq_len+pred_len,d_ff)-->输出为（batch_size,seq_len+pred_len,d_ff)
        self.fc_q=nn.Linear(self.d_ff,self.d_ff)
        self.fc_k=nn.Linear(self.d_ff,self.d_ff)
        self.fc_k_fei=nn.Linear(self.d_ff,self.d_ff)
        self.fc_q_fei=nn.Linear(self.d_ff,self.d_ff)

        # 输入为（batch_size,seq_len+pred_len,d_ff)-->输出为（batch_size,seq_len+pred_len,d_model)
        self.fc_out=nn.Linear(self.d_ff,self.d_model) # 将数据的维度变回输入进attention的维度

    def attention(self, q, k, v, q_fei, k_fei):
        '''
        Args:
        :param q: 综合信息的q,shape(batch_size,seq_len+pred_len,d_ff)
        :param k: 综合信息的k,shape(batch_size,seq_len+pred_len,d_ff)
        :param v: 综合信息的v,shape(batch_size,seq_len+pred_len,d_ff)
        :param q_fei: 知识信息的q,shape(batch_size,seq_len+pred_len,d_ff)
        :param v_fei: 知识信息的v,shape(batch_size,seq_len+pred_len,d_ff)
        :return: out,shape(batch_size,seq_len+pred_len,d_model)
        '''

        # 多头注意力q,k,q_fei,k_fei的shape变为（batch_size,h,seq_len+pred_len,dk) ,dk*h=d_ff
        q=self.fc_q(q).reshape(q.shape[0],self.h,q.shape[1],-1)
        k = self.fc_k(k).reshape(k.shape[0],self.h,k.shape[1],-1)
        q_fei = self.fc_q_fei(q_fei).reshape(q_fei.shape[0],self.h ,q_fei.shape[1],-1)
        k_fei = self.fc_k_fei(k_fei).reshape(k_fei.shape[0], self.h, k_fei.shape[1],-1)

        d = q.shape[-1]
        d_fei = q_fei.shape[-1] # 为了计算attention时候作为比例因子,就为dk

        # K的最后两个维度进行一个转置，从(batch_size,h,seq_len+pred_len,dk)-->((batch_size,h,dk,seq_len+pred_len),为了能够进行矩阵的乘法
        # att and att_fei shape(batch_size,h,seq_len+pred_len,seq_len+pred_len)，表示的就是每一时间点之间的注意力关系
        att = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(2 * d))
        att_fei = torch.matmul(q_fei, k_fei.transpose(-1, -2)) / (math.sqrt(2 * d_fei))
        att_final = att + att_fei
        score = torch.softmax(att_final, dim=-1)  # score的shape(batch_size,h,seq_len+pred_len,dk)
        score=self.drop_out(score)
        # 为了使得V可以和score进行矩阵乘法，因此把V reshape成(batch_size,h,seq_len+pred_len,dk)
        v=v.reshape(v.shape[0],self.h,v.shape[1],-1) # v从(batch_size,seq_len+pred_len,d_ff)变为(batch_size,h,seq_len+pred_len,dk)
        out = torch.matmul(score, v) # out shape(batch_size,seq_len+pred_len,dk)
        out=out.reshape(out.shape[0],out.shape[2],-1) # reshape为(batch_size,seq_len+pred_len,d_ff)
        out=self.fc_out(out) # 对特征维度进行降维，输入为(batch_size,seq_len+pred_len,d_ff)-->（batch_size,seq_len+pred_len,d_model）
        return out

    def forward(self,x,x_knowledge):
        '''
        Args:
        :param x: shape是(batch_size,seq_len+pred_len,d_model)
        :param x_knowledge: shape是(batch_size,seq_len+pred_len,mark),知识信息，时间维度
        :return:out: shape(batch_size,seq_len+pred_len,d_model)
        '''
        # 得到q，k，v
        # v,k,q都是(batch_size, seq_len + pred_len, d_model)-->（batch_size, seq_len + pred_len, d_ff)
        v=self.fc_x_to_v(x)
        k=self.fc_x_to_k(x)
        q=self.fc_x_to_q(x)
        # 得到知识信息的q，k，称为q_fei,k_fei
        # k_fei,q_fei（batch_size, seq_len + pred_len, d_model)-->输出为（batch_size, seq_len + pred_len, d_ff)
        k_fei=self.fc_knowledge_x_to_k(x_knowledge)
        q_fei=self.fc_knowledge_x_to_q(x_knowledge)

        # 输入进attention中得到输出out为（batch_size,seq_len+pred_len,d_model)
        out=self.attention(q,k,v,q_fei,k_fei)
        return out

class Aliformer(nn.Module):
    def __init__(self,args,h=8,p2=0.5,loss_fn=torch.nn.MSELoss()):
        '''
        Args:
        :param h:头数 多头注意力网络中的头数
        :param d_feature:输入数据的最后一个维度dim（本数据集为 7）
        :param d_mark:知识信息的维度（本数据集为时间信息的维度 4）
        :param d_model:embedding后的维度
        :param d_ff:升为后的维度
        d_ff % h==0
        :param p2: 在训练的时候使用跨度掩码的概率为p2
        :param loss_fn 使用的loss函数
        '''
        super().__init__()
        self.h=h
        self.d_feature=args.d_feature
        self.d_mark=args.d_mark
        self.d_model=args.d_model
        self.d_ff=args.d_ff
        self.p2=p2

        # embedding层
        self.time_embeded = TimeEmbedding(d_mark=self.d_mark, d_model=self.d_model)  # 对时间维度进行embedding
        self.embeded = DataEmbedding_time_token(d_feature=self.d_feature, d_mark=self.d_mark,
                                     d_model=self.d_model)  # 把原始数据和时间维度embedding并且加在一起

        self.aliattention=AliAttention(h=self.h,d_feature=self.d_feature,
                                       d_mark=self.d_mark,d_model=self.d_model,d_ff=self.d_ff) # 初始化AliAttention层

        self.pred_len=args.pred_len
        self.label_len=args.label_len
        self.seq_len=args.seq_len

        self.out_dim=nn.Linear(self.d_model,self.d_feature) # 对特征维度进行降纬
        self.out_time=nn.Linear((self.seq_len+self.pred_len),self.pred_len) # 对时间维度进行降维

        self.choice=torch.zeros((100))
        # 为了后面的选择是否使用span masking 策略,如果选择为1，那么就使用span mask策略
        self.choice[:int(p2*100)]=1 # 把p2*100的数字变成1，剩下的是0，然后在这一个列表中随机抽样
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
        # 初始化x装的是data，x_knowledge装的是时间信息
        x = torch.zeros(enc_x.shape[0], enc_x.shape[1] + self.pred_len, enc_x.shape[2],device=enc_x.device) # x（batch_size,seq_len+pred_len,dim)
        x_knowledge = torch.zeros(enc_mark.shape[0], enc_mark.shape[1] + self.pred_len, enc_mark.shape[2],device=enc_x.device) # x_knowledge（batch_size,seq_len+pred_len,d_mark)

        if mode == 'train':
            # 将seq_len+pred_len的时间数据都拼到x上
            x[:,:self.seq_len,:]=x[:,:self.seq_len,:]+enc_x
            x[:,self.seq_len:,:]=x[:,self.seq_len:,:]+y[:,self.label_len:,:] # 将pred_len的数据拼接上去

            # 将seq_len+pred_len的时间数据都拼接到x_knowledge上
            x_knowledge[:, :self.seq_len, :] = x_knowledge[:, :self.seq_len, :] + enc_mark
            x_knowledge[:, self.seq_len:, :] = x_knowledge[:, self.seq_len:, :] + y_mark[:, self.label_len:, :] # 将pred_len的数据拼接上去

            choice=random.choice(self.choice) #self.choice中有100*int(p2)个1，剩下的都是0,在其中随机抽取来代表是使用span masking还是正常的预测
            if choice==0: #正常的预测
                label=x[:,-self.pred_len:,:].clone() # 把label分出来
                x[:,-self.pred_len:,:]=0 # 把label对应的部分设置为0
                star=self.seq_len # 需要预测的部分对应的起始index

            else: # span masking
                star=random.choice(range(self.seq_len)) # 在0~seq_len中随机选取初始点
                label=x[:,star:star+self.pred_len,:].clone() # label就是strar-star+pred_len部分
                x[:,star:star+self.pred_len,:]=0 # 把label对应的部分设置为0

        else: # 如果是在测试，那么直接把pred_len部分的数据mask掉，设置为0
        # 测试下的x shape也是(batch_size,seq_len+pred_len,dim),x_knowledge (batch_size,seq_len+pred_len,d_mark),
        # 只不过pred_len部分全部为0
            x[:, :self.seq_len, :] = x[:, :self.seq_len, :] + enc_x # 将seq_len部分的data赋值给x
            label = y[:, self.label_len:, :] # 得到label 是pred_len 部分对应的data
            x_knowledge[:, :self.seq_len, :] = x_knowledge[:, :self.seq_len, :] + enc_mark # 将seq_len部分的时间数据赋值给x_knowledge
            x_knowledge[:, self.seq_len:, :] = x_knowledge[:, self.seq_len:, :] + y_mark[:, self.label_len:,:] # 将pred_len的数据也赋值给x_knowledge,因为知识数据是可以预先知道的
            star=self.seq_len # 需要预测的部分对应的index
        '''
        Args:
        :param x: shape是(batch_size,seq_len+pred_len,d_feature)
        :param x_knowledge: shape是(batch_size,seq_len+pred_len,d_mark),知识信息，时间维度
        '''

        x=self.embeded(x,x_knowledge) # 输出x的shape为 (batch_size,seq_len+pred_len，d_model）
        x_knowledge=self.time_embeded(x_knowledge) # x_knowledge shape为（batch_size,seq_len+pred_len,d_model）

        x=self.aliattention(x,x_knowledge)
        x_new=x.clone()

        pred=self.out_time(x.permute(0,2,1)).permute(0,2,1)
        x_new[:,star:star+self.pred_len,:]=pred # 把使用AliAttention预测的值拼接回去

        #  堆叠12层的AliAttention
        # 输入、输出x都为（batch_size,seq_len+pred_len,d_model)
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

        # 输入为(batch_size,seq_len+pred_len,d_model)-->输出为(batch_size,seq_len+pred_len,d_feature)
        x_new = self.out_dim(x_new)
        # 输入为(batch_size,seq_len+pred_len,d_feature)-->输出为(batch_size,pred_len,d_feature)
        y_hat=(self.out_time(x_new.permute(0,2,1))).permute(0,2,1)

        loss=self.loss_fn(y_hat,label) # 计算损失loss
        return y_hat,loss


