# 数据构建
import math
import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from classes.dataset.Dataset import *
from classes.dataset.Generator import *
from classes.Vocabulary import *

CONTEXT_LENGTH = 128
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 3
STEPS_PER_EPOCH = 72000
# 训练次数
epochs = 100
# device = 'cpu'
device = 'cuda'


def main():
    np.random.seed(1234)
    input_path = "./data/web/all_data"
    output_path = "./data/web/out"

    dataset = Dataset()
    # 将所有的gui数据和图片数据存入内存中 构建了 DataSet() Vocabulary() 对象  对象有有 图片 序列  预测值
    dataset.load(input_path, generate_binary_sequences=False)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)
    src_vocab_size = tgt_vocab_size = len(dataset.voc.vocabulary)
    src_len = CONTEXT_LENGTH  # 输入序列最大长度
    tgt_len = CONTEXT_LENGTH  # 输出序列最大长度

    # 模型参数
    d_lan_encoder_model = 256
    d_model = 512  # embedding 维度
    d_ff = 2048  # FeedForward dimension 前馈神经网络 提取特征
    d_k = d_v = 64  # Q K两个矩阵的维度
    n_layers = 6  # encoder和decode的层数
    n_heads = 8  # 多头的个数

    def make_data():
        enc_inputs = dataset.partial_sequences
        dec_inputs = dataset.partial_sequences
        dec_outputs = dataset.dec_out_sequences
        return enc_inputs, dec_inputs, dec_outputs, dataset.input_images
        # return torch.LongTensor(np.array(enc_inputs)), torch.LongTensor(np.array(dec_inputs)), torch.LongTensor(
        #     np.array(dec_outputs)), torch.tensor(np.array(dataset.input_images), dtype=torch.float)

    enc_inputs, dec_inputs, dec_outputs, images = make_data()

    class MyDataSet(Data.Dataset):
        def __init__(self, enc_inputs, dec_inputs, dec_outputs, images):
            super(MyDataSet, self).__init__()
            self.enc_inputs = enc_inputs
            self.dec_inputs = dec_inputs
            self.dec_outputs = dec_outputs
            self.images = images

        def __len__(self):
            return len(self.enc_inputs)

        def __getitem__(self, idx):
            return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], self.images[idx]

    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs, images), batch_size=5, shuffle=True)

    class PositionalEncoding(nn.Module):  # transformer模型
        def __init__(self, d_model, drop_out=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=drop_out)

            pe = torch.zeros(max_len, d_model)  # 位置编码矩阵
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 5000*1
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 1*256
            pe[:, 0::2] = torch.sin(position * div_term)  # 5000*512
            pe[:, 1::2] = torch.cos(position * div_term)  # 5000*512
            pe = pe.unsqueeze(0).transpose(0, 1)  # 5000*1*512
            self.register_buffer('pe', pe)

        def forward(self, x):
            """
              x: [seq_len, batch_size, d_model]
            :param x:
            :return:
            """
            x = x + self.pe[:x.size(0), :]  # self.pe[:x.size(0), :] 维度为 5*1*512
            return self.dropout(x)

    def get_attn_pad_mask(seq_q, seq_k):  # [2,5] [2,5]
        """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
            encoder和decoder都可能调用这个函数，所以seq_len视情况而定
            seq_q: [batch_size, seq_len]
            seq_k: [batch_size, seq_len]
            seq_len could be src_len or it could be tgt_len
            seq_len in seq_q and seq_len in seq_k maybe not equal
            """
        batch_size, len_q = seq_q.size()  # 2 , 5
        batch_size, len_k = seq_k.size()  # 2 , 5

        # eq(zero) is PAD token
        # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]] 2*5  2*1*5
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 增加维度   squeeze：降低维度
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # 2*5*5

    def get_attn_subsequence_mask(seq):
        """建议打印出来看看是什么的输出（一目了然）
           seq: [batch_size, tgt_len]
           """
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape))  # 生成上三角矩阵
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask  # [batch_size, tgt_len, tgt_len]

    class ScaleDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaleDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask):
            """
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_v(=len_k), d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]
            说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
            """
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
            scores.masked_fill_(attn_mask, -1e9)  # attn_mask为1的填充为了 -1e9 为0的不变 attn_mask必须为bool
            attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度做softmax 最后一个维度 softmax的和为1
            context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, len_k]   [batch_size, n_heads, len_k, d_v]
            return context, attn

    class MultiHeadAttention(nn.Module):
        """这个Attention类可以实现:
            Encoder的Self-Attention
            Decoder的Masked Self-Attention
            Encoder-Decoder的Attention
            """

        def __init__(self):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
            self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        def forward(self, input_Q, input_K, input_V, attn_mask):
            """
                    input_Q: [batch_size, len_q, d_model]
                    input_K: [batch_size, len_k, d_model]
                    input_V: [batch_size, len_v(=len_k), d_model]
                    attn_mask: [batch_size, seq_len, seq_len]
                    """
            residual, batch_size = input_Q, input_Q.size(0)
            # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
            # B: batch_size, S:seq_len, D: dim
            # (B,S,D) - proj -> (B,S,D_new) -split -> (B,S,Head,W) -trans ->(B,Head,S,W)
            #  线性变换    拆分多头
            # Q: [batch_size,n_heads,len_q,d_k]
            Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

            # K: [batch_size, n_heads, len_k, d_k] # K和V的维度一定相同，长度可以不同
            K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

            # V: [batch_size,n_heads,len_v=len_k,d_v]
            V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

            # 因为是多头，所以mask矩阵要扩充成4维的
            # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

            # context :[batch_size,n_heads,len_q,d_v] attn:[batch_size,n_heads,len_q,len_k]
            context, attn = ScaleDotProductAttention().forward(Q, K, V, attn_mask)

            # 下面将不同头的输出向量拼接在一起
            # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
            context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

            # 再做一个projection
            out_put = self.fc(context)  # [batch_size, n_heads, len_q, len_k] -> (n_heads * d_v, d_model)
            return nn.LayerNorm(d_model).to(device)(out_put + residual), attn

    class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
            super(PoswiseFeedForwardNet, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            )

        def forward(self, inputs):
            """
                  inputs: [batch_size, seq_len, d_model]
                  """
            residual = inputs
            out_put = self.fc(inputs)
            return nn.LayerNorm(d_model).to(device)(out_put + residual)

    class EncoderLayer(nn.Module):
        def __init__(self):
            super(EncoderLayer, self).__init__()
            self.enc_self_attn = MultiHeadAttention()
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, enc_input, enc_self_attn_mask):
            """
            enc_inputs: [batch_size,sec_len,d_model]
            enc_self_attn_mask:[batch_size,src_len,src_len] mask矩阵(pad_mask or sequence mask)
            :param
            :param enc_self_attn_mask:
            :return:
            """
            # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
            # 第一个enc_input * W_Q = Q
            # 第二个enc_input * W_K = K
            # 第三个enc_input * W_V = V
            enc_outputs, attn = self.enc_self_attn(enc_input
                                                   , enc_input, enc_input, enc_self_attn_mask)
            enc_outputs = self.pos_ffn(enc_outputs)
            return enc_outputs, attn

    class DecoderLayer(nn.Module):
        def __init__(self):
            super(DecoderLayer, self).__init__()
            self.dec_self_attn = MultiHeadAttention()
            self.dec_enc_attn = MultiHeadAttention()
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
            """
                    dec_inputs: [batch_size, tgt_len, d_model]
                    enc_outputs: [batch_size, src_len, d_model]
                    dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                    dec_enc_attn_mask: [batch_size, tgt_len, src_len]
                    """
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Attention层的Q(来自decoder) 和 K,V(来自encoder)
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                          enc_outputs, dec_enc_attn_mask)
            dec_outputs = self.pos_ffn(dec_outputs)
            return dec_outputs, dec_self_attn, dec_enc_attn

    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.resnet = torchvision.models.resnet50(pretrained=False)
            self.fc = nn.Linear(10000, 256)

        def forward(self, x):
            x = self.resnet(x)
            x = self.fc(x)
            return x

    img_model = ResNet().to(device)

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.src_emb = nn.Embedding(src_vocab_size, d_lan_encoder_model)
            self.pos_emb = PositionalEncoding(d_lan_encoder_model)
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        def forward(self, enc_input, image):
            """
            enc_input:[batch_size,src_len]

            :param enc_input:
            :return:
            """
            # [batch_size, src_len, d_model]
            enc_output = self.src_emb(enc_input)
            enc_output = self.pos_emb(enc_output.transpose(0, 1).transpose(0, 1))

            img_input = img_model(image)
            img_input = img_input.unsqueeze(1).repeat(1, CONTEXT_LENGTH, 1)
            rel_output = torch.cat((enc_output, img_input), dim=2)
            # Encoder输入序列的pad mask矩阵
            # [batch_size, src_len, src_len]
            enc_self_attn_mask = get_attn_pad_mask(enc_input, enc_input)
            #  在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
            enc_self_attns = []
            for layer in self.layers:  # for循环访问nn.ModuleList对象
                #  上一个block的输出enc_outputs作为当前block的输入
                # enc_outputs: [batch_size, src_len, d_model],
                # enc_self_attn: [batch_size, n_heads, src_len, src_len]
                # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
                enc_out, enc_self_attn = layer(rel_output, enc_self_attn_mask)
                enc_self_attns.append(enc_self_attn)
                return enc_out, enc_self_attns

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_emb = PositionalEncoding(d_model)
            self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

        def forward(self, dec_inputs, enc_inputs, enc_outputs):
            """
                  dec_inputs: [batch_size, tgt_len]
                  enc_inputs: [batch_size, src_len]
                  enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
                  """
            dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
            dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
            # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(
                device)  # [batch_size, tgt_len, tgt_len]
            # Masked Self_Attention：当前时刻是看不到未来的信息的
            dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)

            # Decoder 中把两种mask矩阵相加(既屏蔽了pad的信息，也屏蔽了未来时刻的信息)
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                          0).to(
                device)  # # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

            # 这个mask主要用于encoder-decoder attention层
            # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
            #                       dec_inputs只是提供expand的size的
            dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
            dec_self_attns, dec_enc_attns = [], []
            for layer in self.layers:
                # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
                # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
                dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                                 dec_enc_attn_mask)
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)
                # dec_outputs: [batch_size, tgt_len, d_model]
            return dec_outputs, dec_self_attns, dec_enc_attns

    class Transformer(nn.Module):
        def __init__(self):
            super(Transformer, self).__init__()
            self.encoder = Encoder().to(device)
            self.decoder = Decoder().to(device)
            self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        def forward(self, enc_input, dec_input, image):
            """Transformers的输入：两个序列
                    enc_inputs: [batch_size, src_len]
                    dec_inputs: [batch_size, tgt_len]
                    """
            # tensor to store decoder outputs
            # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

            # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
            # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
            enc_output, enc_self_attns = self.encoder(enc_input, image)
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_input, enc_input, enc_output)
            # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            dec_logits = self.projection(dec_outputs)
            return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

    model = Transformer().to(device)
    # 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 梯度选择
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(epochs):
        for enc_input, dec_input, dec_output, image in loader:
            """
                    enc_input: [batch_size, src_len]
                    dec_input: [batch_size, tgt_len]
                    dec_output: [batch_size, tgt_len]
                    image: [3, 3, 224]
                    """
            enc_input, dec_input, dec_output, image = torch.LongTensor(np.array(enc_input)), torch.LongTensor(
                np.array(dec_input)), torch.LongTensor(
                np.array(dec_output)), torch.tensor(np.array(image), dtype=torch.float)
            enc_input, dec_input, dec_output, image = enc_input.to(device), dec_input.to(device), dec_output.to(
                device), image.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            output, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_input, dec_input, image)
            pred, idx = output.max(1)
            # print(idx)
            # print(dec_output)
            loss = criterion(output,
                             dec_output.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def greedy_decoder(model, enc_input, start_symbol):
        """贪心编码
           For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
           target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
           Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
           :param model: Transformer Model
           :param enc_input: The encoder input
           :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
           :return: The target input
           """
        enc_outputs, enc_self_attns = model.encoder(enc_input)
        dec_input = torch.zeros(1, 0).type_as(enc_input.data)
        terminal = False
        next_symbol = start_symbol

        while not terminal:
            # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
            dec_input = torch.cat(
                [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
                -1)
            dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
            projected = model.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            # 增量更新（我们希望重复单词预测结果是一样的）
            # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
            next_word = prob.data[-1]  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
            next_symbol = next_word
            if next_symbol == tgt_vocab["E"]:
                terminal = True
            # print(next_word)

            # greedy_dec_predict = torch.cat(
            #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
            #     -1)
        greedy_dec_predict = dec_input[:, 1:]
        return greedy_dec_predict

    # ==========================================================================================
    # 预测阶段
    enc_inputs, _, _ = next(iter(loader))
    for i in range(len(enc_inputs)):
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
        print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
        print([src_idx2word[t.item()] for t in enc_inputs[i]], '->',
              [idx2word[n.item()] for n in greedy_dec_predict.squeeze()])


if __name__ == '__main__':
    main()
