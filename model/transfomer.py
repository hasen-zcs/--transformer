import math
import torch
from torch import nn
import torch.nn.functional as F

# 旋转位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dorpout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dorpout)

        # 预训练旋转位置编码参数，注册为buffer（不参与训练）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe形状()
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            pe = torch.zeros(x.size(1), self.pe.size(-1), device=x.device)
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(-1), 2, device=x.device).float() *
                                (-math.log(10000) / self.pe.size(-1)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # pe形状()
            pe = pe.unsqueeze(0)
            x = x + pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
def single_attention(query, key, values, mask=None):
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    attention_weights = F.softmax(score, dim=-1)

    # 将mask后Nan的地方改为0
    attention_weights = torch.where(torch.isnan(attention_weights), 
                                    torch.zeros_like(attention_weights),
                                    attention_weights)
    return torch.matmul(attention_weights, values), attention_weights

# 多头自注意力机制
class MulHeadAttention(nn.Module):
    def __init__(self, d_model , num_heads = 8, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model //num_heads
        self.d_v = d_model //num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)

        self.att_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, values, mask=None):
        batch_size, sen_num, d_m = query.shape

        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(-2, -1)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(-2, -1)
        values = self.values(values).view(batch_size, -1, self.num_heads, self.d_k).transpose(-2, -1)

        x, att_weights = single_attention(query, key, values, mask)

        x = x.transpose(-1, -2).contiguous().view(batch_size, -1, self.d_model)
        x = self.dropout(x)

        return self.att_out(x), att_weights
    
# 前馈神经网络 linear-》relu-》dropout-》linear
class FFN(nn.Module):
    def __init__(self, d_model, latent_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)
        
class AddNorm(nn.Module):
    """
    残差连接 + 层归一化 — Vaswani et al., Sec 5.4
    output = LayerNorm(x + Dropout(Sublayer(x)))
    Post-LN 架构（论文原版）。
    """
    def __init__(self, Normalized_shape, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(Normalized_shape)

    def forward(self, res, x):
        return self.ln(res + self.dropout(x))


class EncoderLayer(nn.Module):
    """
        MulHeadAttention -> AddNorm -> FFN -> AddNorm
    """
    def __init__(self, d_model, d_ffn, num_heads, dropout=0.1):
        super().__init__()
        self.atten = MulHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.addNorm1 = AddNorm(d_model, dropout)
        self.addNorm2 = AddNorm(d_model, dropout)
    
    def forward(self, x, mask = None):
        att_output, _ = self.atten(x, x, x, mask)
        x = self.addNorm1(x, att_output)
        ffn_output = self.ffn(x)
        return self.addNorm2(x, ffn_output)
    
class DecoderLayer(nn.Module):
    """
        MulheadMaskAttiontion -> AddNorm -> CrossAttention -> AddNorm -> FFN -> AddNorm
    """
    def __init__(self, d_model, d_ffn, num_heads, dropout=0.1):
        super().__init__()
        # 第一层，maskAttention层
        self.maskAttention = MulHeadAttention(d_model, num_heads, dropout)
        # 第二层，第一个AddNorm层
        self.addNorm1 = AddNorm(d_model, dropout)
        # 第三层，CrossAttention层，我们服用MulHeadAttention
        self.crossAttention = MulHeadAttention(d_model, num_heads, dropout)
        # 第四层，第二个AddNorm层
        self.addNorm2 = AddNorm(d_model, dropout)
        # 第五层，FFN层
        self.ffn = FFN(d_model, d_ffn, dropout)
        # 第六层，最后一层AddNorm层
        self.addNorm3 = AddNorm(d_model, dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # x进入先经过maskAttention->addNorm1
        mask_Attention_output, _ = self.maskAttention(x, x, x, tgt_mask)
        x = self.addNorm1(x, mask_Attention_output)

        # 在经过crossAttenton与enc_output来计算, 然后在经过AddNorm层
        cross_attention_output, _ = self.crossAttention(x, enc_output, enc_output, src_mask)
        x = self.addNorm2(x, cross_attention_output)

        # 最后经过FFN层和AddNorm层
        ffn_output = self.ffn(x)
        x = self.addNorm3(x, ffn_output)

        # 最后返回值
        return x



if __name__ == "__main__":
    x = torch.rand(10, 100, 512)

    # # # 测试MulHeadAttention 输入(batch_size, num, d_model)->(batch, num, d_modle)
    # att = MulHeadAttention(512)
    # y, att_weights= att(x, x, x)
    # print("y形状:", y.shape)
    # print("y内容:", y, "\nweights：", att_weights)
    
    # # 测试PositionalEncoding 输入(batch_size, num, d_model)
    # pe = PositionalEncoding(512)
    # out = pe(x)
    # print("out：", out, "\noutshape:", out.shape)

    # # 测试FFN 输入(batch, num, d_modle)->(batch, num, d_modle)
    # ffn = FFN(512, 2048)
    # out = ffn(x)
    # print("out：", out, "\noutshape:", out.shape)

    # # 测试AddNorm 输入(batch, num, d_modle)->(batch, num, d_modle)
    # an = AddNorm(512)
    # out = an(x, x)
    # print("out：", out, "\noutshape:", out.shape)

    # EncoderLayer层测试
    encoderlayer = EncoderLayer(512, 2048, 8)
    out = encoderlayer(x)
    print("out：", out, "\noutshape:", out.shape)

    exit()