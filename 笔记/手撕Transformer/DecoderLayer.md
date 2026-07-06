## 介绍

![[DecoderLayer.png]]
解码器层就是


## 代码
```python
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
```