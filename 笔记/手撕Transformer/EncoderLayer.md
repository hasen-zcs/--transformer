## 编码器层
![[EncoderLayer.png]]

Encoder层由一个[[MultiHeadAttention]]层，一个[[PositionWiseFFN]]和两个[[AddNorm]]组成，我们只需要简单的堆叠就可以获得一层的EncoderLayer，然后再将这一层给叠加起来多层就是完整的Encoder了


## 代码
```python
class EncoderLayer(nn.Module):
	"""
		atten -> addNorm1 -> ffn -> addNorm
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
```
