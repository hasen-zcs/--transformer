## 介绍
残差归一化层，本质上是就是一个残差网络后面跟一个归一化层，这里我也不过多赘述，大家直接看代码吧

## 代码
```python
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
```
