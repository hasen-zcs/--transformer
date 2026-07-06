## 介绍
前馈神经网路$FFN$($Feed Forward Network$)，本质上就是一个两层的线性网络层，中间用$ReLU()$激活函数，相信大家已经有了深厚的深度学习基础，所以在这里我就不过多的赘述

## 数学公式
	$FFN(x) = \max(0, x W_1^\top+b_2)*W_2^\top+b_2$



## 代码
```python
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
```