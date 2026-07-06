位置编码，既旋转位置编码

## 为什么要用旋转位置编码
	语言是由语序的不同的单词排布顺序语言所表达的意思是完全不一样的比如“我吃苹果”是日常中正常的动作，但是“苹果吃我”就是一个细思极恐的故事了，在我们训练模型的过程中，语句被切成了一个一个的token，而当token经历了词嵌入（embedding）成为词向量后，token的顺序完全被打乱了，毫无顺序特征。
	为了解决这个语言顺序信息丢失的问题，我们就在词嵌入（embedding）后我们对语言向量经过一个预处理——旋转位置编码，讲语言的顺序信息融入到词向量中，这样子模型就能理解语言的顺序
	学习的是语言的相对位置，因为正余弦函数的线性性质

## 怎么引入语言顺序信息
	为了引入语言顺序信息，用了这么一种方式来引入，他将圆根据语句token的长度，分成了无数个小块，而用无数个小块的数学形式（sin，cos）来加入到词向量当中，这样子模型就能通过数字信息来获得语序信息

## 数学公式
	$PE(pos,2i) = sin\left(\frac{pos}{10000^\frac{2i}{d_{model}}}\right)$
	$PE(pos,2i+1) = cos\left(\frac{pos}{10000^\frac{2i}{d_{model}}}\right)$

## 下面我将精读代码
```python
class PositionalEncoding(nn.Module):
    """
    正弦位置编码 — Vaswani et al., Sec 3.5
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    正弦函数的线性性质使模型可学习相对位置关系。
    陌生的操作：
	    self.register_buffer
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 预计算位置编码，注册为 buffer（不参与梯度更新）
        pe = torch.zeros(max_len, d_model)
        # 创建一个从0到max_len-1长度的张量，然后将每一个数分别分配一个维度，张量变为[[0],[1],······]
        position = torch.arange(0, max_len, dtype=torch.float.unsqueeze(1)
        # 计算公式中的位置编码得分分母位置，为了方便计算机写代码，我们先将除数进行一个log然后再exp
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        # 计算sin和cos数值然后分别依次装入
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将张量的第一的维度之前再加上一个维度，让位置编码的维度数与batch词向量数相符
        pe = pe.unsqueeze(0)
        # 将位置编码的数值变为不训练的权重参数固定到模型内
        self.register_buffer('pe', pe)
    def forward(self, x):
	    # 如果输入的句子token数大于默认的值则重新在计算一次
        if x.size(1) > self.pe.size(1):
            pe = torch.zeros(x.size(1), self.pe.size(-1), device=x.device)
            position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(-1), 2, device=x.device).float() * (-math.log(10000.0) / self.pe.size(-1)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe[:, :x.size(1), :]
        else:
	        # 将位置编码数加到词向量中
            x = x + self.pe[:, :x.size(1), :]
            # 再最后经过一个dropout，提高模型训练的稳定性
        return self.dropout(x)
```
