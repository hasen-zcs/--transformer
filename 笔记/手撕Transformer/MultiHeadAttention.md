![[Multi_HeadAttention.png]]

Scaled Dot-Product Attention 点积注意力机制

公式
![[注意力机制公式.png|697]]
Q：查询向量，表示**我要来找什么信息**
K：键向量，表示**我有什么信息可以被检索**
V：值向量，表示**该位置储存的信息**
$\sqrt{d_k}$：缩放因子，防止点积值过大倒是$softmax()$的值过于极端导致梯度消失

代码
```python
def ScaledDotProductAttention(query, key, values, mask=None):
	"""
		陌生操作
		· torch.matmul(mat1, mat2。transpose(-2, -1))->两张量相乘
		· x.transpose(dim1, dim2)->将两个维度上的数转置
		· x.masked_fill(条件, 值)->将条件中为True的位置用值替代，代码中的mask是用False代表要占用的位置,True代表不占用的位置，所以要用~mask来取反
		· torch.where(条件, 值1, 值2)->通过条件来取用两个值中相同位置上的值，成立时取用值1，不成立时取用值2
	"""
	# 保存Query的最后一维的维度信息
    d_k = query.size(-1)
    # 将query和key相乘得到注意力得分，再将得分除以缩放因子
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    attention_weights = F.softmax(score, dim=-1)
    # 将mask后Nan的地方改为0
    attention_weights = torch.where(torch.isnan(attention_weights),
                                    torch.zeros_like(attention_weights),
                                    attention_weights)
    return torch.matmul(attention_weights, values), attention_weights
```

## 多头自注意力机制
![[多头自注意力机制.png]]
## 公式

$$
\begin{align}
\mathrm{MultiHead}(Q, K, V) &= \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)W^O \\
\mathrm{where}\ \mathrm{head}_i &= \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{align}
$$
形式上是将多个注意力堆叠起来，让词向量经过多个头，这样子来提取更多的特征
。。。。明天再写今天不想写了^3^
OK的我又来了，多头说简单点就是因为一个注意力获取的文本信息十分有限，为了获取更多的信息，我们就用更多的注意力来获取更多的信息，要提醒的是，多头自注意力机制虽然看起来是多个注意力叠起来，但其实在实现的时候我们是将原来词向量的最后一维d_model拆分成num_heads个部分，如一开始词向量维度是(batch_size, sen_len, d_model)我们将最后一维进行拆分成head个头(batch_size, -1, num_heads, d_model//num_heads),然后我们再将1维度和2维度经过转置将num_heads维度放在batch_size后面(batch_size, num_heads, -1, d_model//num_heads)，这样子我们在逻辑上讲第一维看成头数，为什么将num_heads放在batch_size后面呢？因为我们进行attention操作的时候进行计算，我们是对最后两个维度进行计算和变换的，所以当我们在，num_heads维度我们不会经过计算，这样子做逻辑上就将每一个num_heads维度上的子向量隔绝开了，就达到了多个头的效果，OK的接下来wine吧直接看代码

代码：
```python
class MulHeadAttention(nn.Module):
	"""
		多头自注意力机制张量形状变化：(batch_size, length, d_model)->
		(batch_size, length, num_heads, d_model//num_heads)->
		(batch_size, num_heads, length, d_model//num_heads)->
		(batch_size, length, num_heads, d_model//num_heads)->
		(batch_size, length, d_model)
	"""
    def __init__(self, d_model , num_heads = 8, dropout = 0.1):

        super().__init__()
        self.d_model = d_model
        self.d_k = d_model //num_heads
        self.d_v = d_model //num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

		# 定义Q, K, V三个输入向量
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)
        self.att_out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, values, mask=None):
    # 保留张量形状
        batch_size, high, wid = query.shape
        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(-2, -1)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(-2, -1)
        values = self.values(values).view(batch_size, -1, self.num_heads, self.d_k).transpose(-2, -1)
        x, att_weights = single_attention(query, key, values, mask)
        x = x.transpose(-1, -2).contiguous().view(batch_size, -1, self.d_model)
        x = self.dropout(x)
        return self.att_out(x), att_weights
```
