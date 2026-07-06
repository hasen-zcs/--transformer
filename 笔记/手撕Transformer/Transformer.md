![[transformer架构.png]]
## 正文
终于咱们已经搭建完了所有的transformer内部组件，现在咱们来将transformer的内部组件拼凑起来，完全完成完整transformer模型


## 初始化模型参数
```python
class Transformer(nn.Module):
    """
	    完整 Transformer 模型 — Vaswani et al., Sec 3.1
	    三个入口：forward() 训练, encode() 编码, decode() 解码（支持逐步生成）
    """
    def __init__(self, 
				src_vocab_size, 
				tgt_vocab_size, 
				d_model=512, 
				num_heads=8,
				num_encoder_layers=6, 
				num_decoder_layers=6, 
				d_ffn=2048, 
				dropout=0.1,
				max_len=5000, 
				pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_decoder_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()
	# 1.初始化权重
    def _init_weights(self):
        """Xavier uniform 初始化 — Vaswani et al., Sec 3.2.2"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -- 掩码生成
    
    def generate_square_subsequent_mask(self, sz):
        """
        因果掩码 — 下三角矩阵，True 表示可见。
        """
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask

    def make_src_mask(self, src):
        """padding 掩码，返回 [batch, 1, 1, src_len] 支持广播。"""
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """目标掩码 = 因果掩码 & padding 掩码。"""
        tgt_len = tgt.size(1)
        subsequent_mask = /
	        /self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask & subsequent_mask

    # -- 前向传播
    def encode(self, src):
        src_padding_mask = self.make_src_mask(src)
        src_mask = src_padding_mask & src_padding_mask.transpose(-2, -1)
        src_embed = self.src_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)
        enc_output = src_embed
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        return enc_output, src_padding_mask

    def decode(self, tgt, encoder_output, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        tgt_embed = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)
        dec_output = tgt_embed
        for layer in self.decoder:
            dec_output = layer(dec_output, encoder_output, tgt_mask, src_mask)
        return dec_output

    def forward(self, src, tgt):
        encoder_output, src_mask = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output, src_mask)
        output = self.linear(decoder_output)
        return output
```

## 类内方法解析
#### 初始化参数
```python
def _init_weights(self):
        """Xavier uniform 初始化 — Vaswani et al., Sec 3.2.2"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
```
初始化所有模型内的参数，通过for循环遍历所有的参数，判断$p.dim()>1$的原因->因为一般dim=1的参数张量一般是偏执项参数，而偏执参数一般初始化为零，所以不需要参数初始化

### 掩码部分
	掩码是transformer架构中十分重要的一个机制，它决定了模型能不能真正的理解语言的意思，掩码可以理解为一张挡板，掩码的大小对应于注意力分数score（也就是注意力中Q和K进行点积后得到的得分），用来遮挡score，在掩码是一个bool张量，掩码中为True的元素相对应score位置上的元素是模型能看到的，而掩码中为False的元素相对应的score位置上的元素模型是看不到的，这个样子我们就可以有选择性地让模型看到我们让他看到的内容。
#### 因果掩码
```python
def generate_square_subsequent_mask(self, sz):
        """
        因果掩码 — 下三角矩阵，True 表示可见。
        陌生操作：
		· torch.tril(input, diagonal):
			input：要改变的矩阵
			diagonal=0：保留主对角线及以下的元素信息，上三角全变为零。
			diagonal=k>0；保留主对角线上k条对角线及以下的元素信息，上三角全变为零。
			diagonal=k<0：保留主对角线下k条对角线及以下的元素信息，上三角全变为零。
        """
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask
```
是一个下三角矩阵，其中下面的是True可见，上三角是False不可见，所以这样子就可以这样防止模型在生成的时候不会看到未来的信息，这样子自回归的时候就只会看到现在的和过去的信息。这个因果掩码使用在Decoder中的第一个MulHeadsAttention里面的，用来遮蔽目标语句
可视化：
```
# sz = 5 时，生成的下三角矩阵
[[True, False, False, False, False],   # 第0个位置只能看到自己
 [True, True,  False, False, False],   # 第1个位置能看到0和1
 [True, True,  True,  False, False],   # 第2个位置能看到0,1,2
 [True, True,  True,  True,  False],   # 第3个位置能看到0,1,2,3
 [True, True,  True,  True,  True ]]   # 第4个位置能看到全部
```

### Padding掩码
```python
def make_src_mask(self, src):
	mask = (src != self.pad_idx)
```