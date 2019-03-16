 ### 用LSTM进行词性标注 ###
 在这部分, 我们将会使用一个 LSTM 网络来进行词性标注.<br>
 整个模型的参数定义如下: 输入的句子定义为 \(w_1, \dots, w_M\), 其中动词定义 为 \(w_1, \dots, w_M\), 标签集合定义为 \(T\), 单词 \(w_i\) 的实际 标签为 \(y_i\). 定义单词 \(w_i\) 的预测标签为 \(\hat{y}_i\).<br>
 这是一个结构预测模型, 我们的输出是一个序列 \(\hat{y}_1, \dots, \hat{y}_M\), 其中 \(\hat{y}_i \in T\).<br>
 在进行预测时, 需将句子每个词输入到一个 LSTM 网络中. 将时刻 \(i\) 的隐状态标记 为 \(h_i\). 同样地, 对每个标签赋一个独一无二的索引 (类似 word embeddings 部分 word_to_ix 的设置). 然后就得到了 \(\hat{y}_i\) 的预测规则:
<pre><code>
\[\hat{y}_i = \text{argmax}_j \ (\log \text{Softmax}(Ah_i + b))j]
</code></pre>
先对隐状态进行一个仿射变换，然后计算一个对数softmax，最后得到的预测标签，即为对数softmax中最大的值对应的标签。注意, 这也意味着 \(A\) 空间的维度是 \(|T|\).<br>

**准备数据：**
<pre><code>
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

#实际中通常使用更大的维度如32维, 64维.
#这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
</code></pre>

**构造模型：**
<pre><code>
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 线性层将隐状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
</code></pre>

**训练模型：**
<pre><code>
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 查看下训练前得分的值
# 注意: 输出的 i,j 元素的值表示单词 i 的 j 标签的得分
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # 再次说明下, 实际情况下你不会训练300个周期, 此例中我们只是构造了一些假数据
    for sentence, tags in training_data:
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        model.zero_grad()

        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        model.hidden = model.init_hidden()

        # Step 2\. 准备网络输入, 将其变为词索引的 Variables 类型数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3\. 前向传播
        tag_scores = model(sentence_in)

        # Step 4\. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# 查看训练后得分的值
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# 句子是 "the dog ate the apple", i,j 表示对于单词 i, 标签 j 的得分.
# 我们采用得分最高的标签作为预测的标签. 从下面的输出我们可以看到, 预测得
# 到的结果是0 1 2 0 1\. 因为 索引是从0开始的, 因此第一个值0表示第一行的
# 最大值, 第二个值1表示第二行的最大值, 以此类推. 所以最后的结果是 DET
# NOUN VERB DET NOUN, 整个序列都是正确的!
print(tag_scores)
</code></pre>
