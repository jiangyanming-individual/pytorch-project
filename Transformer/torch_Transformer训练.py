import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils import clip_grad_norm_
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, hid_dim)
        self.src_pos_encoder = nn.Embedding(100, hid_dim)  # 设置序列的最大长度为100
        self.transformer = nn.Transformer(hid_dim, n_heads, n_layers, pf_dim, dropout)
        self.fc_output = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_pos):
        # src: [seq_len, batch_size]
        # src_pos: [seq_len, batch_size]
        src_embedded = self.dropout(self.encoder(src) + self.src_pos_encoder(src_pos))
        # src_embedded: [seq_len, batch_size, hid_dim]
        src_padding_mask = src.eq(0)
        # src_padding_mask: [seq_len, batch_size]

        src_features = self.transformer(src_embedded, src_key_padding_mask=src_padding_mask)
        # src_features: [seq_len, batch_size, hid_dim]

        output = self.fc_output(src_features)
        # output: [seq_len, batch_size, output_dim]

        return output


# 数据预处理和加载
src = Field(tokenize="spacy", tokenizer_language="de", lower=True)
trg = Field(tokenize="spacy", tokenizer_language="en", lower=True)
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(src, trg))

src.build_vocab(train_data, min_freq=2)
trg.build_vocab(train_data, min_freq=2)

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=64,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 设置超参数
input_dim = len(src.vocab)
output_dim = len(trg.vocab)
hid_dim = 256
n_layers = 6
n_heads = 8
pf_dim = 512
dropout = 0.1
learning_rate = 0.0005
num_epochs = 10

# 创建模型和优化器
model = Transformer(input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=trg.vocab.stoi[trg.pad_token])

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_iter):
        src_data = batch.src
        src_pos = torch.arange(0, src_data.shape[0]).unsqueeze(1).repeat(1, src_data.shape[1])
        trg_data = batch.trg

        optimizer.zero_grad()

        output = model(src_data, src_pos)
        output = output[1:].view(-1, output.shape[-1])
        trg_data = trg_data[1:].view(-1)

        loss = criterion(output, trg_data)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(train_iter.dataset) // train_iter.batch_size)

    print("Epoch:", epoch + 1, "Loss:", avg_loss)


# 保存训练好的模型
torch.save(model.state_dict(), "transformer_model.pt")