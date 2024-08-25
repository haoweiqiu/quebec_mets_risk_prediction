import pandas as pd
import torch as torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CategoricalEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim, unique_dim):
        super(CategoricalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim - unique_dim)
        self.unique = nn.Parameter(torch.randn(unique_dim))

    def forward(self, x):
        x = self.embedding(x)
        unique = self.unique.repeat(x.shape[0], 1)
        return torch.cat([x, unique], dim=1)

class MHA(nn.Module):
    def __init__(self, embedding_dimensions, attention_count):
        super(MHA, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.attention_count = attention_count
        self.attention_dimensions = embedding_dimensions // attention_count
        assert (self.attention_count * self.attention_dimensions == self.embedding_dimensions), \
            'attentions count unmatched'
        self.queries_linear = nn.Linear(self.embedding_dimensions, self.embedding_dimensions, bias=False)
        self.keys_linear = nn.Linear(self.embedding_dimensions, self.embedding_dimensions, bias=False)
        self.values_linear = nn.Linear(self.embedding_dimensions, self.embedding_dimensions, bias=False)
        self.fully_connected_output = nn.Linear(self.attention_dimensions * self.attention_count, self.embedding_dimensions)

    def forward(self, x):
        batch_size = x.shape[0]
        sentence_len = x.shape[1]
        Q = self.queries_linear(x).reshape(batch_size, sentence_len, self.attention_count, self.attention_dimensions).permute(0, 2, 1, 3)
        K = self.keys_linear(x).reshape(batch_size, sentence_len, self.attention_count, self.attention_dimensions).permute(0, 2, 3, 1)
        V = self.values_linear(x).reshape(batch_size, sentence_len, self.attention_count, self.attention_dimensions).permute(0, 2, 1, 3)
        score_att = torch.einsum('bijk,bikl->bijl', Q, K)
        distribution_att = torch.softmax(score_att / (self.embedding_dimensions ** (1 / 2)), dim=-1)
        attention_out = torch.einsum('bijk,bikl->bijl', distribution_att, V)
        return attention_out.permute(0, 2, 1, 3).reshape(batch_size, sentence_len, self.embedding_dimensions)

class Encoder(nn.Module):
    def __init__(self, embedding_dimensions, attention_count, expan_factor, dp_rate=0.1):
        super(Encoder, self).__init__()
        self.attention = MHA(embedding_dimensions, attention_count)
        self.attention_norm_layer = nn.LayerNorm(embedding_dimensions)
        self.ffn_norm_layer = nn.LayerNorm(embedding_dimensions)
        self.ff_layer = nn.Sequential(
            nn.Linear(embedding_dimensions, expan_factor * embedding_dimensions),
            nn.ReLU(),
            nn.Linear(expan_factor * embedding_dimensions, embedding_dimensions)
        )
        self.dp_rate = nn.Dropout(dp_rate)

    def forward(self, input):
        attention_out = self.dp_rate(self.attention(input))
        input = self.attention_norm_layer(input + attention_out)
        forward_out = self.dp_rate(self.ff_layer(input))
        encoder_output = self.ffn_norm_layer(input + forward_out)
        return encoder_output

class EncoderModel(nn.Module):
    def __init__(self, embeddings, max_length, embedding_dimensions, attention_count, expan_factor, num_layers=2, dp_rate=0.1):
        super(EncoderModel, self).__init__()
        self.embeddings = nn.ModuleList(embeddings.values())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimensions))
        self.encoder_layers = nn.ModuleList([Encoder(embedding_dimensions, attention_count, expan_factor, dp_rate) for _ in range(num_layers)])
        embedding_std = next(iter(embeddings.values())).embedding.weight.std().item()
        self.group_bias_1 = nn.Parameter(torch.randn(1) * embedding_std)
        self.group_bias_2 = nn.Parameter(torch.randn(1) * embedding_std)
        while torch.equal(self.group_bias_1, self.group_bias_2):
            self.group_bias_2 = nn.Parameter(torch.randn(1) * embedding_std)
        for col, emb in embeddings.items():
            print(f"Embeddings for column {col}:")
            print(emb.embedding.weight)

    def forward(self, x):
        x_emb = torch.stack([emb(x[:, i].long()) for i, emb in enumerate(self.embeddings)], dim=1)
        batch_size = x_emb.shape[0]
        group_bias_1_expanded = self.group_bias_1.expand(batch_size, 4, x_emb.size(2))
        group_bias_2_expanded = self.group_bias_2.expand(batch_size, 12, x_emb.size(2))
        x_emb[:, :4, :] += group_bias_1_expanded
        x_emb[:, 4:, :] += group_bias_2_expanded
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        for encoder in self.encoder_layers:
            x_emb = encoder(x_emb)
        return x_emb[:, 0]

class TabularDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return torch.tensor(self.dataframe.iloc[idx].values)

class Classifier(nn.Module):
    def __init__(self, embeddings, max_length, embedding_dimensions, attention_count, expan_factor, num_layers, dp_rate, output_dim):
        super(Classifier, self).__init__()
        self.encoder_model = EncoderModel(embeddings, max_length, embedding_dimensions, attention_count, expan_factor, num_layers, dp_rate)
        self.fc = nn.Linear(embedding_dimensions, output_dim) 

    def forward(self, x):
        cls_encoding = self.encoder_model(x)
        out = self.fc(cls_encoding)
        return torch.sigmoid(out)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
