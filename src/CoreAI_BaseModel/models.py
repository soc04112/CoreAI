import torch
import torch.nn as nn


class STSModel(nn.Module):
  def __init__(self, pretrained_model):
    super().__init__()
    self.model = pretrained_model
    self.config = self.model.config
    self.regression_head = nn.Sequential(
        nn.Linear(self.config.hidden_size * 3, self.config.hidden_size),
        nn.ReLU(),
        nn.Linear(self.config.hidden_size, 1)
    )

 # 토큰 임베딩 평균
  def _mean_pooling(self, model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# 문장1, 문장2 임베딩, 차이 벡터 합쳐 *MLP 회귀
  def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
    outputs1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1)
    outputs2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2)
    embedding1 = self._mean_pooling(outputs1, attention_mask1)
    embedding2 = self._mean_pooling(outputs2, attention_mask2)
    diff = torch.abs(embedding1 - embedding2)
    combined_embedding = torch.cat([embedding1, embedding2, diff], dim=1)
    return self.regression_head(combined_embedding)