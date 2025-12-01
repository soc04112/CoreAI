import tqdm
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.cuda.amp import autocast
from torch.optim import Adafactor

from models import STSModel


# STS 미세튜닝 함수
def finetune_and_evaluate_sts(
    main_model,
    sts_train_dataloader,
    sts_dev_dataloader,
    device,
    finetune_epochs: int,
    lr: float,
):
    print("STS 파인튜닝 시작")

    base_model = main_model.transformer if hasattr(main_model, 'transformer') else main_model
    sts_model = STSModel(base_model).to(device)

    optimizer = Adafactor(sts_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_spearman = -1.0

    for epoch in range(finetune_epochs, desc="STS Fine-tuning", leave=True):
        sts_model.train()
        total_train_loss = 0
        for batch in tqdm(sts_train_dataloader, desc=f"STS Train Epoch{epoch+1}", leave=True):
            optimizer.zero_grad()

            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            scores = batch['score'].to(device)

            with autocast():
                predicted_scores = sts_model(input_ids1, attention_mask1, input_ids2, attention_mask2).squeeze(-1)
                loss = loss_fn(predicted_scores, scores)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(sts_train_dataloader)
        print(f"STS 파인튜닝 Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

        # 매 에포크 후 dev 셋으로 검증
        sts_model.eval()
        real_scores, model_scores = [], []

        with torch.no_grad():
            for batch in sts_dev_dataloader:
                input_ids1 = batch['input_ids1'].to(device)
                attention_mask1 = batch['attention_mask1'].to(device)
                input_ids2 = batch['input_ids2'].to(device)
                attention_mask2 = batch['attention_mask2'].to(device)

                with autocast():
                    predicted_scores = sts_model(input_ids1, attention_mask1, input_ids2, attention_mask2).squeeze(-1)

                model_scores.extend(predicted_scores.cpu().numpy())
                real_scores.extend(batch['score'].cpu().numpy())

        spearman_corr, _ = spearmanr(real_scores, model_scores)
        print(f"STS 파인튜닝 Epoch {epoch+1} - Spearman Correlation: {spearman_corr:.4f}")

        if spearman_corr > best_spearman:
            best_spearman = spearman_corr

    print(f"STS 파인튜닝 및 검증 종료. 최고 점수: {best_spearman:.4f}")

    return best_spearman
