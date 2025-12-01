import os

import torch
import wandb
import yaml
# from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adafactor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from load_datasets import STSDataset, JsonlTextDataset
from trainer import finetune_and_evaluate_sts
from utils import (find_latest_checkpoint, print_model_info,
                    register_signal_handlers, save_checkpoint)


def load_config(path="./configs/configs.yaml"):
    return yaml.safe_load(open(path, "r"))

def build_tokenizer(tokenizer_dir):
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

def build_model(model_dir, device):
 
    # Avoid requesting a hub-specific flash-attn kernel on Windows/local dev environments.
    # Use the default/torch attention implementation for compatibility.
    # Use the eager (PyTorch-native) attention implementation which is broadly compatible
    # and avoids requiring external hub kernels on local/Windows environments.
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    return model.to(device)

def main():

    cfg = load_config()
    # device = torch.device(cfg["training"]["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg["data"]["tokenizer_dir"])
    per_device_batch_size = int(cfg["training"]["batch_size_per_accum"] / cfg["training"]["grad_accum_steps"] )
    
    #os.chdir("/content/drive/MyDrive/Colab Notebooks")
    #cached_dataset_path = "cached_full_dataset.pt"

    cached_dataset_path = cfg["data"]["processed_dir"] + "/cached_full_dataset.pt"
    
    if os.path.exists(cached_dataset_path):
        print("전처리 데이터셋 파일이 존재하여 불러옵니다.")
        full_dataset = torch.load(cached_dataset_path, weights_only=False)
    else:
        print("전처리 데이터셋 파일이 존재하지 않아 새로 생성합니다.")
        full_dataset = JsonlTextDataset(cfg["data"]["text_dir"], tokenizer,
                                   cfg["model"]["block_size"], cfg["model"]["stride"])
        print("전처리 데이터셋 생성 완료")
        torch.save(full_dataset, cached_dataset_path)
        print("데이터셋 저장 완료")

    train_size = int(len(full_dataset) * 0.975)
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size], 
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=per_device_batch_size,
    )

    try:
        sts_dev_dataset = STSDataset(cfg["data"]["sts_dev"], tokenizer)
        sts_dev_dataloader = DataLoader(sts_dev_dataset, batch_size=per_device_batch_size, shuffle=False)
        sts_train_dataset = STSDataset(cfg["data"]["sts_train"], tokenizer)
        sts_train_dataloader = DataLoader(sts_train_dataset, batch_size=per_device_batch_size, shuffle=True)
        run_sts_eval = True
        print("STS 데이터셋 로드 완료")
    except FileNotFoundError:
        print("Warning: STS 데이터셋이 없습니다. STS 평가는 건너뜁니다.")
        run_sts_eval = False

    # 2) 모델 & optimizer & scheduler & scaler
    model = build_model(cfg["data"]["model_dir"], device)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = Adafactor(
        model.parameters(), 
        lr=float(cfg["training"]["lr"]), 
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["scheduler"]["T_max"], eta_min=cfg["training"]["scheduler"]["eta_min"]
    )

    # 체크포인트 복원 시도
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    latest_ckpt, last_step = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        print(f"체크포인트를 찾아 해당 부분부터 학습을 시작합니다. {latest_ckpt} (step={last_step}) …")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt["step"]
        if hasattr(scheduler, 'last_epoch'):
            scheduler.last_epoch = global_step - 1
        else:
            for _ in range(global_step):
                scheduler.step()
    else:
        print("체크포인트를 찾지 못하여 처음부터 학습을 시작합니다.")
        global_step = 0

    # W&B 초기화
    wandb.login(key="d5b25fa78b19fef961f3f6b203f821bd5d2c5b91")
    wandb.init(
        project="CoreAI_pretrained",           # W&B 웹에서 만들 프로젝트 이름
        name=f"run-{os.getpid()}",            # (옵션) 실험 이름
        config=cfg["training"],
        dir="./data"
    )
    print_model_info(model)
    wandb.watch(model, log="all", log_freq=100)

    # 시그널 핸들러
    register_signal_handlers(ckpt_dir, model, optimizer, scheduler, lambda: global_step)

    # 4) 학습 루프
    grad_accum_steps = int(cfg["training"]["grad_accum_steps"])
    validation_interval = int(cfg["training"]["validation_interval"])
    total_steps = int(cfg["training"]["total_steps"])
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))

    model.train()
    register_signal_handlers(ckpt_dir, model, optimizer, scheduler, get_step_fn=lambda: global_step)
    for epoch in range(1):
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", leave=True)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1

                if global_step % validation_interval == 0:
                    model.eval()
                    total_val_loss = 0.0
                    num_val_batches = 0
                    with torch.no_grad():
                        print("검증 중")
                        for val_batch in val_dl:
                            val_input_ids = val_batch["input_ids"].to(device)
                            val_labels    = val_batch["labels"].to(device)

                            val_outputs = model(input_ids=val_input_ids, labels=val_labels)
                            total_val_loss += val_outputs.loss.item()
                            num_val_batches += 1

                    avg_val_loss = total_val_loss / num_val_batches
                    perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
                    print(f"*** Validation Step {global_step} - Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")
                    wandb.log({
                        "eval/loss":      avg_val_loss,
                        "eval/perplexity": perplexity,
                    }, step=global_step)

                    if run_sts_eval:
                        spearman_correlation = finetune_and_evaluate_sts(
                            model, 
                            sts_train_dataloader, 
                            sts_dev_dataloader, 
                            device,
                            cfg["training"]["finetune_epochs"],
                            cfg["training"]["sts_lr"],
                        )
                        wandb.log({
                            "sts/spearman": spearman_correlation
                        }, step=global_step)

                    model.train()

                cur_loss = loss.item() * grad_accum_steps
                cur_lr   = scheduler.get_last_lr()[0]

                # tqdm
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{cur_loss:.4f}",
                    "lr":   f"{cur_lr:.2e}"
                })

                # W&B 로그
                wandb.log({
                    "train/loss": cur_loss,
                    "train/lr":   cur_lr,
                    "step":       global_step
                }, step=global_step)

                # 매 100 스텝마다 체크포인트 저장
                if global_step > 0 and global_step % 100 == 0:
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, global_step)

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    if global_step > 0:
        save_checkpoint(ckpt_dir, model, optimizer, scheduler, global_step)

    # 2-4) 모델&토크나이저 저장
    model_dir = "foundation_ckpt"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # 아티팩트 생성 및 로깅
    artifact = wandb.Artifact(
        name="gpt2-pretrained",   # 아티팩트 이름
        type="model",             # 유형(모델, 데이터셋 등)
        metadata={"step": global_step}
    )
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)

    # 실험 종료 알림
    wandb.finish()

if __name__ == "__main__":
    main()