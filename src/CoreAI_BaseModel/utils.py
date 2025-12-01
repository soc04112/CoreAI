import glob
import os
import signal

import torch


def find_latest_checkpoint(ckpt_dir: str):
    """
    ckpt_dir/checkpoint-<step>.pt 파일들 중
    가장 step 이 큰 파일 경로, 그리고 그 스텝 번호를 리턴합니다.
    없으면 (None, None).
    """
    paths = glob.glob(os.path.join(ckpt_dir, "checkpoint-*.pt"))
    if not paths:
        return None, None
    # 파일명에서 숫자만 추출 (checkpoint-1234.pt → 1234)
    def extract_step(path):
        base = os.path.basename(path)
        num = base.replace("checkpoint-", "").replace(".pt", "")
        return int(num)
    steps = [extract_step(p) for p in paths]
    idx = int(steps.index(max(steps)))
    return paths[idx], steps[idx]

def save_checkpoint(output_dir, model, optimizer, scheduler, step):
    os.makedirs(output_dir, exist_ok=True)
    tmp_path   = os.path.join(output_dir, f"checkpoint-{step}.pt.tmp")
    final_path = os.path.join(output_dir, f"checkpoint-{step}.pt")

    # 1) temp 파일에 저장
    with open(tmp_path, "wb") as f:
        torch.save({
            "step":      step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, f)
        f.flush()
        os.fsync(f.fileno())   # 디스크에 완전 기록 보장

    # 2) atomic rename
    os.replace(tmp_path, final_path)
    print(f"*** Saved checkpoint: {final_path}")

def register_signal_handlers(output_dir, model, optimizer, scheduler, get_step_fn):
    def handler(signum, frame):
        step = get_step_fn()
        print(f"\n=== Received signal {signum}, saving final checkpoint ...")
        save_checkpoint(output_dir, model, optimizer, scheduler, step)
        exit(0)

    signal.signal(signal.SIGINT, handler)   # Ctrl-C
    signal.signal(signal.SIGTERM, handler)  # kill

def print_model_info(model):
   # ============================================================================
    # 1. 기본 모델 정보
    # ============================================================================
    print("\n [모델 기본 정보]")
    print(f"모델 타입: {model.config.model_type}")
    print(f"모델 아키텍처: {model.__class__.__name__}")

    # ============================================================================
    # 2. 모델 설정(config) 확인
    # ============================================================================
    print("\n [모델 설정 정보]")
    print(f"히든 레이어 크기 (hidden_size): {model.config.hidden_size}")
    print(f"디코더 레이어 수 (num_hidden_layers): {model.config.num_hidden_layers}")
    print(f"어텐션 헤드 수 (num_attention_heads): {model.config.num_attention_heads}")

    # Phi-3는 intermediate_size가 없을 수 있으므로 안전하게 처리
    if hasattr(model.config, 'intermediate_size'):
        print(f"중간 레이어 크기 (intermediate_size): {model.config.intermediate_size}")
    else:
        # FFN 크기는 보통 hidden_size의 4배
        estimated_intermediate = model.config.hidden_size * 4
        print(f"중간 레이어 크기 (추정): {estimated_intermediate}")

    print(f"어휘 크기 (vocab_size): {model.config.vocab_size}")
    print(f"최대 시퀀스(문맥) 길이 (max_position_embeddings): {model.config.max_position_embeddings}")


    # 추가 정보 (Phi-3 특화)
    if hasattr(model.config, 'num_key_value_heads'):
        print(f"Key-Value 헤드 수 (GQA): {model.config.num_key_value_heads}")
    if hasattr(model.config, 'rope_theta'):
        print(f"RoPE Theta: {model.config.rope_theta}")
    if hasattr(model.config, 'sliding_window'):
        print(f"Sliding Window: {model.config.sliding_window}")

    # ============================================================================
    # 3. 전체 파라미터 통계
    # ============================================================================
    print("\n [파라미터 통계]")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"전체 파라미터: {total_params:,}개 ({total_params / 1e9:.2f}B)")
    print(f"학습 가능 파라미터: {trainable_params:,}개 ({trainable_params / 1e9:.2f}B)")
    print(f"고정 파라미터: {frozen_params:,}개")

    # 메모리 사용량 추정
    mem_fp32 = (total_params * 4) / (1024 ** 2)
    mem_fp16 = mem_fp32 / 2
    print(f"예상 메모리:")
    print(f"  - Float32: {mem_fp32:.2f} MB ({mem_fp32/1024:.2f} GB)")
    print(f"  - Float16: {mem_fp16:.2f} MB ({mem_fp16/1024:.2f} GB)")

    # ============================================================================
    # 4. 모델 구조 계층 분석
    # ============================================================================
    print("\n [모델 구조 계층]")

    # 모델의 주요 컴포넌트 확인
    print("주요 컴포넌트:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  - {name}: {module_params:,}개 파라미터")


    # ============================================================================
    # 5. 카테고리별 파라미터 분석
    # ============================================================================
    print("\n [카테고리별 파라미터 분석]")

    categories = {
        'Embeddings  (임베딩)': 0,
        'Attention   (어텐션)': 0,
        'MLP/FFN (피드포워드)': 0,
        'LayerNorm   (정규화)': 0,
        'Output Head (출력층)': 0,
        'Others        (기타)': 0
    }

    for name, param in model.named_parameters():
        param_count = param.numel()

        # 카테고리 분류
        if 'embed' in name.lower():
            categories['Embeddings  (임베딩)'] += param_count
        elif 'attn' in name.lower() or 'attention' in name.lower():
            categories['Attention   (어텐션)'] += param_count
        elif 'mlp' in name.lower() or 'ffn' in name.lower() or 'fc' in name.lower():
            categories['MLP/FFN (피드포워드)'] += param_count
        elif 'norm' in name.lower() or 'ln' in name.lower():
            categories['LayerNorm   (정규화)'] += param_count
        elif 'lm_head' in name.lower() or 'output' in name.lower():
            categories['Output Head (출력층)'] += param_count
        else:
            categories['Others        (기타)'] += param_count

    print(f"{'카테고리':<30} {'파라미터 수':<20} {'비율'}")
    print("-" * 65)

    for category, count in categories.items():
        if count > 0:
            percentage = (count / total_params) * 100
            print(f"{category:<20} {count:>20,}개 {percentage:>6.2f}%")