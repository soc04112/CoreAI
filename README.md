# CoreAI

## 1. 프로젝트 개요

- GPT-2 기반의 자체 제작한 모델을  
- 자체 토크나이저(CoreAI_Tokenizer)와 함께  
- 대규모 JSONL 텍스트 데이터셋을 sliding-window 방식으로 전처리하여  
- Gradient Accumulation, Adafactor 옵티마이저, Cosine LR 스케줄러를 적용해  
- 사전학습(pre-training)을 수행하며
- Validation Loss 및 Perplexity로 평가하여 법률에 특화된 LLM 기본 모델 구축.
- 해당 프로젝트로 LLM 기본 모델 구축에 대한 이해 및 경험을 쌓는 것을 목표로 진행하였습니다.

## 2. 주요 기능

- 데이터 처리  
  - JSONL 포맷 원본 텍스트를 `JsonlTextDataset` 으로 로드  
  - block_size/stride 기반 sliding-window 토큰화  
  - 캐시(.pt) 기능으로 재처리 시간 절약  
- 모델 구성  
  - `GPT2Config` 를 이용해 자유로운 하이퍼파라미터 설정  
  - optional Flash-Attn/Triton/XFormers 어텐션 가속 지원  
- 학습 파이프라인  
  - DataLoader + Gradient Accumulation (예: grad_accum_steps=64)  
  - Adafactor + CosineAnnealingLR 스케줄러  
  - AMP(autocast & GradScaler)로 FP16 메모리·속도 최적화  
  - 중간/최종 체크포인트 저장 & 복원(`save_checkpoint` / `find_latest_checkpoint`)  
  - 시그널 핸들러: Ctrl-C / SIGTERM 시에도 깨끗하게 체크포인트 저장  
- STS 미세튜닝 & 평가  
  - `STSDataset` 으로 STS-Train/STS-Dev TSV 로드  
  - 문장 임베딩 평균 + |差| 벡터 + MLP 회귀(head) 구조  
  - Spearman 상관계수 기반 Dev 평가  
- 로깅 & 모니터링  
  - Weights & Biases (wandb) 연동  
  - 학습/검증 loss, LR, 스텝별 메트릭 자동 기록  
  - 모델·토크나이저·아티팩트 버전 관리

## 3. 파일/디렉터리 구조

- CoreAI_BaseModel/
- ├── configs/
- │   └── configs.yaml              # 데이터 경로, 모델/학습 설정
- ├── data/
- │   ├── dataset                   # 원본 jsonl 파일 (법률 데이터 및 한국어 데이터)
- │   ├── sts                       # STS 데이터
- │   ├── processed                 # 전처리 데이터
- │   ├── model                     # 제작된 모델
- │   ├── tokenizer                 # 제작된 토크나이저
- │   └── wandb                     # wandb 임시 파일
- ├── src/
- │   ├── CoreAI_BaseModel/
- │   ├──   ├── datasets.py         # JsonlTextDataset, STSDataset
- │   ├──   ├── models.py           # STSModel 정의
- │   ├──   ├── trainer.py          # pretrain/finetune 루프
- │   ├──   ├── utils.py            # checkpoint, 시그널핸들러, 모델 통계
- │   └──   └── main.py             # 전체 파이프라인 엔트리포인트
- │   ├── mk_tokenizer/
- │   └──   └── make_tokenizers.py  # 토크나이저 제작
- ├── notebooks/
- │   ├── DIYModel.ipynb            # 모델 생성용 파일
- │   └── pretrainined.ipynb        # Colab 테스트용 파일
- ├── checkpoints/                  # 자동 생성되는 체크포인트 저장 폴더
- ├── requirements.txt              # 의존성 리스트
- └── README.md
