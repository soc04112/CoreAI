from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

train_data_filename = ""

# 토크나이저 초기화
tokenizer = Tokenizer(models.Unigram(unk_id=0))

# 전처리기 설정 (Metaspace)
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=" ", add_prefix_space=True)

# 훈련 설정 (unigram)
special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]", "[BOS_Q]", "[EOS_Q]", "[BOS_A]", "[BOS_E]"]
trainer = trainers.UnigramTrainer(
    vocab_size = 32000,
    special_tokens = special_tokens,
    character_coverage = 1.0 # 텍스트에서 얼마나 많은 문자를 커버할지(1.0은 모든 문자)
)

# 훈련 실행
tokenizer.train(files=[train_data_filename], trainer=trainer)

# 디코더 설정
tokenizer.decoder = decoders.BPEDecoder(suffix=" ")

# tokenizers 라이브러리 형식으로 저장
tokenizer_json_path = "tokenizer.json"
tokenizer.save(tokenizer_json_path)

# 훈련한 토크나이저 로드
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)

# 특수 토큰 설정
hf_tokenizer.unk_token = "[UNK]"
hf_tokenizer.pad_token = "[PAD]"
hf_tokenizer.bos_token = "[BOS]"
hf_tokenizer.eos_token = "[EOS]"
hf_tokenizer.mask_token = "[MASK]"

# hugging face 형식으로 저장 
# tokenizer_config.json, special_tokens_map.json 파일 생성
hf_tokenizer_dir = "./coreAI_tokenizer"
hf_tokenizer.save_pretrained(hf_tokenizer_dir)