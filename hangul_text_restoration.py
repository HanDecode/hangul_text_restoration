import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_scheduler,
)
from accelerate import Accelerator
from tqdm import tqdm
from sklearn.model_selection import train_test_split

###################################
# 1) 데이터 불러오기
###################################
train_path = "./train.csv"
test_path = "./test.csv"
train = pd.read_csv(train_path, encoding='utf-8-sig')
test = pd.read_csv(test_path, encoding='utf-8-sig')

###################################
# 2) 한글 증강/전처리 함수들
###################################
def decompose_hangul(char):
    base_code = ord(char) - 0xAC00
    initial_idx = base_code // 588
    medial_idx = (base_code % 588) // 28
    final_idx = base_code % 28
    initial = chr(initial_idx + 0x1100)  # 초성
    medial = chr(medial_idx + 0x1161)    # 중성
    final = chr(final_idx + 0x11A7) if final_idx > 0 else ''
    return initial, medial, final

initial_transform = {
    chr(0x1100): [chr(0x1101), chr(0x110F), chr(0x1100)],
    # ...
}
medial_transform = {
    chr(0x1161): [chr(0x1163), chr(0x116A), chr(0x1161)],
    # ...
}
final_consonants = [chr(0x11A8 + i) for i in range(28)]  # 종성 코드 (0x11A8 ~ 0x11C2)

def transform_initial(initial):
    if initial in initial_transform:
        return random.choice(initial_transform[initial])
    return initial

def transform_medial(medial):
    if medial in medial_transform:
        return random.choice(medial_transform[medial])
    return medial

def transform_final(final):
    if final:
        return random.choice(final_consonants)
    return ''

def compose_hangul(decomposed):
    HANGUL_BASE = 0xAC00
    INITIAL_BASE = 0x1100
    MEDIAL_BASE = 0x1161
    FINAL_BASE = 0x11A7
    initial, medial, final = decomposed
    initial_idx = ord(initial) - INITIAL_BASE
    medial_idx = ord(medial) - MEDIAL_BASE
    final_idx = ord(final) - FINAL_BASE if final else 0
    return chr(HANGUL_BASE + (initial_idx * 588) + (medial_idx * 28) + final_idx)

def transform_text(text):
    transformed_text = []
    for char in text:
        if '가' <= char <= '힣':
            i, m, f = decompose_hangul(char)
            new_i = transform_initial(i)
            new_m = transform_medial(m)
            new_f = transform_final(f)
            transformed_text.append(compose_hangul([new_i, new_m, new_f]))
        else:
            transformed_text.append(char)
    return ''.join(transformed_text)

# 연결음 처리 예시 함수
def decompose_hangul_2(char):
    code = ord(char) - 0xAC00
    if code < 0 or code > 11171:
        return 0, 0, 0
    final = code % 28
    medial = (code // 28) % 21
    initial = code // 28 // 21
    return initial, medial, final

def compose_hangul_2(initial, medial, final):
    return chr(0xAC00 + (initial * 21 + medial) * 28 + final)

FINAL_TO_INITIAL = {
    1: 0, 2: 1, 4: 2, 7: 3, 8: 5, 16: 6, 17: 7,
    19: 9, 20: 10, 21: 11, 22: 12, 23: 14,
    24: 15, 25: 16, 26: 17, 27: 18
}
COMPLEX_FINALS = {
    3: (1, 9), 5: (4, 12), 6: (4, 18), 9: (8, 0), 10: (8, 6),
    11: (8, 7), 12: (8, 9), 13: (8, 16), 14: (8, 17),
    15: (8, 18), 18: (17, 9)
}

def apply_liaison(text):
    if not text:
        return ""
    words = text.split()
    result = []
    for word in words:
        chars = list(word)
        i = 0
        while i < len(chars) - 1:
            curr_char = chars[i]
            next_char = chars[i + 1]
            try:
                curr_i, curr_m, curr_f = decompose_hangul_2(curr_char)
                next_i, next_m, next_f = decompose_hangul_2(next_char)
                # 종성 있고, 다음 글자 초성=='ㅇ' 일 때
                if curr_f > 0 and next_i == 11:
                    if curr_f in COMPLEX_FINALS:
                        f1, f2 = COMPLEX_FINALS[curr_f]
                        chars[i] = compose_hangul_2(curr_i, curr_m, f1)
                        new_init = FINAL_TO_INITIAL.get(f2, f2)
                        chars[i + 1] = compose_hangul_2(new_init, next_m, next_f)
                    else:
                        chars[i] = compose_hangul_2(curr_i, curr_m, 0)
                        new_init = FINAL_TO_INITIAL.get(curr_f, curr_f)
                        chars[i + 1] = compose_hangul_2(new_init, next_m, next_f)
            except:
                pass
            i += 1
        result.append(''.join(chars))
    return ' '.join(result)

# 전처리 컬럼 생성
train["raw_liaison"] = train["output"].apply(lambda x: apply_liaison(x))
train["liaison_trans"] = train["raw_liaison"].apply(lambda x: transform_text(x))

train["prompt_input"] = "난독화된 문장을 복원하세요: " + train["input"]
test["prompt_input"] = "난독화된 문장을 복원하세요: " + test["input"]

###################################
# 3) 데이터셋 분리
###################################
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)

###################################
# 4) 토크나이저/모델 불러오기
###################################
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

###################################
# 5) Dataset 정의
###################################
class ReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = row["prompt_input"]
        target_text = row["output"]

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze(),
        }

train_dataset = ReviewDataset(train_data, tokenizer)
val_dataset = ReviewDataset(val_data, tokenizer)

###################################
# 6) Accelerate 설정
###################################
accelerator = Accelerator(mixed_precision="fp16")

###################################
# 7) DataLoader, Optimizer, Scheduler 정의
###################################
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

###################################
# (추가) 체크포인트 불러오기(모델만)
###################################
resume_checkpoint = "best_model.pth"  # 불러올 체크포인트 경로
if os.path.exists(resume_checkpoint):
    print(f"[INFO] Found checkpoint at {resume_checkpoint}. Loading...")
    # 아직 Accelerator.prepare 전이므로 그냥 model.load_state_dict(...) 가능
    state_dict = torch.load(resume_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    print("[INFO] Checkpoint loaded!")

###################################
# 8) Accelerator 준비
###################################
model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

if hasattr(model.config, "use_cache"):
    model.config.use_cache = False
model.gradient_checkpointing_enable()

###################################
# 9) 학습 루프 (체크포인트 저장/로드 포함)
###################################
best_val_loss = float('inf')
patience = 3
early_stop_counter = 0

for epoch in range(num_epochs):
    accelerator.print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    # Training
    model.train()
    total_train_loss = 0.0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training", disable=not accelerator.is_local_main_process)):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    accelerator.print(f"Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    accelerator.print(f"Validation Loss: {avg_val_loss:.4f}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        if accelerator.is_main_process:
            accelerator.print("Best model saved.")
            # 모델 파라미터만 저장
            # (옵티마이저/스케줄러 상태까지 저장하려면 accelerator.save_state() 사용)
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                "best_model.pth"
            )
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            accelerator.print("Early stopping triggered.")
            break

accelerator.print("Training Complete!")
