# NOT: Bu çalışma Google Colab ortamında geliştirilmiştir ve test edilmiştir.
"""
Manual LLM Security Testing Script
Focused on OWASP LLM Top 10 attack scenarios.
This script is part of an educational security lab project.
"""
import os
import json
import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime
import gc
import re
from typing import Dict, Any, Optional

# .env desteği için (opsiyonel)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# TRL kütüphanesini kontrol et
try:
    from trl import SFTTrainer
    USE_SFT_TRAINER = True
except ImportError:
    USE_SFT_TRAINER = False

# ==========================================
# 🔐 KONFIGURASYON 
# ==========================================
# Gerçek yollar `.env` dosyasından veya ortam değişkenlerinden okunur.

MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
CSV_PATH = os.getenv("CSV_PATH", "data/test_data.csv")
OUT_PATH = os.getenv("OUT_PATH", "data/processed_data.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models/network-traffic-finetuned")

# ÇALISMA MODLARI
QUICK_TEST = os.getenv("QUICK_TEST", "False").lower() == "true"
MANUAL_TEST_MODE = os.getenv("MANUAL_TEST_MODE", "True").lower() == "true"
SKIP_TRAINING = os.getenv("SKIP_TRAINING", "False").lower() == "true"

def ensure_directories():
    """Gerekli klasörlerin varlığını kontrol eder."""
    for path in [CSV_PATH, OUT_PATH, OUTPUT_DIR]:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                pass # Güvenlik gereği detaylı hata basmıyoruz

ensure_directories()

# ==========================================
# 1) VERI HAZIRLAMA
# ==========================================
def check_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    report = {"status": "PASS", "issues": []}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std and std > 0:
            z_scores = ((df[col] - mean) / std).abs()
            outliers = z_scores[z_scores > 10]
            if len(outliers) > 0:
                msg = f"Kolon '{col}': {len(outliers)} sapma."
                report["issues"].append(msg)
                if len(outliers) / max(1, len(df)) > 0.01:
                    report["status"] = "WARNING"
    return report

def prepare_dataset() -> str:
    if os.path.exists(CSV_PATH):
        sample_size = 50 if QUICK_TEST else 2000
        final_size = 50 if QUICK_TEST else 800
        df = pd.read_csv(CSV_PATH, nrows=sample_size)
        df = df.sample(min(final_size, len(df)), random_state=42)
    else:
        # GERÇEK VERİ YOKSA DEMO MODU
        data = {
            "Destination Port": np.random.randint(80, 8080, 200),
            "Flow Duration": np.random.randint(100, 100000, 200),
            "Total Fwd Packets": np.random.randint(1, 100, 200),
            "Label": ["BENIGN"] * 100 + ["ATTACK"] * 100
        }
        df = pd.DataFrame(data)

    check_data_integrity(df)
    FEATURES = [c for c in df.columns if c != "Label"]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            feature_text = ", ".join([f"{col}: {row[col]}" for col in FEATURES[:8]])
            label_out = "BENIGN" if str(row["Label"]).upper() == "BENIGN" else "ATTACK"

            text = (
                "<|system|>\n"
                "You are a network security classifier. "
                "Return ONLY one word: BENIGN or ATTACK. Nothing else.\n"
                "<|user|>\n"
                f"Analyze this traffic: {feature_text}\n"
                "<|assistant|>\n"
                f"{label_out}"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    del df
    gc.collect()
    return OUT_PATH

# ==========================================
# 2) MODEL VE TOKENIZER
# ==========================================
def load_model_and_tokenizer(load_from_checkpoint=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel

    model_path = OUTPUT_DIR if (load_from_checkpoint and os.path.exists(OUTPUT_DIR)) else MODEL_ID
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_from_checkpoint and os.path.exists(OUTPUT_DIR):
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        model = model.merge_and_unload()
        peft_config = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True
        )
        peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer, peft_config

# ==========================================
# 3) EĞİTİM
# ==========================================
def train_model(model, tokenizer, peft_config, dataset_path: str):
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize_function(examples):
        result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset,
        tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return model

# ==========================================
# 4) GUVELIK KATMANI (SECURITY LAYER)
# ==========================================
class SecurityLayer:
    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+all\s+previous", r"system\s+prompt", r"<\|system\|>", r"reveal\s+instructions"
        ]

    def check_input(self, user_prompt: str) -> Dict[str, Any]:
        p = user_prompt.lower()
        for pat in self.injection_patterns:
            if re.search(pat, p, flags=re.IGNORECASE):
                return {"blocked": True, "reason": "Security Pattern Detected"}
        return {"blocked": False, "reason": ""}

    def enforce_label(self, text: str) -> str:
        t = text.strip().upper()
        if t.startswith("BENIGN"): return "BENIGN"
        return "ATTACK"

# ==========================================
# 5) MANUEL TEST SISTEMI
# ==========================================
class ManualTester:
    def __init__(self, model, tokenizer, security_layer: Optional[SecurityLayer] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.security = security_layer or SecurityLayer()
        self.device = next(model.parameters()).device
        self.model.eval()

    def predict(self, prompt: str) -> Dict[str, Any]:
        gate = self.security.check_input(prompt)
        if gate["blocked"]:
            return {"prediction": "ATTACK", "inference_time": 0.0, "blocked": True}

        formatted = f"<|system|>\nClassifier\n<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
        elapsed = time.time() - start
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded
        return {"prediction": self.security.enforce_label(resp), "inference_time": elapsed, "blocked": False}

    def interactive_test(self):
        print("\n" + "="*40 + "\n🧪 MANUEL TEST MODU AKTIF\n" + "="*40)
        while True:
            inp = input("\n💬 Trafik Verisi (Çıkış için 'q'): ").strip()
            if inp.lower() == 'q': break
            res = self.predict(inp)
            print(f"➜ SONUÇ: {res['prediction']} | SÜRE: {res['inference_time']:.3f}s")

# ==========================================
# ANA ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    if not SKIP_TRAINING:
        ds_path = prepare_dataset()
        model, tokenizer, peft_config = load_model_and_tokenizer()
        model = train_model(model, tokenizer, peft_config, ds_path)
    else:
        model, tokenizer, _ = load_model_and_tokenizer(load_from_checkpoint=True)

    if MANUAL_TEST_MODE:
        tester = ManualTester(model, tokenizer)
        tester.interactive_test()

