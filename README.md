# 🔐 OWASP-LLM-SECURITY-LABS

> [!WARNING]
> **EĞİTİM AMAÇLI UYARI:** Bu proje yalnızca siber güvenlik eğitim ve araştırma faaliyetleri için geliştirilmiştir. Gerçek sistemlere yönelik saldırı denemeleri veya etik dışı kullanım için uygun değildir. Tüm sorumluluk kullanıcıya aittir.

Tek bir LLM modeli üzerinde fine-tuning, manuel jailbreak testleri ve OWASP Top 10 güvenlik analizi yapılan uygulamalı bir güvenlik laboratuvarı.

## 🎯 Proje Amacı

Bu proje, tek bir Large Language Model (LLM) üzerinde gerçekleştirilen güvenlik testleriyle modelin davranışsal zafiyetlerini analiz etmeyi amaçlayan uygulamalı ve deneysel bir güvenlik laboratuvarıdır. Proje kapsamında şu alanlara odaklanılmaktadır:

*   **Prompt Injection:** Jailbreak girişimlerine karşı model davranışı.
*   **Supply Chain:** Üçüncü taraf iddialarına ve veri zehirlenmesi risklerine verilen tepkiler.
*   **Reliability:** Hallucination (halüsinasyon) ve görev sapması eğilimleri.
*   **Compliance:** OWASP Top 10 for LLM Applications kapsamındaki risk analizleri.

> [!IMPORTANT]
> Projenin ana odak noktası model performansı değil, model güvenliği ve davranışsal zafiyetlerin tespitidir.

---

## 📌 Mevcut Kapsam ve Durum

> [!TIP]
> **🚧 AKTİF GELİŞTİRME:** Bu proje sürekli olarak güncellenmekte ve yeni güvenlik test senaryoları eklenmektedir. Geliştirmeler devam etmektedir.

| Özellik | Durum |
| 1 adet LLM Modeli | ✅ |
| Fine-tuning (LoRA / PEFT) | ✅ |
| Manuel Etkileşimli Test Sistemi | ✅ |
| OWASP Top 10 Güvenlik Taramaları | ✅ |
| RAG Entegrasyonu | **❌ (Gelecek Planı)** |

---

## 🧠 Model ve Veri

### Model Detayları
*   **Base Model:** `TinyLlama-1.1B-Chat`
*   **Fine-tuning Yöntemi:** LoRA (PEFT)
*   **Trainable Parametre Oranı:** ~%0.2

### Veri Seti
*   **Kaynak:** CICIDS2017 tabanlı ağ trafiği verisi
*   **Çıkış Etiketleri:**
    1.  `BENIGN` (Normal Trafik)
    2.  `ATTACK` (Saldırı Trafiği)

> [!NOTE]
> Model, güvenlik gereği yalnızca bu iki etiketi üretecek şekilde kısıtlanmıştır.

---

## 🛡️ Güvenlik Yaklaşımı

Güvenlik tek katmanlı değil, çok aşamalı (defense-in-depth) bir yaklaşımla ele alınmıştır:

### 1️⃣ Girdi Kontrolleri (Input Controls)
*   Prompt injection ve role-play tespiti.
*   Zararlı patern (system prompt ifşası, script enjeksiyonu vb.) engelleme.

### 2️⃣ Çıktı Kısıtlaması (Output Guardrails)
*   Çıktıların katı bir şekilde `BENIGN` veya `ATTACK` ile sınırlandırılması.
*   Etiket dışı üretimlerin bastırılması ve zararlı çıktıların sanitize edilmesi.

### 3️⃣ Davranış Analizi (Behavioral Analysis)
*   **Hallucination** tespiti.
*   **Supply chain** uydurma ve yanlış bilgi yayma kontrolü.
*   **Görev sapması** (task drift) analizi.

---

## 🧪 Test Metodolojisi

### Manuel Test Senaryoları
*   **Prompt Injection:** Instruction override denemeleri.
*   **Semantic Jailbreak:** Role-play ve dolaylı manipülasyon testleri.
*   **Supply Chain Probing:** Modelin eğitim verisi veya kaynağı hakkındaki uydurma bilgileri doğrulama eğilimi.
*   **Robustness:** Anlamsız, bozuk veya sınır değerlerdeki girdilere verilen tepkiler.

### İzlenen Metrikler
Her test oturumunda şu veriler kayıt altına alınır:
*   **Final Etiket:** Modelin ürettiği sonuç.
*   **Ham Çıktı:** Filtreleme öncesi modelin gerçek davranışı.
*   **Güvenlik Durumu:** Zararlı girişimin engellenip engellenmediği.
*   **Performans:** Çıkarım süresi (latency).
