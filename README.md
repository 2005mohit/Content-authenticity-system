# AI-Generated Content Detection System

A web-based system to detect whether **text and images are AI-generated or human-created** using Machine Learning.

---

## Team Members

| Name | Roll No |
|------|---------|
| Praduman Kumar | 2501560002 |
| Payal Jain | 2501560009 |
| Mohit Chandra Fulara | 2501560010 |
| Anurag Ramola | 2501560023 |
| Kartik Kumar | 2501560029 |

**Industry Mentor:** Mr. Amit Aggarwal
**Faculty Mentor:** Mr. Devansh Garg
**University:** K.R. Mangalam University | MCA 2025–2026

---

## Project Overview

Generative AI tools like ChatGPT, Claude, DALL·E, and Midjourney can create realistic content instantly, leading to:

- Plagiarism & academic dishonesty
- Fake news & misinformation
- Deepfake images

### Solution

A **single web application** that detects:
- AI-generated **text**
- AI-generated **images**

### Output

- Authenticity Score (0–100%)
- Label → *Likely AI / Likely Human*
- Confidence %

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **Frontend** | Streamlit |
| **Image Processing** | OpenCV, Pillow |
| **Data & ML** | Scikit-learn, NumPy, Pandas |

---

## System Architecture

User Input (Text / Image)
↓
Streamlit UI (app.py)
↓
Text Pipeline Image Pipeline
↓ ↓
RoBERTa (40%) EfficientNet B0
+ ↓
Heuristics (60%) Softmax Output
↓ ↓
Ensemble Score AI / Real Label
↓ ↓
Authenticity Score + Confidence %

---

## Project Structure

├── app.py
├── pipeline/
│ ├── text_pipeline.py
│ └── image_pipeline.py
├── model/
│ ├── text_model/
│ └── image_model/
├── requirements.txt
└── README.md

---

## Text Detection Module

### Model Used
- **RoBERTa** (`chatgpt-detector-roberta` — HuggingFace)
- Fine-tuned for binary classification (AI vs Human)

### Ensemble Approach
- Final AI Score = (0.4 × RoBERTa Score) + (0.6 × Heuristic Score)

| Component | Weight | Type |
|-----------|--------|------|
| RoBERTa Classifier | 40% | Deep Learning |
| Heuristic Detector | 60% | Rule-Based |

### Heuristic Features Analyzed
- Perplexity
- Burstiness (sentence length variance)
- Vocabulary Diversity
- Average Sentence Length
- N-gram Patterns

### Dataset

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| HC3 | HuggingFace | ~50,000 | ChatGPT + Human responses |
| MEGA | Kaggle | ~30,000 | Diverse AI & human text |
| **Total** | Combined | **~80,000** | 70% Train / 15% Val / 15% Test |

- Max token length: 512
- Balanced class distribution

### Performance
- **Test Accuracy: ~90%**
- Robust against: ChatGPT, Claude, Gemini, paraphrased AI text

---

## Image Detection Module

### Model Used
- **EfficientNet-B0** (Pretrained on ImageNet)
- Fine-tuned for binary classification (AI vs Real)

### Training Strategy
- **Phase 1** → Freeze backbone, train classifier head only
- **Phase 2** → Unfreeze all layers, full fine-tuning

### Dataset

| Dataset | Source | Size | Split |
|---------|--------|------|-------|
| DefactifyCOCOAI | HuggingFace | ~48,000 images | 21k Train / 4.5k Val / 22.5k Test |

- Resolution: 224×224
- Classes: Real images & AI-generated images
- Balanced via undersampling (1:1 ratio)

### Detection Features
- Pixel anomalies
- Texture artifacts
- Lighting inconsistencies
- Edge irregularities

### Performance
- **Test Accuracy: ~70%**

---

## Datasets — Combined Overview

| Modality | Dataset | Source | Size | Split |
|----------|---------|--------|------|-------|
| Text | HC3 | HuggingFace | ~50k | 70/15/15 |
| Text | MEGA | Kaggle | ~30k | 70/15/15 |
| Image | DefactifyCOCOAI | HuggingFace | ~48k | 21k / 4.5k / 22.5k |

---

## Authenticity Scoring
Authenticity Score = (1 - AI_Probability) × 100

| Score Range | Label |
|-------------|-------|
| 0 – 30 | 🔴 Likely AI |
| 30 – 70 | 🟡 Uncertain |
| 70 – 100 | 🟢 Likely Human |

---

## Web Application

**Built with Streamlit**

### Features
- **2 Detection Tabs**: Text Detection | Image Detection
- Color-coded results: 🔴 AI-Generated / 🟢 Human/Real
- Real-time inference (< 3 seconds)
- Confidence score + probability breakdown

---

## Results Summary

| Module | Model | Accuracy | Inference Time |
|--------|-------|----------|----------------|
| Text | RoBERTa + Heuristics | ~90% | < 1 sec |
| Image | EfficientNet B0 | ~70% | < 2 sec |

- Tested on real-world AI-generated outputs
- Both models deployed on Streamlit Community Cloud

---

## Deployment

- **Current**: [Streamlit Community Cloud](https://content-appenticity-system-entcrhhe9unn8pxhyjb2rp.streamlit.app/)
- **Planned**:
  - Render
  - HuggingFace Spaces

---

## Use Cases

- **Education** → Detect AI-written assignments
- **Media** → Verify image & news authenticity
- **Corporate** → Ensure original reports
- **Research** → Screen AI-generated papers
- **Cybersecurity** → Deepfake image detection

---

## Limitations

- Results are probabilistic and should not be used as a definitive verdict
- Image detection accuracy varies with image quality and generation style
- Hard cases:
  - Short text (< 40 words)
  - Heavily edited or paraphrased AI content

---

## Future Work

- [ ] Improve real-world image accuracy beyond 70% through domain adaptation
- [ ] Add Grad-CAM heatmaps for image explainability
- [ ] Handle multi-language text detection
- [ ] Deploy a publicly accessible demo on HuggingFace Spaces
- [ ] Detect mixed AI-human content

---

## Project Status

**~90% Complete**

| Task | Status |
|------|--------|
| Text Model Training | Done |
| Image Model Training | Done |
| Pipelines Built | Done |
| Web App Deployed | Done |
| Final Testing | In Progress |

---

## Acknowledgments

Special thanks to our mentors and the open-source community:

- **Mr. Amit Aggarwal** (Industry Mentor) — for real-world project guidance
- **Mr. Devansh Garg** (Faculty Mentor) — for academic mentorship
- **HuggingFace** — for pretrained models and datasets
- **Kaggle** — for dataset resources
- **K.R. Mangalam University** — for infrastructure and support
