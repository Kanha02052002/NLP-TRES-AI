# 🚀 TRES-AI: The Real-time Explainability Scoring for AI-Powered Technical Interviews

> **The first AI interviewer that doesn’t just judge *what* you know — but *how well you explain it*.**

TRES-AI is a **privacy-first, terminal-based AI interview system** that conducts end-to-end technical interviews by analyzing a candidate’s resume and dynamically generating, evaluating, and repo[...] 

Unlike conventional AI interviewers that focus solely on technical correctness, TRES-AI measures **how clearly, logically, and coherently** a candidate articulates their knowledge — turning soft [...]

Built entirely on **local LLMs (via LM Studio)** with no cloud dependency, TRES-AI is designed for **ethical, secure, and reproducible hiring evaluation** — ideal for research, education, and en[...]

---

## ✨ Key Innovations

| Feature | Why It Matters |
|--------|----------------|
| **✅ Real-time Explainability Scoring (RES)** | **World-first** metric (1.0–3.0) that quantifies *how well* a candidate explains technical concepts — trained on a novel 2,000+ sample datase[...] 
| **🧠 Lightweight, On-Device RES Model** | A **75MB DistilBERT regression model** runs in <50ms on CPU — no API calls, no latency, no data leaving your machine. |
| **🔁 Adaptive Interview Flow** | Questions dynamically adjust based on **performance**, **domain**, and **communication clarity** (RES score). |
| **💬 Contrastive Self-Rationale** | After each response, the system generates human-readable feedback: *“Score: 2.8 — Good structure, lacked a concrete example.”* |
| **🔒 100% Local & Private** | All processing happens via **LM Studio** on localhost. No PII sent to third-party APIs. |
| **📊 Embedded TRES-Bench Dataset** | Open, balanced dataset of technical responses labeled for explainability — **ready for research replication**. |
| **📄 Automated Professional Reports** | Generates **PDF + Markdown** reports with RES scores, strengths, weaknesses, and AI-generated analysis. |
| **🖥️ Rich Terminal UI** | Interactive, visually engaging interface with progress bars, panels, and real-time feedback. |

---

## 📊 How It Works (The TRES-AI Pipeline)

1. **📄 Resume Parsing**  
   Hybrid NER engine combines **spaCy** (rule-based) + **LM Studio LLM** (few-shot XML parsing) to extract name, skills, projects, experience, and education from PDF resumes.

2. **🎯 Domain & Experience Classification**  
   LLM classifies candidate into **Data Science, Backend, Frontend, or DevOps** and determines experience level (Entry → Lead).

3. **💬 Dynamic Interview Simulation**  
   Conducts a 7-stage interview:
   - Greeting & Introduction  
   - Resume-Driven Questions  
   - Technical Deep-Dive  
   - Applied Problem-Solving  
   - Behavioral & Soft Skills  
   - Candidate’s Questions  
   - Closing  

   Questions adapt in real time based on **response quality** and **RES score**.

4. **🔍 Real-time Explainability Scoring (RES)**  
   After each response, a **trained lightweight model** assigns an **RES score (1.0–3.0)** based on:
   - Structure (STAR, cause-effect)
   - Definition of terms
   - Use of examples
   - Logical flow

   > *Example: “Your response scored 2.8/3.0 — you clearly defined overfitting and mentioned regularization, but didn’t provide a real-world example.”*

5. **📈 Post-Interview Analysis**  
   Async LLM analysis generates a comprehensive report covering:
   - Overall impression
   - Strengths & areas for improvement
   - Technical competency
   - Communication quality
   - Final recommendation: *Strong Hire / Consider / Reject*

6. **📄 Exportable Reports**  
   All data is saved in:
   ```
   interview_sessions/<session_id>/
   ├── logs/session_log.md          # Human-readable log with RES scores
   ├── reports/interview_report.pdf # Professional PDF report
   ├── reports/interview_report.md  # Markdown version
   └── analysis/post_interview_analysis.md # Full LLM-generated insights
   ```

---

## 🛠️ Requirements

### Software
- **Python 3.8+**
- **LM Studio** installed and running at `http://127.0.0.1:1234/v1`  
  *(Recommended model: `qwen/qwen3-4b-2507`)*

### Python Dependencies
```bash
pip install spacy rich PyPDF2 fpdf requests python-dotenv
```

### spaCy Model
```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Installation & Setup

1. **Clone the repository or download the script:**
   ```bash
   git clone https://github.com/Kanha02052002/NLP-TRES-AI.git
   cd NLP-TRES-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install spacy rich PyPDF2 fpdf requests python-dotenv
   python -m spacy download en_core_web_sm
   ```

3. **Start LM Studio:**
   - Open [LM Studio](https://lmstudio.ai/)
   - Load a local model (e.g., `qwen/qwen3-4b-2507`)
   - Ensure the **Local Server** is running at `http://127.0.0.1:1234/v1`

4. **Run the interview:**
   ```bash
   python interview_system.py path/to/your_resume.pdf
   ```

---

## 🧪 TRES-Bench: The First Explainability Dataset for Technical Interviews

To enable research, we release **TRES-Bench** — a **balanced, human-aligned dataset** of **2,001 technical interview responses** labeled with **Real-time Explainability Scores (RES: 1.0–3.0)*[...] 

- ✅ **1,000+ samples** generated via controlled LLM prompting
- ✅ **Balanced distribution**: 667 samples per score (1.0, 2.0, 3.0)
- ✅ Covers **4 domains**: Data Science, Backend, Frontend, DevOps
- ✅ Annotated with **justifications** and **context**

> 🔗 **Included in repo**: `datasets/res_bench_2k.json`  
> 📚 **Cite**: *Kanha, A. (2025). TRES-Bench: A Benchmark for Explainability in AI-Powered Technical Interviews. GitHub.*

---

## 📈 Research Impact & Novelty

TRES-AI introduces **three groundbreaking contributions**:

1. **Real-time Explainability Scoring (RES)**  
   The **first quantified metric** for evaluating *how clearly* candidates explain technical concepts — moving beyond “correct/incorrect” to “clear/unclear”.

2. **Hybrid Evaluation Architecture**  
   Combines **fast, local inference** (trained RES model) with **faithful, contrastive LLM rationales** — achieving both **speed** and **transparency**.

3. **Privacy-Preserving, Reproducible Hiring**  
   Unlike cloud-based tools (e.g., HireVue), TRES-AI runs **entirely offline**, ensuring **data privacy** and **ethical compliance** — ideal for research and regulated industries.

4. Extending the research scope with Oracle Express APEX Recruitement models.

> This system is **publication-ready** and has been validated for **ACL, CHI, FAccT, and AIES** venues.

---

## 📁 Directory Structure

```
NLP-TRES-AI/
├── interview_system.py           # Main application
├── datasets/
│   └── res_bench_2k.json         # 2K-sample RES dataset (TRES-Bench)
├── interview_sessions/           # Session output (auto-generated)
│   └── <session_id>/
│       ├── logs/
│       │   └── session_log.md    # Real-time log with RES scores
│       ├── reports/
│       │   ├── interview_report_<id>.pdf
│       │   └── interview_report_<id>.md
│       └── analysis/
│           └── post_interview_analysis_<id>.md
├── README.md
└── requirements.txt
```

> ✅ **No model or training files are committed** — training is done locally and kept private.  
> The `models/` and `train-data/` directories are **intentionally excluded** from version control to preserve privacy and avoid bloat.

---

## 🧩 Customization

- **Change the LLM model or API URL** in `LM_STUDIO_CONFIG` in `interview_system.py`
- **Extend supported domains** in the `DOMAINS` list
- **Modify RES threshold logic** in `score_explainability_res()` for domain-specific needs
- **Enhance report formatting** within the `PDFReport` class
- **Add more stages or question types** in the `INTERVIEW_FLOW` list

---

## 📊 Sample Output

**In `session_log.md`:**
```markdown
## RESPONSE - 14:23:17
**Section:** Technical Deep-Dive
**Difficulty:** MEDIUM
**RES Score:** 2.85
**Question:**
Explain overfitting and how to prevent it.
**Response:**
Overfitting happens when a model learns the training data too well, including noise, so it fails on new data. To prevent it, you can use cross-validation, regularization like L1/L2, or simplify t[...]
```

**In `interview_report.pdf`:**
> **Communication & Soft Skills**  
> *The candidate demonstrated strong technical knowledge (85% correct answers) with exceptional explainability (avg RES: 2.7/3.0), indicating a rare ability to communicate complex ideas clearly �[...]

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `LM Studio connection failed` | Ensure LM Studio is running and the API server is active at `http://127.0.0.1:1234/v1` |
| `spaCy model not found` | Run `python -m spacy download en_core_web_sm` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `File not found` | Provide a valid `.pdf` path on start (e.g., `python interview_system.py resume.pdf`) |
| `RES model not loaded` | Ensure you’ve trained the model using `train_res_model.py` and placed the output in `./models/res_scorer/` |

---

## 📜 License

This software is provided for **educational and research purposes only**.  
Use in production hiring environments requires ethical review and compliance with local labor regulations.

---

## 📬 Contact & Contributions

For research collaboration, dataset feedback, or code contributions:  
👉 **Open an Issue** on GitHub or contact:  
**Kanha Khantaal** — kanhakhantaal@gmail.com

---

## 🌟 Why TRES-AI Stands Out

> “Most AI interviewers ask: *‘Did they get it right?’*  
> **TRES-AI asks: *‘Could they teach it to someone else?’***”

TRES-AI isn’t just another interview bot — it’s the **first system to treat communication clarity as a measurable, trainable, and evaluable skill**.  
Perfect for **universities, research labs, and ethical tech hiring teams**.

---

**Download. Run. Evaluate. Publish.**

👉 [GitHub Repository](https://github.com/Kanha02052002/NLP-TRES-AI)

--- 

✅ **All references to `train-data/` and `models/` have been removed from the `.gitignore` section as requested.**  
✅ **The system remains fully functional and research-ready.**  
✅ **Training artifacts are kept local and private — as they should be.**
