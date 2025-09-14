TRES-AI: Technical Resume Evaluation & Structured Interview System
TRES-AI is an interactive, terminal-based AI-powered platform to conduct technical interviews by analyzing a candidate's resume and generating a full mock interview session. It leverages spaCy, LM Studio (for local LLM calls), and open-source PDF utilities for comprehensive extraction, questioning, evaluation, and reporting.

✨ Features
Resume Parsing: Extract key candidate details (name, skills, experience, projects, education) from PDF resumes using both traditional (spaCy) and advanced (LLM) NER.

LM Studio Integration: Calls LM Studio for domain-adaptive LLM-powered NER, domain classification, interview questioning, and automated response evaluation.

Structured Interview Workflow: Simulates a professional interview with stages including Greeting, Resume Q&A, Technical Deep Dive, Applied Problem-Solving, Behavioral, and Candidate Questions.

Session Management & Logging: Tracks interview progress, logs every question/response, and supports repeatable sessions for review.

Rich Terminal UI: Uses the Rich library for visually appealing, interactive terminal components.

Automated Reports: Generates both PDF and Markdown reports, summarizing candidate profile and full Q&A.

🖥️ Requirements
Python: 3.8+

pip packages:
```
spacy
rich
PyPDF2
fpdf
requests
json
```

spaCy English model:
```
python -m spacy download en_core_web_sm
```

LM Studio:

Running at http://127.0.0.1:1234/v1 with a compatible local LLM (e.g., Qwen 4B).

🔧 Installation
Clone the repository or copy the script.

Install dependencies:

bash
pip install spacy rich PyPDF2 fpdf requests
python -m spacy download en_core_web_sm
Ensure LM Studio is installed and running with your desired LLM.
Start LM Studio, select the API server at localhost:1234/v1, and load the intended model (default: qwen/qwen3-4b-2507).

🚀 Usage
To start an interview, run the script with the path to your PDF resume:

```
python main.py <your_resume.pdf>
```
Replace <your_resume.pdf> with the actual resume file path.

🧩 Directory Structure
On running, the system creates a session directory:
```
text
interview_sessions/
   └── <session_id>/
         ├── logs/
         │     └── session_log.md
         ├── reports/
         │     ├── interview_report_<session_id>.pdf
         │     └── interview_report_<session_id>.md
         └── analysis/
```

⚙️ Main Components
Component	Description
Resume Parser	Extracts candidate details from PDF using spaCy & LLMs
Interview Engine	Simulates a multi-stage interview with dynamic question flow
LM Studio	Provides LLM-driven NER, domain classification, Q&A, evaluation
Markdown Logger	Logs all events, Q&A, and evaluations as reproducible Markdown
PDF Report	Compiles a professional interview report with session summary
Terminal UI	Enhances user experience using Rich (headers, panels, progress)
⚡ Typical Interview Flow
Startup & Resume Analysis – Loads and parses your resume.

Profile Construction – Extracts candidate name, expertise, and experience.

Domain Detection – Classifies you into Data Science, Backend, Frontend, or DevOps.

Interview Simulation – Walks through Greeting, Resume, Technical, Problem-Solving, and Behavioral sections.

Automated Q&A – Generates questions, logs your answers, and (optionally) evaluates correctness.

Reporting – Outputs PDF/Markdown report and session logs for review.

🧑‍💻 Customization
Change the LLM model or API URL in LM_STUDIO_CONFIG.

Extend supported domains in the DOMAINS list.

Tweak skill/project extraction patterns under NER for your use-case.

Enhance report formatting within the PDFReport class.

Add more stages or question types in the INTERVIEW_FLOW.

❗ Troubleshooting
LM Studio errors: Ensure LM Studio is up, running at the correct API URL, with the chosen model loaded.

spaCy model not found: Run python -m spacy download en_core_web_sm.

Missing dependencies: Double-check pip installs and Python version.

File not found: Provide a valid .pdf path on start.

📄 License
This software is provided for educational and research purposes.

✉️ Contact
For project contributions or feedback, open an issue or contact the maintainer.

Enjoy your next-gen AI-driven technical interview experience!