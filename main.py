import PyPDF2
import requests
import uuid
from datetime import datetime, timedelta
import sys
import re
import spacy
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.text import Text
from rich import box
import time
import logging
from fpdf import FPDF
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

LM_STUDIO_CONFIG = {
    "base_url": "http://127.0.0.1:1234/v1",
    "api_key": "not-needed",
    "default_model": "qwen/qwen3-4b-2507",
    "timeout": 60,
    "headers": {
        "Content-Type": "application/json"
    }
}

DOMAINS = ["Data Science", "Frontend", "Backend", "DevOps"]
INTERVIEW_FLOW = [
    {"name": "Greeting & Introduction", "description": "Warm up and test communication skills", "question_count": 1},
    {"name": "Resume-Driven Questions", "description": "Verify resume and ease candidate in", "question_count": 2},
    {"name": "Technical Deep-Dive", "description": "Test core domain knowledge progressively", "question_count": 3},
    {"name": "Applied Problem-Solving", "description": "Test real-world thinking", "question_count": 1},
    {"name": "Behavioral & Soft Skills", "description": "Assess team fit and personality", "question_count": 2},
    {"name": "Candidate's Questions", "description": "Check curiosity and motivation", "question_count": 1},
    {"name": "Closing", "description": "End politely and leave a positive note", "question_count": 1}
]
MAX_SKILLS_DISPLAY = 10
MAX_PROJECTS_DISPLAY = 5
EXPERIENCE_LEVELS = {
    "ENTRY_LEVEL": (0, 2), "JUNIOR": (2, 5), "MID_LEVEL": (5, 8), "SENIOR": (8, 15), "LEAD": (15, 50)
}

console = Console()

logging.basicConfig(filename='interview_terminal.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully")
except OSError:
    console.print("[bold red]Please install spaCy English model:[/bold red] python -m spacy download en_core_web_sm")
    logging.error("Failed to load spaCy model")
    sys.exit(1)

# ==================== SESSION DIRECTORIES ====================
def create_session_directories(session_id):
    """Create directories for session logs and reports"""
    base_dir = f"interview_sessions/{session_id}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)
    os.makedirs(f"{base_dir}/reports", exist_ok=True)
    os.makedirs(f"{base_dir}/analysis", exist_ok=True)
    return base_dir

# ==================== MODEL UTILITY FUNCTIONS ====================
def call_lm_studio_model(prompt, max_tokens=500, temperature=0.7, stop=None):
    """Call LM Studio model with standardized parameters"""
    try:
        data = {
            "model": LM_STUDIO_CONFIG["default_model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop
        }
        logging.info(f"Sending request to LM Studio with prompt (first 100 chars): {prompt[:100]}...")
        response = requests.post(
            f"{LM_STUDIO_CONFIG['base_url']}/chat/completions",
            headers=LM_STUDIO_CONFIG["headers"],
            json=data,
            timeout=LM_STUDIO_CONFIG["timeout"]
        )
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip()
            logging.info(f"LM Studio response received (first 100 chars): {result[:100]}...")
            return result
        else:
            logging.error(f"LM Studio API error: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"LM Studio connection error: {str(e)}")
        return None

async def call_lm_studio_model_async(session, prompt, max_tokens=500, temperature=0.7, stop=None):
    """Async version of call_lm_studio_model using aiohttp"""
    try:
        data = {
            "model": LM_STUDIO_CONFIG["default_model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop
        }
        logging.info(f"Async: Sending request to LM Studio with prompt (first 100 chars): {prompt[:100]}...")
        async with session.post(
            f"{LM_STUDIO_CONFIG['base_url']}/chat/completions",
            headers=LM_STUDIO_CONFIG["headers"],
            json=data,
            timeout=aiohttp.ClientTimeout(total=LM_STUDIO_CONFIG["timeout"])
        ) as response:
            if response.status == 200:
                result_json = await response.json()
                result = result_json["choices"][0]["message"]["content"].strip()
                logging.info(f"Async: LM Studio response received (first 100 chars): {result[:100]}...")
                return result
            else:
                logging.error(f"Async: LM Studio API error: {response.status}")
                return None
    except Exception as e:
        logging.error(f"Async: LM Studio connection error: {str(e)}")
        return None

# ==================== MARKDOWN LOGGER ====================
class MarkdownLogger:
    def __init__(self, session_id, base_dir):
        self.session_id = session_id
        self.base_dir = base_dir
        self.log_file = f"{base_dir}/logs/session_log.md"
        self.session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "events": [],
            "responses": [],
            "evaluations": []
        }
        with open(self.log_file, 'w', encoding='utf-8') as f: 
            f.write(f"# Interview Session Log - {session_id}\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("---\n")
        logging.info(f"Markdown logger initialized for session {session_id}")

    def log_event(self, event_type, message, data=None):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data
        }
        self.session_data["events"].append(event)
        with open(self.log_file, 'a', encoding='utf-8') as f: 
            f.write(f"## {event_type.upper()} - {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"**Message:** {message}\n")
            if data is not None:
                f.write(f"**Data:** {json.dumps(data, indent=2)}\n")
            f.write("---\n")
        logging.info(f"[{event_type.upper()}] {message}")

    def log_response(self, question, response, section, difficulty="N/A"):
        response_data = {
            "question": question,
            "response": response,
            "section": section,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat()
        }
        self.session_data["responses"].append(response_data)
        with open(self.log_file, 'a', encoding='utf-8') as f: 
            f.write(f"## RESPONSE - {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"**Section:** {section}\n")
            f.write(f"**Difficulty:** {difficulty}\n")
            f.write(f"**Question:**\n{question}\n")
            f.write(f"**Response:**\n{response}\n")
            f.write("---\n")
        logging.info(f"Response logged for section {section}")

    def log_evaluation(self, question, response, evaluation_data, section):
        eval_data = {
            "question": question,
            "response": response,
            "evaluation": evaluation_data,
            "section": section,
            "timestamp": datetime.now().isoformat()
        }
        self.session_data["evaluations"].append(eval_data)
        logging.info(f"Evaluation logged for section {section}")

    def save_session_data(self):
        """Save raw session data as JSON"""
        data_file = f"{self.base_dir}/logs/session_data.json"
        with open(data_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        logging.info(f"Session data saved to {data_file}")

# ==================== RESUME PARSER ====================
def extract_text_from_pdf(pdf_path, logger):
    """Extract text from PDF resume"""
    try:
        logger.log_event("info", f"Extracting text from PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() for page in reader.pages)
        logger.log_event("success", f"Successfully extracted text, total length: {len(text)}")
        return text
    except Exception as e:
        logger.log_event("error", f"Failed to extract text from PDF: {str(e)}")
        return ""
# ==================== IMPROVED TRADITIONAL NER (spaCy + HF fallback) ====================
def extract_entities_traditional(text, logger):
    """
    Enhanced NER extraction using spaCy for structure and HuggingFace for name fallback.
    Extracts: Name, Skills, Projects, Work Experience, Responsibilities, Education.
    """
    logger.log_event("info", "Starting traditional NER extraction (spaCy + HuggingFace fallback)")

    try:
        # ===== Preprocess =====
        doc = nlp(text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # -----------------------------------------------------------------
        #                      SMART NAME DETECTION
        # -----------------------------------------------------------------
        name = None
        top_lines = [l for l in lines[:12] if len(l) > 1]

        edu_keywords = ['b.tech', 'm.tech', 'bsc', 'msc', 'phd', 'university', 'college', 'institute', 'degree']
        contact_keywords = ['phone', 'email', 'linkedin', 'github', 'portfolio', 'resume', 'cv']
        forbidden = edu_keywords + contact_keywords

        # 1️⃣ Try spaCy PERSON entities from top lines
        person_entities = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        if person_entities:
            for p in person_entities:
                pl = p.lower()
                if any(k in pl for k in forbidden):
                    continue
                if any(p in l for l in top_lines):
                    name = p
                    break
            if not name:
                name = max(person_entities, key=lambda s: len(s.split()))

        # 2️⃣ Regex fallback: two or more capitalized words before contacts
        if not name:
            for line in top_lines:
                if any(k in line.lower() for k in forbidden):
                    continue
                match = re.search(r'\b([A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,2})\b', line)
                if match:
                    cand = match.group(1).strip()
                    if len(cand.split()) >= 2:
                        name = cand
                        break

        # 3️⃣ HuggingFace NER fallback (dslim/bert-base-NER)
        if not name or len(name.split()) < 2:
            try:
                from transformers import pipeline
                ner_pipe = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
                sample_text = "\n".join(top_lines)
                hf_entities = ner_pipe(sample_text)
                candidates = []
                for e in hf_entities:
                    if e.get("entity_group") in ["PER", "PERSON"]:
                        word = re.sub(r'##', '', e.get("word", "")).strip()
                        if len(word.split()) >= 2 and not any(k in word.lower() for k in forbidden):
                            candidates.append(word)
                if candidates:
                    name = max(candidates, key=lambda s: len(s))
            except Exception as e:
                logger.log_event("warning", f"HuggingFace fallback unavailable: {e}")

        # 4️⃣ Cleanup
        if name:
            name = re.sub(r'[\d.,;:]+$', '', name).strip()
            name = re.sub(r'^(hi|hello|hey)\b.*', '', name, flags=re.IGNORECASE).strip()
            for bad in edu_keywords:
                name = re.sub(bad, '', name, flags=re.IGNORECASE)
            name = ' '.join([w for w in name.split() if w.isalpha() and len(w) > 1])
            if len(name.split()) < 2:
                name = None
        if not name:
            name = "Unknown"

        # -----------------------------------------------------------------
        #                      SECTION EXTRACTION
        # -----------------------------------------------------------------
        sections = re.split(r'\n(?=[A-Z][A-Z ]{2,})', text)
        section_map = {}
        for sec in sections:
            header = sec.split('\n')[0].strip()
            body = "\n".join(sec.split("\n")[1:]).strip()
            section_map[header.upper()] = body

        # ===== SKILLS =====
        skills = set()
        skills_text = ""
        for k in section_map.keys():
            if "SKILL" in k:
                skills_text = section_map[k]
                break
        if skills_text:
            skill_lines = re.split(r'[\n,;/•]', skills_text)
            for s in skill_lines:
                s = re.sub(r'[^a-zA-Z0-9+#.\- ]', '', s.strip())
                if 1 < len(s) < 40:
                    skills.add(s)
        else:
            common_skills = [
                'python', 'java', 'c++', 'flask', 'django', 'html', 'css', 'javascript',
                'react', 'mysql', 'mongodb', 'azure', 'aws', 'pandas', 'numpy', 'keras',
                'tensorflow', 'pytorch', 'powerbi', 'tableau', 'opencv', 'hugging face',
                'langchain', 'streamlit', 'nlp', 'rag', 'deep learning', 'machine learning'
            ]
            for kw in common_skills:
                if re.search(rf'\b{kw}\b', text.lower()):
                    skills.add(kw.title())

        # ===== PROJECTS =====
        projects = []
        for k in section_map.keys():
            if "PROJECT" in k or "ACADEMIC" in k:
                proj_lines = re.split(r'\n•|\n-', section_map[k])
                for p in proj_lines:
                    if len(p.strip()) > 25:
                        projects.append(p.strip())

        # ===== WORK EXPERIENCE =====
        work_experience = []
        for k in section_map.keys():
            if "EXPERIENCE" in k or "WORK" in k or "FREELANCE" in k:
                exp_lines = re.split(r'\n•|\n-', section_map[k])
                for e in exp_lines:
                    if len(e.strip()) > 25:
                        work_experience.append(e.strip())

        # ===== EDUCATION =====
        education = []
        for k in section_map.keys():
            if "EDUCATION" in k:
                ed_lines = re.split(r'\n•|\n-', section_map[k])
                for e in ed_lines:
                    if len(e.strip()) > 10:
                        education.append(e.strip())

        # -----------------------------------------------------------------
        #                      FINAL ENTITY STRUCTURE
        # -----------------------------------------------------------------
        entities = {
            "name": name,
            "skills": list(skills)[:MAX_SKILLS_DISPLAY],
            "projects": projects[:MAX_PROJECTS_DISPLAY],
            "work_experience": work_experience[:5],
            "responsibilities": [],
            "education": education[:3],
            "total_experience_years": 0
        }

        logger.log_event("success", f"Traditional NER completed — Name Detected: {name}")
        return entities

    except Exception as e:
        logger.log_event("error", f"Traditional NER failed: {str(e)}")
        return {
            "name": None,
            "skills": [],
            "projects": [],
            "work_experience": [],
            "responsibilities": [],
            "education": [],
            "total_experience_years": 0
        }


# ==================== LM STUDIO ENHANCED NER ====================
def extract_entities_with_lm_studio(text, logger):
    """Extract structured resume entities using LM Studio (XML format)."""
    logger.log_event("info", "Starting LM Studio NER extraction")
    try:
        prompt = f"""<system>
You are an expert resume parser. Extract structured information in XML format.
</system>
<task>
Extract:
1. Full Name
2. Skills
3. Projects
4. Work Experience
5. Responsibilities
6. Education
7. Total Experience Years (integer)
</task>
<resume>
{text[:2500]}
</resume>
<response>"""

        response_text = call_lm_studio_model(prompt, max_tokens=1500, temperature=0.3, stop=["</response>"])
        if not response_text:
            return None

        import xml.etree.ElementTree as ET
        xml_response = "<response>" + response_text.strip() + "</response>"
        root = ET.fromstring(xml_response)
        extraction = root.find('extraction')
        if extraction is None:
            return None

        entities = {
            "name": extraction.findtext('name', ''),
            "skills": [s.text.strip() for s in extraction.findall('.//skill') if s.text],
            "projects": [],
            "work_experience": [],
            "responsibilities": [r.text.strip() for r in extraction.findall('.//responsibility') if r.text],
            "education": [e.text.strip() for e in extraction.findall('.//degree') if e.text],
            "total_experience_years": int(extraction.findtext('total_experience_years', 0))
        }

        for proj in extraction.findall('.//project'):
            title = proj.findtext('title', '').strip()
            desc = proj.findtext('description', '').strip()
            if title:
                entities["projects"].append(f"{title}: {desc}" if desc else title)

        for exp in extraction.findall('.//experience'):
            role = exp.findtext('role', '').strip()
            company = exp.findtext('company', '').strip()
            duration = exp.findtext('duration', '').strip()
            entry = f"{role} at {company} ({duration})".strip()
            if entry:
                entities["work_experience"].append(entry)

        logger.log_event("success", "LM Studio NER extraction completed successfully")
        return entities

    except Exception as e:
        logger.log_event("error", f"LM Studio NER extraction failed: {str(e)}")
        return None


# ==================== HYBRID NER (COMBINED) ====================
def extract_entities_hybrid(text, logger):
    """Combine improved traditional NER with LM Studio output."""
    logger.log_event("info", "Starting hybrid NER extraction")
    traditional = extract_entities_traditional(text, logger)
    lm_studio = extract_entities_with_lm_studio(text, logger)

    def best_name(n1, n2):
        if not n1 or n1 == "Unknown":
            return n2 or "Unknown"
        if not n2 or n2 == "Unknown":
            return n1 or "Unknown"
        return n1 if len(n1.split()) >= len(n2.split()) else n2

    if lm_studio:
        final = {
            "name": best_name(traditional.get("name"), lm_studio.get("name")),
            "skills": sorted(set(traditional.get("skills", []) + lm_studio.get("skills", [])))[:MAX_SKILLS_DISPLAY],
            "projects": sorted(set(traditional.get("projects", []) + lm_studio.get("projects", [])))[:MAX_PROJECTS_DISPLAY],
            "work_experience": sorted(set(traditional.get("work_experience", []) + lm_studio.get("work_experience", [])))[:5],
            "responsibilities": sorted(set(traditional.get("responsibilities", []) + lm_studio.get("responsibilities", [])))[:3],
            "education": sorted(set(traditional.get("education", []) + lm_studio.get("education", [])))[:3],
            "total_experience_years": lm_studio.get("total_experience_years", traditional.get("total_experience_years", 0))
        }
        logger.log_event("success", f"Hybrid NER completed — Name: {final['name']}")
        return final

    logger.log_event("info", f"LM Studio unavailable — using traditional NER (Name: {traditional['name']})")
    return traditional



# ==================== EXPERIENCE LEVEL DETERMINATION ====================
def determine_experience_level(years_experience, logger):
    """Determine experience level based on years"""
    logger.log_event("info", f"Determining experience level for {years_experience} years")
    for level, (min_years, max_years) in EXPERIENCE_LEVELS.items():
        if min_years <= years_experience < max_years:
            logger.log_event("success", f"Experience level determined: {level}")
            return level
    logger.log_event("warning", "Defaulting to ENTRY_LEVEL experience")
    return "ENTRY_LEVEL"

# ==================== DOMAIN CLASSIFIER ====================
def classify_domain_with_lm_studio(resume_text, logger):
    """Classify resume domain using LM Studio"""
    logger.log_event("info", "Starting domain classification")
    try:
        prompt = f"""<system>
Classify the resume into one of these domains: {', '.join(DOMAINS)}.
Provide your response in XML format.
</system>
<examples>
<example>
<resume>Experience with Python, machine learning, data analysis, pandas, scikit-learn</resume>
<response><classification><domain>Data Science</domain></classification></response>
</example>
<example>
<resume>React, Vue.js, HTML, CSS, JavaScript frameworks</resume>
<response><classification><domain>Frontend</domain></classification></response>
</example>
<example>
<resume>Node.js, Python backend, REST APIs, databases</resume>
<response><classification><domain>Backend</domain></classification></response>
</example>
<example>
<resume>Docker, Kubernetes, AWS, CI/CD pipelines</resume>
<response><classification><domain>DevOps</domain></classification></response>
</example>
</examples>
<task>
Classify the following resume:
</task>
<resume>
{resume_text[:1000]}
</resume>
<response>"""
        response_text = call_lm_studio_model(prompt, max_tokens=100, temperature=0.1, stop=["</response>"])
        if response_text:
            import xml.etree.ElementTree as ET
            try:
                xml_response = "<response>" + response_text.strip() + "</response>"
                root = ET.fromstring(xml_response)
                classification = root.find('classification')
                if classification is not None:
                    domain_elem = classification.find('domain')
                    if domain_elem is not None and domain_elem.text:
                        domain = domain_elem.text.strip()
                        for d in DOMAINS:
                            if d.lower() == domain.lower():
                                logger.log_event("success", f"Domain classified as: {d}")
                                return d
            except ET.ParseError:
                pass
        logger.log_event("warning", f"Fallback to default domain: {DOMAINS[0]}")
        return DOMAINS[0]
    except Exception as e:
        logger.log_event("error", f"Domain classification failed: {str(e)}")
        return DOMAINS[0]

# ==================== PROFILE CREATOR ====================
def create_profile(entities, domain, logger):
    """Create candidate profile"""
    logger.log_event("info", "Creating candidate profile")
    profile = {
        "name": entities.get("name", "Unknown"),
        "domain": domain,
        "skills": entities.get("skills", []),
        "projects": entities.get("projects", []),
        "work_experience": entities.get("work_experience", []),
        "responsibilities": entities.get("responsibilities", []),
        "education": entities.get("education", []),
        "total_experience_years": entities.get("total_experience_years", 0)
    }
    logger.log_event("success", "Candidate profile created")
    return profile

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self, logger):
        self.sessions = {}
        self.logger = logger

    def create_session(self, user_id, domain, experience_level, profile):
        session_id = str(uuid.uuid4())[:8]
        expiry = datetime.now() + timedelta(minutes=60)
        self.sessions[session_id] = {
            "user_id": user_id,
            "domain": domain,
            "experience_level": experience_level,
            "profile": profile,
            "start_time": datetime.now(),
            "expiry": expiry,
            "responses": [],
            "pre_generated_questions": None,  # Will store all pre-generated questions
            "current_question_index": 0,  # Track which question we're on
            "interview_progress": {
                "current_section_index": 0,
                "current_question_in_section": 0,
                "questions_asked": 0,
                "technical_difficulty": "EASY",
                "performance_score": 0,
                "last_question_type": None,
                "previous_context": None,
                "asked_questions": set(), # Use a set for efficient lookup
                "asked_questions_list": [] # Use a list to provide recent questions to the prompt
            }
        }
        self.logger.log_event("success", f"Session created: {session_id}")
        return session_id
    
    def set_pre_generated_questions(self, session_id, questions):
        """Store pre-generated questions for the session"""
        if self.is_valid(session_id):
            self.sessions[session_id]["pre_generated_questions"] = questions
            self.logger.log_event("success", f"Pre-generated questions stored for session {session_id}")
            return True
        return False
    
    def get_next_question(self, session_id):
        """Get the next pre-generated question"""
        if not self.is_valid(session_id):
            return None
        
        questions = self.sessions[session_id].get("pre_generated_questions")
        if not questions:
            return None
        
        # Flatten all questions from all sections into a single list
        flat_questions = []
        for section_data in questions:
            for q in section_data["questions"]:
                flat_questions.append(q)
        
        current_index = self.sessions[session_id].get("current_question_index", 0)
        if current_index < len(flat_questions):
            question_data = flat_questions[current_index]
            self.sessions[session_id]["current_question_index"] = current_index + 1
            return question_data
        
        return None

    def is_valid(self, session_id):
        valid = session_id in self.sessions and datetime.now() < self.sessions[session_id]["expiry"]
        if not valid:
            self.logger.log_event("warning", f"Invalid or expired session: {session_id}")
        return valid

    def add_response(self, session_id, question, response, section_name):
        if self.is_valid(session_id):
            self.sessions[session_id]["responses"].append({
                "question": question,
                "response": response,
                "section": section_name,
                "timestamp": datetime.now().isoformat()
            })
            progress = self.sessions[session_id]["interview_progress"]
            progress["questions_asked"] += 1
            
            progress["asked_questions"].add(question)
            progress["asked_questions_list"].append(question)
            if len(progress["asked_questions_list"]) > 10:
                progress["asked_questions_list"] = progress["asked_questions_list"][-10:]
            
            self.logger.log_event("info", f"Response added for session {session_id}")
            return True
        return False

    def update_difficulty_and_context(self, session_id, performance_indicator, current_section_name, question_type):
        if self.is_valid(session_id):
            progress = self.sessions[session_id]["interview_progress"]
            progress["last_question_type"] = question_type
            if current_section_name == "Technical Deep-Dive":
                current_difficulty = progress["technical_difficulty"]
                difficulties = ["EASY", "MEDIUM", "HARD"]
                current_index = difficulties.index(current_difficulty)
                if performance_indicator == "CORRECT" and current_index < len(difficulties) - 1:
                    progress["technical_difficulty"] = difficulties[current_index + 1]
                    self.logger.log_event("info", f"Technical difficulty increased to {difficulties[current_index + 1]}")
                elif performance_indicator == "INCORRECT" and current_index > 0:
                    progress["technical_difficulty"] = difficulties[current_index - 1]
                    self.logger.log_event("info", f"Technical difficulty decreased to {difficulties[current_index - 1]}")

    def get_session_data(self, session_id):
        return self.sessions.get(session_id, {})

# ==================== DYNAMIC QUESTION GENERATOR ====================
def generate_dynamic_question(section_info, domain, experience_level, profile, session_manager, session_id, logger, previous_response=None, follow_up=False):
    """Generate a question dynamically based on current context and previous responses"""
    session_data = session_manager.get_session_data(session_id)
    progress = session_data["interview_progress"]
    section_name = section_info["name"]
    
    recent_responses = session_data.get("responses", [])[-3:]
    conversation_context = "\n".join([f"Q: {r['question']}\nA: {r['response']}" for r in recent_responses])
    
    if "asked_resume_items" not in session_data:
        session_data["asked_resume_items"] = {"projects": set(), "skills": set()}
    asked_items = session_data["asked_resume_items"]
    asked_projects = asked_items.get("projects", set())
    asked_skills = asked_items.get("skills", set())
    
    current_q_index = progress.get("current_question_in_section", 0)
    
    if follow_up:
        question_type = "follow_up"
        context_prompt = f"""You need to ask a follow-up question to clarify or probe deeper into the candidate's previous response. The previous question and response were:
Q: {previous_response.get('question', 'N/A')}
A: {previous_response.get('response', 'N/A')}
Ask a specific follow-up question that helps clarify or expand on their answer. Be conversational and natural."""
    elif section_name == "Greeting & Introduction":
        question_type = "introduction"
        context_prompt = f"Start the interview with a warm greeting. Ask them to introduce themselves and tell you about their background in {domain}."
    elif section_name == "Resume-Driven Questions":
        question_type = "resume_descriptive" if current_q_index == 0 else "resume_analytical"
        
        if current_q_index == 0:
            projects = profile.get('projects', [])
            skills = profile.get('skills', [])
            
            unasked_projects = [p for p in projects if p not in asked_projects]
            
            unasked_skills = [s for s in skills if s not in asked_skills]
            
            focus_item = None
            focus_type = None
            
            if unasked_projects:
                focus_item = unasked_projects[0]
                focus_type = "project"
                context_prompt = f"Ask about their project: {focus_item}. Ask them to describe what they built and the technologies used. Be specific about this project."
            elif unasked_skills:
                focus_item = unasked_skills[0]
                focus_type = "skill"
                context_prompt = f"Ask about their skill/technology: {focus_item}. Ask them how they've used it and what they've accomplished with it. Be specific about this skill."
            elif projects:
                focus_item = projects[0]
                focus_type = "project"
                context_prompt = f"Ask about their project experience in general, or a different aspect of their projects. Don't repeat previous questions."
            elif skills:
                focus_item = skills[0]
                focus_type = "skill"
                context_prompt = f"Ask about their technical skills in general, or a different aspect of their skill set. Don't repeat previous questions."
            else:
                context_prompt = "Ask about their resume and experience. Be specific and don't repeat previous questions."
            
            if focus_item and focus_type:
                asked_items[f"{focus_type}s"].add(focus_item)
        else:
            recent_responses_in_section = [r for r in session_data.get("responses", []) 
                                        if r.get("section") == "Resume-Driven Questions"][-1:]
            if recent_responses_in_section:
                context_prompt = "Based on their previous response about their resume, ask a deeper analytical question about a DIFFERENT aspect - challenges, decisions, outcomes, or lessons learned. Don't ask the same type of question."
            else:
                context_prompt = "Ask a deeper analytical question about their resume experience. Focus on different aspects than previously discussed."
    elif section_name == "Technical Deep-Dive":
        difficulty = progress.get("technical_difficulty", "EASY")
        difficulty_map = {"EASY": "definition", "MEDIUM": "application", "HARD": "scenario"}
        question_type = difficulty_map.get(difficulty, "definition")
        context_prompt = f"Ask a {difficulty.lower()} technical question about {domain}. Focus on {question_type} type questions."
    elif section_name == "Applied Problem-Solving":
        question_type = "applied_scenario"
        context_prompt = f"Present a realistic problem-solving scenario in {domain}. Make it practical and relevant to real-world work."
    elif section_name == "Behavioral & Soft Skills":
        question_type = "behavioral_star"
        context_prompt = "Ask a behavioral question using the STAR method. Focus on different soft skills (teamwork, leadership, problem-solving, adaptability)."
    elif section_name == "Candidate's Questions":
        question_type = "candidate_questions"
        context_prompt = "Invite the candidate to ask their own questions about the role, team, or company."
    elif section_name == "Closing":
        question_type = "closing"
        context_prompt = "Thank the candidate sincerely and provide closing remarks. Keep it warm and professional."
    else:
        question_type = "general"
        context_prompt = "Ask a relevant interview question for this section."
    
    
    if section_name == "Resume-Driven Questions":
        already_asked_block = (
            f"<already_asked_resume_items>\n"
            f"Already asked about projects: {', '.join(list(asked_projects)[:5]) if asked_projects else 'None yet'}\n"
            f"Already asked about skills: {', '.join(list(asked_skills)[:5]) if asked_skills else 'None yet'}\n"
            f"</already_asked_resume_items>"
        )
    else:
        already_asked_block = ""

    prompt = f"""<system>
    You are LoRa, a professional and experienced technical interviewer conducting a live interview.
    You are having a natural conversation with the candidate. Be conversational, human-like, and engaging.
    Key Guidelines:
    1. Ask ONE clear, focused question at a time
    2. Be natural and conversational - avoid robotic phrasing
    3. Reference previous answers when appropriate for continuity
    4. Show genuine interest in their responses
    5. Respond ONLY with the question text - no extra commentary
    </system>

    <candidate_info>
    Name: {profile.get('name', 'Candidate')}
    Domain: {domain}
    Experience Level: {experience_level}
    Years of Experience: {profile.get('total_experience_years', 'N/A')}
    Key Skills: {', '.join(profile.get('skills', [])[:5])}
    </candidate_info>

    <current_section>
    Section: {section_name}
    Description: {section_info.get('description', '')}
    Question Type: {question_type}
    </current_section>

    <conversation_history>
    {conversation_context if conversation_context else "This is the beginning of the interview."}
    </conversation_history>

    {already_asked_block}

    <task>
    {context_prompt}

    Generate a single, natural interview question. Make it specific to the candidate's background when possible.
    {"IMPORTANT: Do NOT ask about projects or skills that have already been asked about (listed above). Ask about different items instead." if section_name == "Resume-Driven Questions" and (asked_projects or asked_skills) else ""}
    </task>

    <question>
    """

    
    response_text = call_lm_studio_model(prompt, max_tokens=200, temperature=0.7, stop=["</question>"])
    if response_text and len(response_text.strip()) > 10:
        question = response_text.strip()
        logger.log_event("success", f"Dynamic question generated for {section_name}: {question[:50]}...")
        return question
    
    # Fallback
    fallbacks = {
        "introduction": f"Hi {profile.get('name', 'there')}, thanks for joining us today! To start, could you tell me a bit about yourself and what draws you to {domain}?",
        "resume_descriptive": "I'd love to hear more about one of your projects. Can you walk me through what you built?",
        "resume_analytical": "That's interesting! What was the most challenging part of that project?",
        "definition": f"Let's dive into some technical basics. Can you explain what [a key {domain} concept] is?",
        "application": f"How would you apply [a {domain} concept] in a real-world scenario?",
        "scenario": f"Imagine you're working on a {domain} problem where [scenario]. How would you approach it?",
        "applied_scenario": f"Here's a realistic scenario: [problem]. How would you solve this?",
        "behavioral_star": "Tell me about a time you had to overcome a significant challenge at work.",
        "candidate_questions": "We've covered quite a bit! What questions do you have for me?",
        "closing": f"Thank you so much for your time today, {profile.get('name', 'there')}. We'll be in touch soon!",
        "follow_up": "Can you tell me more about that?",
    }
    return fallbacks.get(question_type, "Could you elaborate on that?")

def evaluate_response_for_followup(question, response, logger):
    """Evaluate if a response needs clarification or follow-up"""
    prompt = f"""<system>
You are an interviewer evaluating if a candidate's response needs clarification or a follow-up question.
Respond with ONLY one word: FOLLOWUP or CONTINUE
</system>
<question>{question}</question>
<response>{response}</response>
<evaluation>
Respond with:
- FOLLOWUP if the answer is vague, incomplete, too brief, or needs clarification
- CONTINUE if the answer is satisfactory and complete
</evaluation>
<decision>"""
    
    result = call_lm_studio_model(prompt, max_tokens=10, temperature=0.3, stop=["\n"])
    if result and "FOLLOWUP" in result.upper():
        logger.log_event("info", "Response needs follow-up")
        return True
    logger.log_event("info", "Response is satisfactory, can continue")
    return False

def generate_followup_question(previous_question, previous_response, domain, logger):
    """Generate a follow-up question based on previous response"""
    prompt = f"""<system>
You are an interviewer asking a follow-up question to clarify or probe deeper.
Be natural and conversational. Ask for specific details, examples, or clarification.
</system>
<previous_question>{previous_question}</previous_question>
<previous_response>{previous_response}</previous_response>
<task>
Generate a single, specific follow-up question that helps clarify or expand on their answer.
Be natural and conversational - this should feel like a real conversation.
</task>
<follow_up_question>"""
    
    response_text = call_lm_studio_model(prompt, max_tokens=150, temperature=0.7, stop=["</follow_up_question>"])
    if response_text and len(response_text.strip()) > 10:
        return response_text.strip()
    return "Can you tell me more about that?"

# ==================== PRE-GENERATE ALL QUESTIONS (DEPRECATED - KEPT FOR REFERENCE) ====================
def generate_all_questions(domain, experience_level, profile, logger):
    """Generate all interview questions upfront before the interview starts"""
    logger.log_event("info", "Pre-generating all interview questions")
    all_questions = []
    previously_generated = []  # Track all previously generated questions to avoid repetition
    
    for section_index, section_info in enumerate(INTERVIEW_FLOW):
        section_name = section_info["name"]
        num_questions = section_info["question_count"]
        section_questions = []
        
        # Determine question types for this section
        question_types_map = {
            "Greeting & Introduction": "introduction",
            "Resume-Driven Questions": ["resume_descriptive", "resume_analytical"],
            "Technical Deep-Dive": ["definition", "application", "scenario"],
            "Applied Problem-Solving": "applied_scenario",
            "Behavioral & Soft Skills": "behavioral_star",
            "Candidate's Questions": "candidate_questions",
            "Closing": "closing"
        }
        
        for q_in_section in range(num_questions):
            # Determine question type
            if section_name == "Resume-Driven Questions":
                question_type = "resume_descriptive" if q_in_section == 0 else "resume_analytical"
            elif section_name == "Technical Deep-Dive":
                # Progressively increase difficulty: EASY -> MEDIUM -> HARD
                difficulties = ["EASY", "MEDIUM", "HARD"]
                difficulty = difficulties[min(q_in_section, len(difficulties) - 1)]
                difficulty_map = {"EASY": "definition", "MEDIUM": "application", "HARD": "scenario"}
                question_type = difficulty_map.get(difficulty, "definition")
            elif section_name == "Applied Problem-Solving":
                question_type = "applied_scenario"
            elif section_name == "Behavioral & Soft Skills":
                question_type = "behavioral_star"
            elif section_name == "Candidate's Questions":
                question_type = "candidate_questions"
            elif section_name == "Closing":
                question_type = "closing"
            else:  # Greeting & Introduction
                question_type = "introduction"
            
            logger.log_event("info", f"Generating question {q_in_section + 1}/{num_questions} for section: {section_name} (Type: {question_type})")
            
            # Build prompt with all previously generated questions to avoid repetition
            previously_generated_text = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(previously_generated)]) if previously_generated else "None yet"
            
            prompt = f"""<system>
You are LoRa, a professional and experienced technical interviewer for {domain} roles. Your goal is to conduct a realistic, engaging, and insightful interview.
Key Principles:
1. Be conversational and human-like. Avoid robotic phrasing.
2. Ask one clear, focused question at a time.
3. NEVER repeat a question verbatim or even similar to previously asked questions.
4. Each question must be unique and cover different aspects.
5. Respond ONLY with the question text.
Context Information:
- Candidate Name: {profile.get('name', 'the candidate')}
- Domain: {domain}
- Experience Level: {experience_level}
- Total Experience: {profile.get('total_experience_years', 'N/A')} years
- Key Skills: {', '.join(profile.get('skills', [])[:5])}
- Notable Projects: {', '.join(profile.get('projects', [])[:2])}
Current Section: {section_name}
Question Type Needed: {question_type}
CRITICAL: Previously Generated Questions (DO NOT REPEAT OR SIMILAR TO THESE):
{previously_generated_text}
</system>
<guidelines>
<introduction>
    - Warmly greet the candidate by name.
    - Ask them to introduce themselves and highlight experiences relevant to {domain}.
    - Keep it open-ended and conversational.
    Example: "Hi {profile.get('name', 'Candidate')}, thanks for joining us today. To start, could you walk me through your background and tell me what draws you to {domain}?"
</introduction>
<resume_descriptive>
    - Pick a SPECIFIC, DIFFERENT skill or project from their resume that hasn't been asked about yet.
    - Ask them to describe it.
    - Be specific about which project/skill you're asking about.
    Example: "I see you worked on [Specific Project Name]. Can you tell me more about what you built and what technologies you used?"
</resume_descriptive>
<resume_analytical>
    - Ask about a DIFFERENT aspect of their resume or projects.
    - Probe deeper about challenges, decisions, or outcomes.
    - DO NOT repeat themes from previous questions.
    Example: "I noticed [specific skill/experience]. Can you walk me through a time when you applied that and what challenges you faced?"
</resume_analytical>
<definition>
    - Ask for the meaning or explanation of a fundamental but DIFFERENT concept in {domain}.
    - Choose a concept not yet covered.
    Example: "To start with some basics, can you explain what [a key concept in {domain}] is?"
</definition>
<application>
    - Ask how a DIFFERENT concept is used in practice.
    - Vary the scenario or use case.
    Example: "How would you apply [different concept] when designing [a relevant system/component]?"
</application>
<scenario>
    - Present a more complex, open-ended problem that's DIFFERENT from previous scenarios.
    - Focus on different aspects of {domain}.
    Example: "Imagine you're tasked with [a different complex task in {domain}]. How would you approach breaking it down and solving it?"
</scenario>
<applied_scenario>
    - Give a realistic, practical problem they might face on the job.
    - Use a DIFFERENT scenario than any previous question.
    Example: "Let's say you're working on [a realistic scenario in {domain}] and you encounter [specific issue]. How would you diagnose and resolve this?"
</applied_scenario>
<behavioral_star>
    - Use the STAR method (Situation, Task, Action, Result).
    - Focus on DIFFERENT soft skills: teamwork, leadership, problem-solving, adaptability, etc.
    - Ask about a different behavioral aspect.
    Example: "Tell me about a time you had to [different situation]. What was the situation, and how did you handle it?"
</behavioral_star>
<candidate_questions>
    - Invite the candidate to ask their own questions.
    - Show genuine interest in their curiosity.
    Example: "We've covered a lot of ground. What questions do you have for me about the role, the team, or what we do here?"
</candidate_questions>
<closing>
    - Thank the candidate sincerely by name.
    - Briefly outline next steps if known.
    - End on a positive note.
    Example: "Thanks so much for your time today, {profile.get('name', 'Candidate')}. It was great learning about your experience. We'll be reviewing all candidates and will be in touch soon. Do you have any final thoughts?"
</closing>
</guidelines>
<task>
Generate a SINGLE, UNIQUE interview question that:
1. Fits the current section and question type
2. Is COMPLETELY DIFFERENT from all previously generated questions listed above
3. Does NOT repeat themes, topics, or phrasing from previous questions
4. Is specific, natural, and engaging
5. If asking about resume items, choose DIFFERENT projects/skills than previously asked
</task>
<question>"""
            
            response_text = call_lm_studio_model(prompt, max_tokens=250, temperature=0.8, stop=["</question>"])
            if response_text and len(response_text.strip()) > 10:
                question = response_text.strip()
                question_data = {
                    "section_name": section_name,
                    "section_index": section_index,
                    "question_index": q_in_section,
                    "question_type": question_type,
                    "question": question
                }
                section_questions.append(question_data)
                previously_generated.append(question_data)
                logger.log_event("success", f"Question generated for {section_name} Q{q_in_section + 1}: {question[:50]}...")
            else:
                # Fallback question
                logger.log_event("warning", f"Failed to generate question, using fallback for {section_name}")
                fallbacks = {
                    "introduction": f"Hi {profile.get('name', 'Candidate')}, could you please introduce yourself and tell me what interests you about {domain}?",
                    "resume_descriptive": f"Can you tell me about a specific project from your resume that you're particularly proud of?",
                    "resume_analytical": f"What was the most challenging aspect of your work experience and how did you overcome it?",
                    "definition": f"What is an important concept in {domain} that you find essential?",
                    "application": f"How would you apply a key {domain} principle in a real-world project?",
                    "scenario": f"Imagine you face a complex problem in {domain}. How would you approach solving it?",
                    "applied_scenario": f"Here's a realistic {domain} challenge: [specific scenario]. How would you go about resolving it?",
                    "behavioral_star": f"Tell me about a time you had to work through a significant technical or team challenge.",
                    "candidate_questions": f"What questions do you have for me about the role or our team?",
                    "closing": f"Thank you for your time today, {profile.get('name', 'Candidate')}. We'll be in touch soon. Any last questions?"
                }
                question = fallbacks.get(question_type, "Could you elaborate on that?")
                question_data = {
                    "section_name": section_name,
                    "section_index": section_index,
                    "question_index": q_in_section,
                    "question_type": question_type,
                    "question": question
                }
                section_questions.append(question_data)
                previously_generated.append(question_data)
        
        all_questions.append({
            "section": section_info,
            "questions": section_questions
        })
    
    logger.log_event("success", f"Successfully pre-generated {sum(len(s['questions']) for s in all_questions)} questions")
    return all_questions

# ==================== MASTER QUESTION GENERATOR ====================
def generate_question_master_prompt(domain, section_info, experience_level, session_manager, session_id, logger):
    """Unified, master prompt for generating interview questions"""
    session_data = session_manager.get_session_data(session_id)
    progress = session_data["interview_progress"]
    section_name = section_info["name"]
    question_types = ["introduction", "resume_descriptive", "resume_analytical", "definition",
                    "application", "scenario", "applied_scenario", "behavioral_star",
                    "candidate_questions", "closing"]

    # Determine current question type based on section and position
    current_q_index_in_section = progress["current_question_in_section"]
    if section_name == "Resume-Driven Questions":
        question_type = "resume_descriptive" if current_q_index_in_section == 0 else "resume_analytical"
    elif section_name == "Technical Deep-Dive":
        difficulty_map = {"EASY": "definition", "MEDIUM": "application", "HARD": "scenario"}
        question_type = difficulty_map.get(progress["technical_difficulty"], "definition")
    elif section_name == "Applied Problem-Solving":
        question_type = "applied_scenario"
    elif section_name == "Behavioral & Soft Skills":
        question_type = "behavioral_star"
    elif section_name == "Candidate's Questions":
        question_type = "candidate_questions"
    elif section_name == "Closing":
        question_type = "closing"
    else:  # Greeting & Introduction
        question_type = "introduction"

    # Avoid repetition by checking last question type
    last_question_type = progress.get("last_question_type")
    if last_question_type == question_type:
        # Cycle to next type if possible, or use a different approach
        current_index = question_types.index(question_type)
        question_type = question_types[(current_index + 1) % len(question_types)]

    technical_difficulty = progress.get("technical_difficulty", "EASY")
    previous_context = progress.get("previous_context")
    profile = session_data.get("profile", {})
    # Use the list of recently asked questions
    asked_questions_list = progress.get("asked_questions_list", [])
    
    logger.log_event("info", f"Generating {question_type} question for {section_name} (Difficulty: {technical_difficulty})")

    # Master Prompt
    prompt = f"""<system>
You are LoRa, a professional and experienced technical interviewer for {domain} roles. Your goal is to conduct a realistic, engaging, and insightful interview. You adapt your questions based on the candidate's experience level ({experience_level}) and previous responses.
Key Principles:
1.  Be conversational and human-like. Avoid robotic phrasing.
2.  Ask one clear, focused question at a time.
3.  Build upon the candidate's previous answers when appropriate.
4.  Vary your question types to keep the interview dynamic.
5.  For technical sections, adjust difficulty based on performance.
6.  NEVER repeat a question verbatim. If you must revisit a topic, rephrase it significantly or ask about a different aspect.
7.  Respond ONLY with the question text.
Context Information:
- Candidate Name: {profile.get('name', 'the candidate')}
- Domain: {domain}
- Experience Level: {experience_level}
- Total Experience: {profile.get('total_experience_years', 'N/A')} years
- Key Skills: {', '.join(profile.get('skills', [])[:5])}
- Notable Projects: {', '.join(profile.get('projects', [])[:2])}
Interview Flow Stage: {section_name}
Question Type Needed: {question_type}
Current Technical Difficulty: {technical_difficulty}
Previously Asked Questions (to avoid): {', '.join(asked_questions_list[-5:]) if asked_questions_list else 'None yet'}
</system>
<guidelines>
<introduction>
    - Warmly greet the candidate by name.
    - Ask them to introduce themselves and highlight experiences relevant to {domain}.
    - Keep it open-ended and conversational.
    Example: "Hi {profile.get('name', 'Candidate')}, thanks for joining us today. To start, could you walk me through your background and tell me what draws you to {domain}?"
</introduction>
<resume_descriptive>
    - Pick a specific skill or project from their resume.
    - Ask them to describe it.
    Example: "I see you worked on [Project/Skill]. Can you tell me more about what you built and what technologies you used?"
</resume_descriptive>
<resume_analytical>
    - Based on their previous answer, probe deeper.
    - Ask about challenges, decisions, or outcomes.
    Example: "That's interesting. What was the most technically challenging part of [that project], and how did you overcome it?"
</resume_analytical>
<definition>
    - Ask for the meaning or explanation of a fundamental concept in {domain}.
    Example: "To start with some basics, can you explain what [a key concept in {domain}] is?"
</definition>
<application>
    - Ask how a concept is used in practice or in a specific scenario.
    Example: "How would you apply [concept] when designing [a relevant system/component]?"
</application>
<scenario>
    - Present a more complex, open-ended problem.
    Example: "Imagine you're tasked with [a complex task in {domain}]. How would you approach breaking it down and solving it?"
</scenario>
<applied_scenario>
    - Give a realistic, practical problem they might face on the job.
    Example: "Let's say you're working on [a realistic scenario in {domain}] and you encounter [specific issue]. How would you diagnose and resolve this?"
</applied_scenario>
<behavioral_star>
    - Use the STAR method (Situation, Task, Action, Result).
    - Focus on teamwork, leadership, problem-solving, adaptability.
    Example: "Tell me about a time you had to quickly learn a new technology for a project. What was the situation, and how did you ensure you could deliver?"
</behavioral_star>
<candidate_questions>
    - Invite the candidate to ask their own questions.
    - Show genuine interest in their curiosity.
    Example: "We've covered a lot of ground. What questions do you have for me about the role, the team, or what we do here?"
</candidate_questions>
<closing>
    - Thank the candidate sincerely by name.
    - Briefly outline next steps if known.
    - End on a positive note.
    Example: "Thanks so much for your time today, {profile.get('name', 'Candidate')}. It was great learning about your experience. We'll be reviewing all candidates and will be in touch soon. Do you have any final thoughts?"
</closing>
</guidelines>
<task>
Generate a single, realistic interview question that fits the current context and guidelines. Be specific, natural, and engaging. Make sure it is significantly different from the previously asked questions: {', '.join(asked_questions_list[-3:]) if len(asked_questions_list) >= 3 else 'None yet'}.
</task>
<question>"""

    # Add previous context if available and relevant
    if previous_context and question_type in ["resume_analytical", "applied_scenario"]:
        prompt += f"\n<previous_context>\n{previous_context.get('response', 'N/A')}\n</previous_context>"

    response_text = call_lm_studio_model(prompt, max_tokens=250, temperature=0.7, stop=["</question>"])
    if response_text and len(response_text.strip()) > 10:
        question = response_text.strip()
        logger.log_event("success", f"Question generated: {question}")
        return question

    logger.log_event("warning", "Failed to generate unique question, using fallback.")
    fallbacks = {
        "introduction": f"Hi {profile.get('name', 'Candidate')}, could you please introduce yourself and tell me what interests you about {domain}?",
        "resume_descriptive": "Can you tell me about a project or skill from your resume that you're particularly proud of?",
        "resume_analytical": "Building on that, what was the most challenging part and how did you tackle it?",
        "definition": f"What is a key concept in {domain} that you find essential?",
        "application": f"How would you use a core {domain} principle in a real project?",
        "scenario": f"Imagine you face a common problem in {domain}. How would you approach solving it?",
        "applied_scenario": f"Here's a realistic {domain} challenge. How would you go about resolving it?",
        "behavioral_star": "Tell me about a time you had to work through a significant technical or team challenge.",
        "candidate_questions": "What would you like to know about the role or our team?",
        "closing": f"Thank you for your time today, {profile.get('name', 'Candidate')}. We'll be in touch soon. Any last questions?"
    }
    question = fallbacks.get(question_type, "Could you elaborate on that?")
    return question

# ==================== EVALUATOR ====================
def evaluate_response_simple(question, response, section_name, logger=None):
    """Simple evaluation for difficulty adjustment"""
    if logger:
        logger.log_event("info", f"Starting response evaluation for section: {section_name}")
    try:
        prompt = f"""<system>
You are a hiring manager reviewing an interview response. Based on the question and answer, quickly assess the response's correctness.
Respond with ONLY ONE of these exact words:
CORRECT
PARTIAL
INCORRECT
</system>
<task>
Evaluate the candidate's response for difficulty adjustment.
</task>
<question>{question}</question>
<answer>{response}</answer>
<evaluation>"""
        response_text = call_lm_studio_model(prompt, max_tokens=20, temperature=0.3, stop=["\n"])
        if response_text:
            indicator = response_text.strip().upper()
            if indicator in ["CORRECT", "PARTIAL", "INCORRECT"]:
                if logger:
                    logger.log_event("success", f"Evaluation completed: {indicator}")
                return indicator
        if logger:
            logger.log_event("warning", "Evaluation failed, defaulting to PARTIAL.")
        return "PARTIAL"
    except Exception as e:
        if logger:
            logger.log_event("error", f"Response evaluation failed: {str(e)}")
        return "PARTIAL"

# ==================== POST-INTERVIEW ANALYSIS GENERATOR ====================
async def generate_analysis_section_async(session, section_name, prompt_template, context_data, logger):
    """Generate a single section of the analysis asynchronously"""
    try:
        prompt = prompt_template.format(**context_data)
        result = await call_lm_studio_model_async(session, prompt, max_tokens=800, temperature=0.6)
        if result:
            logger.log_event("success", f"Analysis section '{section_name}' generated successfully")
            return section_name, result.strip()
        else:
            logger.log_event("warning", f"Analysis section '{section_name}' returned no result")
            return section_name, None
    except Exception as e:
        logger.log_event("error", f"Analysis section '{section_name}' generation failed: {str(e)}")
        return section_name, None

def generate_post_interview_analysis(session_data, profile, logger):
    """Generate a detailed post-interview analysis using LM Studio with async parallelization."""
    logger.log_event("info", "Generating post-interview analysis with LM Studio (async parallel)")
    
    try:
        responses = session_data.get("responses", [])
        if not responses:
            logger.log_event("warning", "No responses found in session data")
            return "# Post-Interview Analysis\n\nNo interview responses available for analysis."
        
        # Prepare context data
        responses_text = "\n".join([f"Q{i+1}: {r['question']}\nA: {r['response']}\nSection: {r['section']}\n---" 
                                    for i, r in enumerate(responses)])
        
        # Separate responses by section type for better analysis
        technical_responses = [r for r in responses if r.get('section') in ['Technical Deep-Dive', 'Applied Problem-Solving']]
        behavioral_responses = [r for r in responses if r.get('section') in ['Behavioral & Soft Skills', 'Greeting & Introduction']]
        resume_responses = [r for r in responses if r.get('section') == 'Resume-Driven Questions']
        
        technical_text = "\n".join([f"Q: {r['question']}\nA: {r['response']}\n---" for r in technical_responses]) if technical_responses else "No technical responses"
        behavioral_text = "\n".join([f"Q: {r['question']}\nA: {r['response']}\n---" for r in behavioral_responses]) if behavioral_responses else "No behavioral responses"
        
        context_base = {
            'candidate_name': profile.get('name', 'Unknown'),
            'domain': session_data.get('domain', 'N/A'),
            'experience_level': session_data.get('experience_level', 'N/A'),
            'years_experience': profile.get('total_experience_years', 'N/A'),
            'key_skills': ', '.join(profile.get('skills', [])[:10]),
            'notable_projects': ', '.join(profile.get('projects', [])[:5]),
            'all_responses': responses_text,
            'technical_responses': technical_text,
            'behavioral_responses': behavioral_text,
            'resume_responses': "\n".join([f"Q: {r['question']}\nA: {r['response']}\n---" for r in resume_responses]) if resume_responses else "No resume responses"
        }
        
        # Define prompts for each section (to be generated in parallel)
        section_prompts = {
            'overall': """<system>
You are a senior hiring manager providing an overall assessment of a candidate.
Be concise, professional, and fair. Provide a clear overall impression.
</system>
<task>
Provide an overall impression of the candidate based on the interview.
</task>
<candidate_profile>
Name: {candidate_name}
Domain: {domain}
Experience Level: {experience_level}
Years of Experience: {years_experience}
</candidate_profile>
<interview_responses>
{all_responses}
</interview_responses>
<task>
Write a concise overall impression (2-3 paragraphs) of the candidate's interview performance. Be specific and reference their responses.
</task>
<overall_impression>""",
            
            'strengths': """<system>
You are a senior hiring manager analyzing candidate strengths.
Focus on concrete examples from their responses.
</system>
<task>
Identify and elaborate on the candidate's key strengths based on interview responses.
</task>
<interview_responses>
{all_responses}
</interview_responses>
<task>
Provide 3-5 specific strengths with brief explanations. Each strength should reference specific examples from their responses. Format as markdown bullet points.
</task>
<strengths>""",
            
            'improvements': """<system>
You are a senior hiring manager providing constructive feedback.
Be honest but constructive and fair.
</system>
<task>
Identify areas where the candidate could improve based on interview responses.
</task>
<interview_responses>
{all_responses}
</interview_responses>
<task>
Provide 2-4 specific areas for improvement with brief explanations. Be constructive and fair. Format as markdown bullet points.
</task>
<areas_for_improvement>""",
            
            'technical': """<system>
You are a technical hiring manager evaluating technical competency.
Focus on demonstrated knowledge, problem-solving, and technical depth.
</system>
<task>
Assess the candidate's technical competency based on technical interview sections.
</task>
<technical_responses>
{technical_responses}
</task>
<candidate_profile>
Domain: {domain}
Key Skills: {key_skills}
</candidate_profile>
<task>
Provide a detailed assessment of technical competency (2-3 paragraphs). Evaluate their understanding, problem-solving approach, and depth of knowledge. Reference specific technical responses.
</task>
<technical_assessment>""",
            
            'communication': """<system>
You are a hiring manager evaluating communication and soft skills.
Focus on clarity, confidence, problem-solving approach, and interpersonal skills.
</system>
<task>
Assess the candidate's communication and soft skills based on behavioral and introduction sections.
</task>
<behavioral_responses>
{behavioral_responses}
</task>
<task>
Provide a detailed assessment of communication and soft skills (2-3 paragraphs). Evaluate clarity, confidence, problem-solving approach, and interpersonal abilities. Reference specific responses.
</task>
<communication_assessment>""",
            
            'recommendation': """<system>
You are a senior hiring manager making a final hiring recommendation.
Consider all aspects: technical skills, communication, experience level, and fit.
</system>
<task>
Provide a final hiring recommendation with justification.
</task>
<candidate_profile>
Name: {candidate_name}
Domain: {domain}
Experience Level: {experience_level}
Years of Experience: {years_experience}
</candidate_profile>
<interview_responses>
{all_responses}
</interview_responses>
<task>
Provide a clear recommendation: "Strong Hire", "Consider", "Reject", or "Needs More Info". Follow with 2-3 sentences justifying your recommendation based on the interview performance.
</task>
<recommendation>"""
        }
        
        # Generate all sections in parallel using async
        async def generate_all_sections():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    generate_analysis_section_async(session, name, prompt, context_base, logger)
                    for name, prompt in section_prompts.items()
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                sections = {}
                for result in results:
                    if isinstance(result, Exception):
                        logger.log_event("error", f"Exception in analysis generation: {str(result)}")
                        continue
                    section_name, content = result
                    if content:
                        sections[section_name] = content
                
                return sections
        
        try:
            sections = asyncio.run(generate_all_sections())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sections = loop.run_until_complete(generate_all_sections())
            loop.close()
        
        # Combine sections into final analysis
        if sections:
            analysis = f"# Post-Interview Analysis for {profile.get('name', 'Unknown')}\n\n"
            
            if 'overall' in sections:
                analysis += f"## Overall Impression\n\n{sections['overall']}\n\n"
            
            if 'strengths' in sections:
                analysis += f"## Strengths\n\n{sections['strengths']}\n\n"
            
            if 'improvements' in sections:
                analysis += f"## Areas for Improvement\n\n{sections['improvements']}\n\n"
            
            if 'technical' in sections:
                analysis += f"## Technical Competency\n\n{sections['technical']}\n\n"
            
            if 'communication' in sections:
                analysis += f"## Communication & Soft Skills\n\n{sections['communication']}\n\n"
            
            if 'recommendation' in sections:
                analysis += f"## Final Recommendation\n\n{sections['recommendation']}\n\n"
            
            logger.log_event("success", "Post-interview analysis generated successfully with all sections")
            return analysis.strip()
        else:
            logger.log_event("warning", "No analysis sections were generated successfully")
            # Fallback: use synchronous method for overall analysis
            fallback_prompt = f"""<system>
You are a senior hiring manager providing a final assessment of a candidate.
</system>
<candidate_profile>
Name: {profile.get('name', 'Unknown')}
Domain: {session_data.get('domain', 'N/A')}
Experience Level: {session_data.get('experience_level', 'N/A')}
</candidate_profile>
<interview_responses>
{responses_text[:3000]}
</interview_responses>
<task>
Provide a brief overall assessment and recommendation (2-3 paragraphs).
</task>
<assessment>"""
            fallback_result = call_lm_studio_model(fallback_prompt, max_tokens=1000, temperature=0.6)
            if fallback_result:
                return f"# Post-Interview Analysis for {profile.get('name', 'Unknown')}\n\n## Overall Assessment\n\n{fallback_result.strip()}\n"
            return "# Post-Interview Analysis\n\nAnalysis generation encountered an error. Please review the interview responses manually."
            
    except Exception as e:
        logger.log_event("error", f"Post-interview analysis generation failed: {str(e)}")
        import traceback
        logger.log_event("error", f"Traceback: {traceback.format_exc()}")
        return f"# Post-Interview Analysis\n\nAn error occurred while generating the analysis: {str(e)}"

# ==================== PDF REPORT GENERATOR ====================
def sanitize_text_for_pdf(text):
    """Replace Unicode characters that can't be encoded in latin-1 with ASCII equivalents"""
    if not text:
        return ""
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '\u2014': '--',  
        '\u2013': '-',   
        '\u2018': "'",   
        '\u2019': "'",   
        '\u201C': '"',   
        '\u201D': '"',   
        '\u2026': '...', 
        '\u00A0': ' ',   
        '\u2022': '*',   
        '\u00AE': '(R)', 
        '\u00A9': '(C)', 
    }
    result = text
    for unicode_char, ascii_char in replacements.items():
        result = result.replace(unicode_char, ascii_char)
    try:
        result.encode('latin-1')
    except UnicodeEncodeError:
        result = result.encode('latin-1', errors='replace').decode('latin-1')
    return result

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, sanitize_text_for_pdf('AI Technical Interview Report'), 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, sanitize_text_for_pdf(f'Page {self.page_no()}'), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, sanitize_text_for_pdf(title), 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        if isinstance(body, list):
            for item in body:
                self.multi_cell(0, 8, sanitize_text_for_pdf(f"  - {item}"))
        else:
            self.multi_cell(0, 8, sanitize_text_for_pdf(body))
        self.ln()

def generate_pdf_report(session_data, filename, analysis_text, logger):
    """Generate a comprehensive PDF report including the LLM analysis."""
    try:
        logger.log_event("info", f"Generating PDF report: {filename}")
        pdf = PDFReport()
        pdf.add_page()
        # Candidate Information
        pdf.chapter_title("Candidate Information")
        profile = session_data.get("profile", {})
        pdf.chapter_body([
            f"Name: {profile.get('name', 'N/A')}",
            f"Domain: {session_data.get('domain', 'N/A')}",
            f"Experience Level: {session_data.get('experience_level', 'N/A')}",
            f"Total Experience: {profile.get('total_experience_years', 'N/A')} years",
            f"Interview Date: {session_data.get('start_time', datetime.now().isoformat())}"
        ])
        # Skills & Experience
        pdf.chapter_title("Skills & Experience")
        pdf.chapter_body([
            f"Key Skills: {', '.join(profile.get('skills', [])[:10])}",
            f"Notable Projects: {', '.join(profile.get('projects', [])[:5])}",
            f"Work Experience: {len(profile.get('work_experience', []))} entries",
            f"Education: {', '.join(profile.get('education', [])[:2])}"
        ])
        # Interview Responses
        pdf.chapter_title("Interview Responses")
        responses = session_data.get("responses", [])
        for i, resp in enumerate(responses, 1):
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 8, sanitize_text_for_pdf(f"Q{i}: {resp.get('section', 'N/A')}"), 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, sanitize_text_for_pdf(f"Question: {resp.get('question', 'N/A')}"))
            pdf.multi_cell(0, 6, sanitize_text_for_pdf(f"Response: {resp.get('response', 'N/A')}"))
            pdf.ln(4)

        pdf.chapter_title("AI Generated Post-Interview Analysis")
        pdf.set_font('Arial', '', 10)
        for line in analysis_text.split('\n'):
            sanitized_line = sanitize_text_for_pdf(line)
            if line.startswith('# '):
                pdf.set_font('Arial', 'B', 12)
                pdf.multi_cell(0, 10, sanitized_line[2:]) # Remove '# ' prefix
                pdf.set_font('Arial', '', 10) # Reset font
            elif line.startswith('## '):
                pdf.set_font('Arial', 'B', 11)
                pdf.multi_cell(0, 9, sanitized_line[3:]) # Remove '## ' prefix
                pdf.set_font('Arial', '', 10) # Reset font
            elif line.startswith('### '):
                pdf.set_font('Arial', 'B', 10)
                pdf.multi_cell(0, 8, sanitized_line[4:]) # Remove '### ' prefix
                pdf.set_font('Arial', '', 10) # Reset font
            elif line.startswith('- '):
                pdf.multi_cell(0, 8, sanitize_text_for_pdf(f"  {line}")) # Add indentation for list items
            else:
                pdf.multi_cell(0, 8, sanitized_line)
        pdf.ln()

        # Save PDF
        pdf.output(filename)
        logger.log_event("success", f"PDF report generated successfully: {filename}")
        return filename
    except Exception as e:
        logger.log_event("error", f"PDF report generation failed: {str(e)}")
        return None

# ==================== UI COMPONENTS ====================
def show_welcome():
    console.clear()
    welcome_text = Text("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                         TRES-AI                              ║
    ║                                                              ║
    ║            Powered by LM Studio & Advanced NLP               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """, style="bold blue")
    console.print(welcome_text)
    console.print("")
    logging.info("Welcome screen displayed")

def show_interview_section_header(section_name, section_description="", question_num=0, total_questions=0):
    section_title = f"[bold magenta]{section_name}[/bold magenta]"
    if question_num > 0 and total_questions > 0:
        section_title += f" [cyan]({question_num}/{total_questions})[/cyan]"
    if section_description:
        section_panel = Panel(
            f"[italic]{section_description}[/italic]",
            title=section_title,
            border_style="magenta"
        )
    else:
        section_panel = Panel("", title=section_title, border_style="magenta")
    console.print(section_panel)
    logging.info(f"Section header displayed: {section_name}")

def show_interview_question(question):
    question_panel = Panel(
        f"[bold yellow]{question}[/bold yellow]",
        title=f"[bold cyan]Interview Question[/bold cyan]",
        border_style="blue"
    )
    console.print(question_panel)
    logging.info(f"Question displayed: {question[:50]}...")

def show_interview_progress(current, section):
    progress_text = f"[dim][cyan]Section:[/cyan] {section}[/dim]"
    console.print(progress_text)
    logging.info(f"Progress: Question {current} in section {section}")

def show_interview_complete():
    complete_panel = Panel(
        f"""Thank you for completing the technical interview!
Your responses have been recorded. A detailed analysis and reports
are being generated.""",
        title="[bold green]🎉 Interview Complete[/bold green]",
        border_style="green"
    )
    console.print(complete_panel)
    logging.info("Interview completion message displayed")

def show_extracted_entities(entities, domain, experience_level):
    """Display extracted entities from resume analysis"""
    console.print("\n[bold cyan]📋 Extracted Resume Information:[/bold cyan]\n")
    
    # Create a table for better formatting
    info_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    info_table.add_column("Category", style="cyan", width=20)
    info_table.add_column("Details", style="yellow", width=60)
    
    # Name
    name = entities.get("name", "Not identified")
    info_table.add_row("👤 Name", name or "Not identified")
    
    # Domain and Experience
    info_table.add_row("🎯 Domain", domain)
    info_table.add_row("📊 Experience Level", experience_level)
    exp_years = entities.get("total_experience_years", 0)
    info_table.add_row("⏱️  Total Years", f"{exp_years} years" if exp_years > 0 else "Not specified")
    
    # Skills
    skills = entities.get("skills", [])
    skills_display = ', '.join(skills[:MAX_SKILLS_DISPLAY]) if skills else "None identified"
    if len(skills) > MAX_SKILLS_DISPLAY:
        skills_display += f" (+{len(skills) - MAX_SKILLS_DISPLAY} more)"
    info_table.add_row("🛠️  Skills", skills_display)
    
    # Projects
    projects = entities.get("projects", [])
    if projects:
        projects_display = '\n'.join([f"  • {p}" for p in projects[:MAX_PROJECTS_DISPLAY]])
        if len(projects) > MAX_PROJECTS_DISPLAY:
            projects_display += f"\n  ... (+{len(projects) - MAX_PROJECTS_DISPLAY} more)"
    else:
        projects_display = "None identified"
    info_table.add_row("💼 Projects", projects_display)
    
    # Work Experience
    work_exp = entities.get("work_experience", [])
    if work_exp:
        work_display = '\n'.join([f"  • {exp}" for exp in work_exp[:5]])
        if len(work_exp) > 5:
            work_display += f"\n  ... (+{len(work_exp) - 5} more)"
    else:
        work_display = "None identified"
    info_table.add_row("🏢 Work Experience", work_display)
    
    # Education
    education = entities.get("education", [])
    edu_display = '\n'.join([f"  • {edu}" for edu in education[:3]]) if education else "None identified"
    info_table.add_row("🎓 Education", edu_display)
    
    # Responsibilities
    responsibilities = entities.get("responsibilities", [])
    resp_display = '\n'.join([f"  • {resp}" for resp in responsibilities[:3]]) if responsibilities else "None identified"
    info_table.add_row("⭐ Responsibilities", resp_display)
    
    console.print(info_table)
    console.print("")
    logging.info("Extracted entities displayed to user")

# ==================== MAIN INTERVIEW FLOW ====================
def run_interview(pdf_path):
    show_welcome()
    logging.info(f"Starting interview process for {pdf_path}")

    # Ask for user's name at the start
    console.print("\n[bold cyan]Before we begin, let's get to know you better![/bold cyan]\n")
    user_name = Prompt.ask("[bold yellow]What is your full name?[/bold yellow]", default="")
    if user_name and user_name.strip():
        user_name = user_name.strip()
        console.print(f"[bold green]Nice to meet you, {user_name}![/bold green]\n")
        logging.info(f"User provided name: {user_name}")
    else:
        console.print("[bold yellow]No name provided, will use extracted name from resume if available.[/bold yellow]\n")
        logging.info("User did not provide name")
    
    console.print("[bold blue]📡 Connecting to LM Studio...[/bold blue]")
    logging.info("Testing LM Studio connection")
    test_prompt = "Hello, this is a connection test."
    test_response = call_lm_studio_model(test_prompt, max_tokens=20, temperature=0.1)
    if not test_response:
        console.print("[bold red]❌ Failed to connect to LM Studio.[/bold red]")
        logging.error("Failed to connect to LM Studio")
        return
    console.print("[bold green]✓ Connected to LM Studio successfully![/bold green]")
    logging.info("Connected to LM Studio successfully")

    session_id = str(uuid.uuid4())[:8]
    base_dir = create_session_directories(session_id)
    md_logger = MarkdownLogger(session_id, base_dir)
    md_logger.log_event("info", f"Starting interview session for: {pdf_path}")
    if user_name:
        md_logger.log_event("info", f"User provided name: {user_name}")

    console.print("[bold blue]📁 Analyzing your resume...[/bold blue]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Extracting text from PDF...", total=None)
        resume_text = extract_text_from_pdf(pdf_path, md_logger)
        if not resume_text:
            md_logger.log_event("error", "Failed to extract resume text")
            console.print("[bold red]❌ Failed to read resume.[/bold red]")
            md_logger.save_session_data()
            return
        time.sleep(1)

    console.print("[bold blue]🔍 Processing resume content...[/bold blue]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Analyzing your skills and experience...", total=None)
        entities = extract_entities_hybrid(resume_text, md_logger)
        time.sleep(2)

    console.print("[bold blue]📊 Analyzing your experience level...[/bold blue]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Determining experience level...", total=None)
        experience_years = entities.get("total_experience_years", 0)
        experience_level = determine_experience_level(experience_years, md_logger)
        time.sleep(1)

    console.print("[bold blue]🎯 Identifying your technical domain...[/bold blue]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Determining best-fit technical area...", total=None)
        domain = classify_domain_with_lm_studio(resume_text, md_logger)
        time.sleep(1)

    profile = create_profile(entities, domain, md_logger)
    
    if user_name:
        profile["name"] = user_name
        entities["name"] = user_name 
        md_logger.log_event("info", f"Using user-provided name in profile: {user_name}")
    else:
        md_logger.log_event("info", f"Using extracted name in profile: {profile.get('name', 'Unknown')}")
    
    show_extracted_entities(entities, domain, experience_level)
    
    # Ask user to confirm or continue
    console.print("[bold yellow]Press Enter to start the interview...[/bold yellow]")
    input()
    # Don't clear console - keep resume info visible
    console.print("\n")  # Just add a newline for separation
    
    session_manager = SessionManager(md_logger)
    session_id = session_manager.create_session(profile["name"], domain, experience_level, profile)
    
    # Store resume items for tracking in resume-driven questions
    session_manager.sessions[session_id]["asked_resume_items"] = {
        "projects": set(),
        "skills": set()
    }

    # Removed session ID print
    # console.print(f"[bold green]✓ Interview session started! Session ID: {session_id}[/bold green]")
    console.print(f"[bold green]✓ Interview session started![/bold green]")
    md_logger.session_data["session_id"] = session_id
    md_logger.session_data["domain"] = domain
    md_logger.session_data["experience_level"] = experience_level
    md_logger.session_data["profile"] = profile

    # Natural interview greeting
    console.print(f"\n[bold magenta]🎤 Hi {profile.get('name', 'there')}, I'm LoRa, your interviewer today. Thanks for taking the time to speak with us![/bold magenta]\n")
    logging.info("Starting dynamic interview")
    time.sleep(1)

    question_counter = 0
    current_section_index = 0
    
    # Dynamic interview flow with follow-ups
    for section_index, section_info in enumerate(INTERVIEW_FLOW):
        section_name = section_info["name"]
        num_questions = section_info["question_count"]
        
        # Show section header when entering a new section
        if section_index > 0:
            # Natural transition between sections
            console.print(f"\n[dim]────────────────────────────────────────────[/dim]\n")
            time.sleep(0.8)
        
        show_interview_section_header(section_name, section_info["description"])
        time.sleep(0.8)
        
        # Reset question counter for this section
        session_manager.sessions[session_id]["interview_progress"]["current_question_in_section"] = 0
        
        # Ask questions in this section
        for q_in_section in range(num_questions):
            session_manager.sessions[session_id]["interview_progress"]["current_question_in_section"] = q_in_section
            
            # Generate question dynamically
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task(description="[dim]Thinking...[/dim]", total=None)
                question = generate_dynamic_question(
                    section_info, domain, experience_level, profile, 
                    session_manager, session_id, md_logger
                )
            
            question_counter += 1
            show_interview_question(question)
            response = Prompt.ask("[bold cyan]Your Response[/bold cyan]")
            
            if not response or not response.strip():
                response = "[No response provided]"
            
            # Store response
            if session_manager.add_response(session_id, question, response, section_name):
                md_logger.log_response(question, response, section_name)
            
            # Evaluate response for follow-up or evaluation
            needs_followup = False
            if len(response.strip()) > 10:  # Only evaluate non-empty responses
                # Check if response needs follow-up (except for closing/candidate questions)
                if section_name not in ["Closing", "Candidate's Questions"]:
                    needs_followup = evaluate_response_for_followup(question, response, md_logger)
            
            # Ask follow-up if needed (once per question)
            if needs_followup and q_in_section < num_questions - 1:  # Don't follow-up on last question in section
                console.print("[dim]Let me ask a quick follow-up...[/dim]\n")
                time.sleep(0.3)
                
                followup_question = generate_followup_question(question, response, domain, md_logger)
                show_interview_question(followup_question)
                followup_response = Prompt.ask("[bold cyan]Your Response[/bold cyan]")
                
                if followup_response and followup_response.strip():
                    session_manager.add_response(session_id, followup_question, followup_response, section_name)
                    md_logger.log_response(followup_question, followup_response, section_name)
                    
                    # Update main response with follow-up context
                    session_manager.sessions[session_id]["responses"][-2]["has_followup"] = True
                    session_manager.sessions[session_id]["responses"][-2]["followup"] = {
                        "question": followup_question,
                        "response": followup_response
                    }
            
            # Evaluate performance for technical sections
            if section_name in ["Technical Deep-Dive", "Applied Problem-Solving"]:
                performance_indicator = evaluate_response_simple(question, response, section_name, md_logger)
                question_type = "definition" if session_manager.sessions[session_id]["interview_progress"]["technical_difficulty"] == "EASY" else "application"
                session_manager.update_difficulty_and_context(session_id, performance_indicator, section_name, question_type)
                md_logger.log_event("info", f"Performance indicator: {performance_indicator}, Difficulty: {session_manager.sessions[session_id]['interview_progress']['technical_difficulty']}")
            
            # Natural acknowledgment (sometimes)
            if section_name in ["Resume-Driven Questions", "Behavioral & Soft Skills"] and not needs_followup:
                # Occasionally acknowledge good responses
                if question_counter % 3 == 0:  # Every 3rd question
                    console.print("[dim]Thanks for sharing that.[/dim]\n")
            
            console.print("")
            time.sleep(0.5)

    show_interview_complete() # Removed session_id from display

    console.print("[bold blue]📊 Generating your interview report (PDF & Markdown)...[/bold blue]")
    logging.info("Generating post-interview analysis")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Compiling your report...", total=None)
        session_data = session_manager.get_session_data(session_id)
        session_data["start_time"] = session_data.get("start_time", datetime.now().isoformat())

        # Generate LLM Analysis
        analysis_text = generate_post_interview_analysis(session_data, profile, md_logger)
        # Save analysis to a markdown file
        analysis_filename = f"{base_dir}/analysis/post_interview_analysis_{session_id}.md"
        try:
            with open(analysis_filename, 'w', encoding='utf-8') as f: # Open with UTF-8 encoding
                f.write(analysis_text)
            md_logger.log_event("success", f"LLM Analysis saved to {analysis_filename}")
        except Exception as e:
            md_logger.log_event("error", f"Failed to save LLM Analysis: {str(e)}")


        # Generate PDF Report (now includes analysis_text)
        pdf_filename = f"{base_dir}/reports/interview_report_{session_id}.pdf"
        pdf_result = generate_pdf_report(session_data, pdf_filename, analysis_text, md_logger) # Pass analysis_text

        # Simple markdown report (includes analysis)
        md_filename = f"{base_dir}/reports/interview_report_{session_id}.md"
        try:
            with open(md_filename, 'w', encoding='utf-8') as f: # Open with UTF-8 encoding
                f.write("# AI Technical Interview Report\n")
                f.write(f"**Candidate:** {profile.get('name', 'Unknown')}\n")
                f.write(f"**Domain:** {domain}\n")
                f.write(f"**Experience Level:** {experience_level}\n")
                f.write(f"**Interview Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("## Responses\n")
                for i, resp in enumerate(session_data.get('responses', []), 1):
                    f.write(f"### Q{i}: {resp.get('section', 'N/A')}\n")
                    f.write(f"**Question:** {resp.get('question', 'N/A')}\n")
                    f.write(f"**Response:** {resp.get('response', 'N/A')}\n")
                f.write("\n## AI Generated Post-Interview Analysis\n")
                f.write(analysis_text) # Include the generated analysis
            md_logger.log_event("success", f"Markdown report generated: {md_filename}")
        except Exception as e:
            md_logger.log_event("error", f"Markdown report generation failed: {str(e)}")

        md_logger.save_session_data()
        time.sleep(2)

    console.print(f"\n[bold green]✓ Report generation complete![/bold green]")
    console.print(f"\n[bold yellow]📁 Your Interview Files:[/bold yellow]")
    if pdf_result:
        console.print(f"   • PDF Report: {pdf_filename}")
    console.print(f"   • Markdown Report: {md_filename}")
    console.print(f"   • LLM Analysis: {analysis_filename}")
    console.print(f"   • Detailed Log: {base_dir}/logs/session_log.md")
    # Generic thank you message - Name removed
    console.print(f"\n[bold blue]Thank you for your time![/bold blue]")
    logging.info("Interview process completed successfully")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("[bold red]Usage:[/bold red] python interview_system.py <resume.pdf>")
        logging.error("Incorrect usage - missing PDF path")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        console.print(f"[bold red]❌ File not found:[/bold red] {pdf_path}")
        logging.error(f"File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith('.pdf'):
        console.print("[bold red]❌ Please provide a PDF file.[/bold red]")
        logging.error("Provided file is not a PDF")
        sys.exit(1)

    # Check for fpdf dependency
    try:
        import fpdf
    except ImportError:
        console.print("[bold red]❌ Required dependency 'fpdf' not found.[/bold red]")
        console.print("[italic]Install it with: pip install fpdf2[/italic]")
        logging.error("fpdf dependency not found")
        sys.exit(1)

    run_interview(pdf_path)