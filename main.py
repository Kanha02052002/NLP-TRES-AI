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

# ==================== GLOBAL CONFIGURATION ====================
# LM Studio Configuration
LM_STUDIO_CONFIG = {
    "base_url": "http://127.0.0.1:1234/v1",
    "api_key": "not-needed",
    "default_model": "qwen/qwen3-4b-2507",
    "timeout": 60,
    "headers": {
        "Content-Type": "application/json"
    }
}

# Interview Configuration
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

# Initialize Rich Console
console = Console()

# Set up logging for terminal/console output
logging.basicConfig(filename='interview_terminal.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
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
        with open(self.log_file, 'w') as f:
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
        with open(self.log_file, 'a') as f:
            f.write(f"## {event_type.upper()} - {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"**Message:** {message}\n")
            if data:
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
        with open(self.log_file, 'a') as f:
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

# ==================== TRADITIONAL NER ====================
def extract_entities_traditional(text, logger):
    """Extract entities using spaCy NER + rule-based patterns"""
    logger.log_event("info", "Starting traditional NER extraction")
    try:
        doc = nlp(text)
        entities = {
            "name": next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), None),
            "skills": [],
            "projects": [],
            "work_experience": [],
            "responsibilities": [],
            "education": []
        }

        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["work_experience"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["education"].append(ent.text)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        skills_found = set()
        skills_indicators = ['skill', 'technology', 'tool', 'framework', 'language', 'stack']
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in skills_indicators):
                skills_text = line.split(':', 1)[1] if ':' in line else (line.split('-', 1)[1] if '-' in line else line)
                skills_list = re.split(r'[,;/|&]\s*', skills_text)
                for skill in skills_list:
                    skill_clean = re.sub(r'[^\w\s\.\-#+]', '', skill.strip())
                    if len(skill_clean) > 1 and not skill_clean.isdigit():
                        skills_found.add(skill_clean.title())
        
        skill_patterns = [
            r'\b(python|java|javascript|typescript|react|vue|angular|node\.js|express|django|flask|spring|sql|mongodb|postgresql|docker|kubernetes|aws|gcp|azure|git|github|gitlab|jenkins|terraform|ansible|bash|linux|html|css|sass|scss|bootstrap|tailwind|material|redux|graphql|rest|api|json|xml|yaml|jwt|oauth|tensorflow|pytorch|pandas|numpy|scikit|matplotlib|seaborn|tableau|powerbi)\b'
        ]
        
        for line in lines:
            for pattern in skill_patterns:
                matches = re.findall(pattern, line.lower())
                for match in matches:
                    skills_found.add(match.title())
        
        entities["skills"] = list(skills_found)[:MAX_SKILLS_DISPLAY]

        projects_found = []
        project_indicators = ['project', 'portfolio', 'developed', 'built', 'created', 'implemented']
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line)
            if any(indicator in line_clean.lower() for indicator in project_indicators):
                project_patterns = [
                    r'(?:project|title):\s*([^\n]+)',
                    r'\b(?:developed|built|created|implemented)\s+(?:a|an)?\s+([^,\n\.]+)',
                    r'[-•]\s*([^\n]+)'
                ]
                for pattern in project_patterns:
                    matches = re.findall(pattern, line_clean, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, str) and len(match.strip()) > 5:
                            projects_found.append(match.strip())
        
        entities["projects"] = list(set(projects_found))[:MAX_PROJECTS_DISPLAY]

        work_found = []
        work_indicators = ['experience', 'work', 'employed', 'intern', 'engineer', 'developer', 'analyst', 'scientist', 'manager']
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line)
            if any(indicator in line_clean.lower() for indicator in work_indicators):
                if len(line_clean) > 10:
                    work_found.append(line_clean)
        entities["work_experience"] = list(set(work_found))[:5]

        resp_found = []
        resp_indicators = ['responsibilit', 'duties', 'role', 'managed', 'led', 'mentored', 'supervised']
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line)
            if any(indicator in line_clean.lower() for indicator in resp_indicators):
                if len(line_clean) > 10:
                    resp_found.append(line_clean)
        entities["responsibilities"] = list(set(resp_found))[:3]

        edu_found = []
        edu_indicators = ['education', 'degree', 'bachelor', 'master', 'phd', 'university', 'college', 'institute']
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line)
            if any(indicator in line_clean.lower() for indicator in edu_indicators):
                if len(line_clean) > 10:
                    edu_found.append(line_clean)
        entities["education"] = list(set(edu_found))[:3]

        logger.log_event("success", "Traditional NER extraction completed")
        return entities
    except Exception as e:
        logger.log_event("error", f"Traditional NER extraction failed: {str(e)}")
        return {
            "name": None, "skills": [], "projects": [], "work_experience": [], "responsibilities": [], "education": []
        }

# ==================== LM STUDIO ENHANCED NER ====================
def extract_entities_with_lm_studio(text, logger):
    """Extract entities using LM Studio with structured prompting"""
    logger.log_event("info", "Starting LM Studio NER extraction")
    try:
        prompt = f"""<system>
You are an expert technical resume analyzer. Extract key information from this resume and provide a structured response in JSON.
</system>

<task>
Extract the following from the resume text below:
1. Name
2. Skills (technical skills, programming languages, tools)
3. Projects (with brief descriptions)
4. Work Experience (companies, roles, duration)
5. Additional Responsibilities
6. Education
7. Total professional experience years (integer)

Format your response exactly like this example:

<example>
<resume>
John Doe
Software Engineer with 5 years experience in Python and React
Skills: Python, JavaScript, React, Node.js, SQL
Projects: 
- E-commerce Platform: Built using React and Node.js with payment integration
- Data Visualization Dashboard: Created with Python, D3.js and PostgreSQL
Work Experience:
- Senior Software Engineer at TechCorp (2020-2023)
- Software Developer at StartupXYZ (2018-2020)
Responsibilities: Team leadership, code reviews, mentoring juniors
Education: B.Tech in Computer Science
</resume>
<response>
<extraction>
    <name>John Doe</name>
    <skills>
        <skill>Python</skill>
        <skill>JavaScript</skill>
        <skill>React</skill>
        <skill>Node.js</skill>
        <skill>SQL</skill>
    </skills>
    <projects>
        <project>
            <title>E-commerce Platform</title>
            <description>Built using React and Node.js with payment integration</description>
        </project>
        <project>
            <title>Data Visualization Dashboard</title>
            <description>Created with Python, D3.js and PostgreSQL</description>
        </project>
    </projects>
    <work_experience>
        <experience>
            <role>Senior Software Engineer</role>
            <company>TechCorp</company>
            <duration>2020-2023</duration>
        </experience>
        <experience>
            <role>Software Developer</role>
            <company>StartupXYZ</company>
            <duration>2018-2020</duration>
        </experience>
    </work_experience>
    <responsibilities>
        <responsibility>Team leadership</responsibility>
        <responsibility>Code reviews</responsibility>
        <responsibility>Mentoring juniors</responsibility>
    </responsibilities>
    <education>
        <degree>B.Tech in Computer Science</degree>
    </education>
    <total_experience_years>5</total_experience_years>
</extraction>
</response>
</example>

<task>
Extract information from the following resume and provide XML response:
</task>

<resume>
{text[:2000]}
</resume>

<response>"""
        response_text = call_lm_studio_model(prompt, max_tokens=1500, temperature=0.3, stop=["</response>"])
        if response_text:
            import xml.etree.ElementTree as ET
            try:
                xml_response = "<response>" + response_text.strip() + "</response>"
                root = ET.fromstring(xml_response)
                extraction = root.find('extraction')
                if extraction is not None:
                    entities = {
                        "name": "",
                        "skills": [],
                        "projects": [],
                        "work_experience": [],
                        "responsibilities": [],
                        "education": [],
                        "total_experience_years": 0
                    }
                    
                    name_elem = extraction.find('name')
                    if name_elem is not None:
                        entities["name"] = name_elem.text or ""
                    
                    skills_elem = extraction.find('skills')
                    if skills_elem is not None:
                        for skill in skills_elem.findall('skill'):
                            if skill.text:
                                entities["skills"].append(skill.text.strip())
                    
                    projects_elem = extraction.find('projects')
                    if projects_elem is not None:
                        for project in projects_elem.findall('project'):
                            title_elem = project.find('title')
                            desc_elem = project.find('description')
                            if title_elem is not None and title_elem.text:
                                project_text = title_elem.text
                                if desc_elem is not None and desc_elem.text:
                                    project_text += f": {desc_elem.text}"
                                entities["projects"].append(project_text.strip())
                    
                    work_elem = extraction.find('work_experience')
                    if work_elem is not None:
                        for exp in work_elem.findall('experience'):
                            role_elem = exp.find('role')
                            company_elem = exp.find('company')
                            duration_elem = exp.find('duration')
                            exp_parts = []
                            if role_elem is not None and role_elem.text:
                                exp_parts.append(role_elem.text)
                            if company_elem is not None and company_elem.text:
                                exp_parts.append(f"at {company_elem.text}")
                            if duration_elem is not None and duration_elem.text:
                                exp_parts.append(f"({duration_elem.text})")
                            if exp_parts:
                                entities["work_experience"].append(" ".join(exp_parts))
                    
                    resp_elem = extraction.find('responsibilities')
                    if resp_elem is not None:
                        for resp in resp_elem.findall('responsibility'):
                            if resp.text:
                                entities["responsibilities"].append(resp.text.strip())
                    
                    edu_elem = extraction.find('education')
                    if edu_elem is not None:
                        for edu in edu_elem.findall('degree'):
                            if edu.text:
                                entities["education"].append(edu.text.strip())
                    
                    exp_years_elem = extraction.find('total_experience_years')
                    if exp_years_elem is not None and exp_years_elem.text:
                        try:
                            entities["total_experience_years"] = int(exp_years_elem.text)
                        except ValueError:
                            pass
                    
                    logger.log_event("success", "LM Studio NER extraction completed successfully")
                    return entities
            except ET.ParseError as e:
                logger.log_event("error", f"XML parsing failed: {str(e)}")
                return None
        else:
            logger.log_event("error", "LM Studio returned no response")
            return None
    except Exception as e:
        logger.log_event("error", f"LM Studio NER extraction failed: {str(e)}")
        return None

# ==================== HYBRID NER ====================
def extract_entities_hybrid(text, logger):
    """Combine traditional NER with LM Studio model for better accuracy"""
    logger.log_event("info", "Starting hybrid NER extraction")
    traditional_entities = extract_entities_traditional(text, logger)
    lm_studio_entities = extract_entities_with_lm_studio(text, logger)
    
    if lm_studio_entities:
        final_entities = {
            "name": lm_studio_entities.get("name") or traditional_entities.get("name", "Unknown"),
            "skills": list(set(lm_studio_entities.get("skills", []) + traditional_entities.get("skills", [])))[:MAX_SKILLS_DISPLAY],
            "projects": list(set(lm_studio_entities.get("projects", []) + traditional_entities.get("projects", [])))[:MAX_PROJECTS_DISPLAY],
            "work_experience": list(set(lm_studio_entities.get("work_experience", []) + traditional_entities.get("work_experience", [])))[:5],
            "responsibilities": list(set(lm_studio_entities.get("responsibilities", []) + traditional_entities.get("responsibilities", [])))[:3],
            "education": list(set(lm_studio_entities.get("education", []) + traditional_entities.get("education", [])))[:3],
            "total_experience_years": lm_studio_entities.get("total_experience_years", traditional_entities.get("total_experience_years", 0))
        }
    else:
        final_entities = traditional_entities
        final_entities["total_experience_years"] = final_entities.get("total_experience_years", 0)
    
    logger.log_event("success", "Hybrid NER extraction completed")
    return final_entities

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
            "interview_progress": {
                "current_section_index": 0,
                "current_question_in_section": 0,
                "questions_asked": 0,
                "technical_difficulty": "EASY",
                "performance_score": 0,
                "last_question_type": None,
                "previous_context": None,
                "asked_questions": [] # Track to avoid repetition
            }
        }
        self.logger.log_event("success", f"Session created: {session_id}")
        return session_id

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
            self.sessions[session_id]["interview_progress"]["questions_asked"] += 1
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
    asked_questions = progress.get("asked_questions", [])

    logger.log_event("info", f"Generating {question_type} question for {section_name} (Difficulty: {technical_difficulty})")

    # Master Prompt
    prompt = f"""<system>
You are Alex, a professional and experienced technical interviewer for {domain} roles. Your goal is to conduct a realistic, engaging, and insightful interview. You adapt your questions based on the candidate's experience level ({experience_level}) and previous responses.

Key Principles:
1.  Be conversational and human-like. Avoid robotic phrasing.
2.  Ask one clear, focused question at a time.
3.  Build upon the candidate's previous answers when appropriate.
4.  Vary your question types to keep the interview dynamic.
5.  For technical sections, adjust difficulty based on performance.
6.  Never repeat a question verbatim. If you must revisit a topic, rephrase it significantly.
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
Previously Asked Questions (to avoid): {', '.join(asked_questions[-3:]) if asked_questions else 'None yet'}
</system>

<guidelines>
<introduction>
    - Warmly greet the candidate.
    - Ask them to introduce themselves and highlight experiences relevant to {domain}.
    - Keep it open-ended and conversational.
    Example: "Hi [Name], thanks for joining us today. To start, could you walk me through your background and tell me what draws you to {domain}?"
</introduction>

<resume_descriptive>
    - Pick a specific skill or project from their resume.
    - Ask them to describe it.
    Example: "I see you worked on [Project/Skill]. Can you tell me more about what you built and what technologies you used?"
</resume_analytical>
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
    - Thank the candidate sincerely.
    - Briefly outline next steps if known.
    - End on a positive note.
    Example: "Thanks so much for your time today, [Name]. It was great learning about your experience. We'll be reviewing all candidates and will be in touch soon. Do you have any final thoughts?"
</closing>
</guidelines>

<task>
Generate a single, realistic interview question that fits the current context and guidelines. Be specific, natural, and engaging.
</task>

<question>"""

    # Add previous context if available and relevant
    if previous_context and question_type in ["resume_analytical", "applied_scenario"]:
        prompt += f"\n<previous_context>\n{previous_context.get('response', 'N/A')}\n</previous_context>"

    response_text = call_lm_studio_model(prompt, max_tokens=250, temperature=0.7, stop=["</question>"])
    
    if response_text and len(response_text.strip()) > 10:
        question = response_text.strip()
        # Add to asked questions list
        progress["asked_questions"].append(question)
        if len(progress["asked_questions"]) > 10:  # Keep last 10
            progress["asked_questions"].pop(0)
        logger.log_event("success", f"Question generated: {question}")
        return question

    logger.log_event("warning", "Failed to generate unique question, using fallback.")
    fallbacks = {
        "introduction": f"Hi, could you please introduce yourself and tell me what interests you about {domain}?",
        "resume_descriptive": "Can you tell me about a project or skill from your resume that you're particularly proud of?",
        "resume_analytical": "Building on that, what was the most challenging part and how did you tackle it?",
        "definition": f"What is a key concept in {domain} that you find essential?",
        "application": f"How would you use a core {domain} principle in a real project?",
        "scenario": f"Imagine you face a common problem in {domain}. How would you approach solving it?",
        "applied_scenario": f"Here's a realistic {domain} challenge. How would you go about resolving it?",
        "behavioral_star": "Tell me about a time you had to work through a significant technical or team challenge.",
        "candidate_questions": "What would you like to know about the role or our team?",
        "closing": "Thank you for your time today. We'll be in touch soon. Any last questions?"
    }
    question = fallbacks.get(question_type, "Could you elaborate on that?")
    progress["asked_questions"].append(question)
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

# ==================== PDF REPORT GENERATOR ====================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AI Technical Interview Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        if isinstance(body, list):
            for item in body:
                self.multi_cell(0, 8, f"  - {item}")
        else:
            self.multi_cell(0, 8, body)
        self.ln()

def generate_pdf_report(session_data, filename, logger):
    """Generate a comprehensive PDF report"""
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
            pdf.cell(0, 8, f"Q{i}: {resp.get('section', 'N/A')}", 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, f"Question: {resp.get('question', 'N/A')}")
            pdf.multi_cell(0, 6, f"Response: {resp.get('response', 'N/A')}")
            pdf.ln(4)

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
    ║                    TRES-AI                                   ║
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

def show_interview_progress(current, total, section):
    progress_text = f"[cyan]Progress:[/cyan] {current}/{total} questions | [magenta]Current Section:[/magenta] {section}"
    console.print(progress_text)
    logging.info(f"Progress: {current}/{total} in section {section}")

def show_interview_complete(session_id):
    complete_panel = Panel(
        f"""Thank you for completing the technical interview!

Your responses have been recorded. A detailed analysis and reports
are being generated.

Session files are available in: interview_sessions/{session_id}/""",
        title="[bold green]🎉 Interview Complete[/bold green]",
        border_style="green"
    )
    console.print(complete_panel)
    logging.info("Interview completion message displayed")

# ==================== MAIN INTERVIEW FLOW ====================
def run_interview(pdf_path):
    show_welcome()
    logging.info(f"Starting interview process for {pdf_path}")

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
    session_manager = SessionManager(md_logger)
    session_id = session_manager.create_session(profile["name"], domain, experience_level, profile)
    console.print(f"[bold green]✓ Interview session started! Session ID: {session_id}[/bold green]")

    md_logger.session_data["session_id"] = session_id
    md_logger.session_data["domain"] = domain
    md_logger.session_data["experience_level"] = experience_level
    md_logger.session_data["profile"] = profile

    console.print("\n[bold magenta]🎤 Alex here, your interviewer. Let's begin...[/bold magenta]\n")
    logging.info("Starting structured interview questions")

    total_questions = sum(section["question_count"] for section in INTERVIEW_FLOW)
    question_counter = 0

    for section_index, section_info in enumerate(INTERVIEW_FLOW):
        section_name = section_info["name"]
        num_questions = section_info["question_count"]

        session_manager.sessions[session_id]["interview_progress"]["current_question_in_section"] = 0
        show_interview_section_header(section_name, section_info["description"])

        for q_in_section in range(num_questions):
            question_counter += 1
            session_manager.sessions[session_id]["interview_progress"]["current_question_in_section"] = q_in_section
            show_interview_progress(question_counter, total_questions, section_name)

            question = generate_question_master_prompt(
                domain, section_info, experience_level, session_manager, session_id, md_logger
            )

            show_interview_question(question)

            response = Prompt.ask("[bold cyan]Your Response[/bold cyan]")

            if session_manager.add_response(session_id, question, response, section_name):
                if section_name == "Resume-Driven Questions":
                    session_manager.sessions[session_id]["interview_progress"]["previous_context"] = {
                        "question": question, "response": response
                    }
                md_logger.log_response(question, response, section_name)
            else:
                md_logger.log_event("error", f"Failed to store response for question {question_counter}")

            if section_name in ["Technical Deep-Dive", "Applied Problem-Solving"]:
                performance_indicator = evaluate_response_simple(question, response, section_name, md_logger)
                dummy_q_type = "definition"  # Simplified for tracking
                session_manager.update_difficulty_and_context(session_id, performance_indicator, section_name, dummy_q_type)
                md_logger.log_event("info", f"Performance indicator for Q{question_counter}: {performance_indicator}")

            console.print("")
            time.sleep(1)

    show_interview_complete(session_id)

    console.print("[bold blue]📊 Generating your interview report (PDF & Markdown)...[/bold blue]")
    logging.info("Generating post-interview analysis")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Compiling your report...", total=None)

        session_data = session_manager.get_session_data(session_id)
        session_data["start_time"] = session_data.get("start_time", datetime.now().isoformat())

        # Generate PDF Report
        pdf_filename = f"{base_dir}/reports/interview_report_{session_id}.pdf"
        pdf_result = generate_pdf_report(session_data, pdf_filename, md_logger)
        
        # Simple markdown report
        md_filename = f"{base_dir}/reports/interview_report_{session_id}.md"
        try:
            with open(md_filename, 'w') as f:
                f.write("# AI Technical Interview Report\n\n")
                f.write(f"**Candidate:** {profile.get('name', 'Unknown')}\n")
                f.write(f"**Domain:** {domain}\n")
                f.write(f"**Experience Level:** {experience_level}\n")
                f.write(f"**Interview Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Responses\n")
                for i, resp in enumerate(session_data.get('responses', []), 1):
                    f.write(f"### Q{i}: {resp.get('section', 'N/A')}\n")
                    f.write(f"**Question:** {resp.get('question', 'N/A')}\n")
                    f.write(f"**Response:** {resp.get('response', 'N/A')}\n\n")
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
    console.print(f"   • Detailed Log: {base_dir}/logs/session_log.md")
    console.print(f"\n[bold blue]Thank you for your time, {profile.get('name', 'candidate')}![/bold blue]")
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