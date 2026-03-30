from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

logger = logging.getLogger("nexvec.llm_query")

_CLASSIFY_PROMPT = """\
You are an expert PostgreSQL query generator for NexVec, a recruitment platform.
Your job: given a recruiter's natural language query, produce (1) a precise SQL SELECT and (2) a routing decision.

────────────────────────────────────────────────────────────
DATABASE SCHEMA (AWS RDS PostgreSQL)
────────────────────────────────────────────────────────────

Table: users  (one row per person)
  user_id   TEXT PRIMARY KEY
  name      TEXT
  email     TEXT
  phone     TEXT
  location  TEXT   e.g. "Bangalore, India"

Table: resumes  (one row per uploaded resume)
  resume_id              TEXT PRIMARY KEY
  user_id                TEXT  FK → users.user_id
  objectives             TEXT  career objective / professional summary
  work_experience_years  NUMERIC  float years e.g. 2.92, 5.5
  work_experience_text   TEXT  full job history — companies, roles, dates, responsibilities
  projects               TEXT  project descriptions and technologies
  education              TEXT  degrees, institutions, graduation years, GPA
  skills                 TEXT[]  array of skill strings
  achievements           TEXT  awards, certifications, extracurriculars
  is_active              BOOLEAN  always filter: is_active = TRUE

────────────────────────────────────────────────────────────
STEP 1 — INPUT NORMALISATION (apply silently before SQL)
────────────────────────────────────────────────────────────

Fix ALL typos, abbreviations, and shorthand BEFORE generating SQL.

## Technology / Skill normalisation
pyhton / pythoon / pythn / pytho → Python
javascrpt / java script / javasript / javscript / js → JavaScript
typescrpit / type script / ts (when meaning TypeScript) → TypeScript
recat / reactjs / react js → React
vuejs / vue js / veu / vueJs → Vue.js
angularjs / angular js / angualr / angualrjs → Angular
nodejs / node js / nodjs (when meaning runtime) → Node.js
postgress / postgressql / psql / pgsql (when meaning database) → PostgreSQL
mongod / mongo (when standalone) → MongoDB
tensorfow / tensorflw / tf (when meaning ML library) → TensorFlow
kuberntes / k8s / k8 → Kubernetes
dockerr / dokcer → Docker
awss / amazon cloud → AWS
gcpp / google cloud platform → Google Cloud Platform (use GCP in ARRAY)
azzure / azur / ms cloud → Azure
msql / my sql → MySQL
djagno / djangoo / dajngo → Django
flassk / flaks / flsk → Flask
fasapi / fastpi / fastpai → FastAPI
expres / expressjs / expresjs → Express.js
pytroch / pytoch → PyTorch
scikit / scikit learn → Scikit-learn
sklearn → Scikit-learn
golng / go lang / golang → Go
rubyon rails / ror → Ruby on Rails
dotnet / dot net → .NET
springboot / spring-boot → Spring Boot
reactnative / react-native → React Native
fluttr / fluuter → Flutter
swfit / swft → Swift
kotln / kotlin → Kotlin
andriod / androit / androiid → Android
elasitcsearch / elasticserach → Elasticsearch
graphql / grphql / grapql → GraphQL
kuberentes → Kubernetes

## Abbreviations (context-aware)
ml → Machine Learning
dl → Deep Learning
nlp → Natural Language Processing
cv → Computer Vision  (NOT "curriculum vitae" in skill context)
ai → Artificial Intelligence
devops → DevOps
sre → Site Reliability Engineering
ci/cd → CI/CD
api → API
dsa → Data Structures and Algorithms
oop → Object Oriented Programming
ror → Ruby on Rails
fe → frontend / front-end
be → backend / back-end
fs → full-stack

## Role abbreviations
dev → developer
eng → engineer
sde → Software Development Engineer
swe → Software Engineer
pm → Product Manager   (NOT Python — context matters)
ba → Business Analyst
qa → Quality Assurance engineer
ux → UX Designer
dba → Database Administrator
sa → Solutions Architect

## Indian city abbreviations / alternate spellings
blore / blr / bengaluru / bangalroe / bangalor → Bangalore
hyd / hydrabad / hyderabd → Hyderabad
mum / bom / bombay / mumbia → Mumbai
del / dellhi / new delhi → Delhi
puna / poona / pune → Pune
madras / chenai / chn → Chennai
calcutta / kolkatta → Kolkata

## Company name typos
googel / gogle / googel → Google
mircosoft / microsft / mircrosoft → Microsoft
amazzon / amzon / amazon inc → Amazon
aplle / aple → Apple
netflex / netlix → Netflix
ubar / ubr → Uber
airbnbb → Airbnb
infy → Infosys
tcs → TCS (do NOT expand to full name in ILIKE, use '%TCS%' OR '%Tata Consultancy%')
cogniznat / cogniznat → Cognizant
wipor / wpr → Wipro
accentre / accenture → Accenture

## Experience unit shorthand
yrs / yr → years
xp / exp → experience (treat as "experience")

────────────────────────────────────────────────────────────
STEP 2 — SQL GENERATION RULES
────────────────────────────────────────────────────────────

### Absolute rules
- Always start: SELECT r.resume_id FROM resumes r
- Always include: WHERE r.is_active = TRUE
- Return ONLY resume_id — no other columns
- No markdown, no explanation, no semicolons
- Only SELECT — never UPDATE/DELETE/DROP/INSERT

### Skills (TEXT[])
- Single required skill:       r.skills @> ARRAY['Python']
- ANY one of these skills:     r.skills && ARRAY['Python','Java']
- ALL of these skills:         r.skills @> ARRAY['Python','React']
- Title Case normalisation:    'Python' 'JavaScript' 'React' 'Node.js' 'PostgreSQL' 'TensorFlow'
- Special chars are fine:      ARRAY['C++']  ARRAY['C#']  ARRAY['.NET']
- When skill is vague/compound, also search text columns with ILIKE

### Work Experience (NUMERIC float — NULLs exist for incomplete resumes)

Explicit numbers:
  "X years" / "at least X" / "X+" / "minimum X" / "no less than X" → >= X
  "more than X" / "over X" / "above X" / "exceeding X"            → > X
  "less than X" / "under X" / "below X"                           → < X
  "exactly X years"                                                → = X
  "between X and Y" / "X to Y" / "X-Y years"                      → BETWEEN X AND Y
  "~X years" / "around X" / "roughly X" / "approximately X"       → >= (X - 0.5)
  "almost X years"                                                 → >= (X - 0.5)

Time-unit conversions (convert to year fractions):
  "6 months" / "half a year" / "half year"                        → >= 0.5
  "18 months" / "a year and a half"                               → >= 1.5
  "a year" / "one year"                                           → >= 1
  "two years" / "couple of years" / "couple years"                → >= 2
  "three years"                                                   → >= 3
  "a few years" / "few years"                                     → >= 1
  "several years"                                                  → >= 3
  "many years"                                                     → >= 5

Vague positive experience (ALWAYS generate a filter — never a full scan):
  "some experience" / "little experience" / "any experience"       → work_experience_years > 0
  "a bit of experience" / "some exp"                               → work_experience_years > 0
  "decent experience" / "good experience" / "solid experience"     → work_experience_years >= 2
  "reasonable experience" / "moderate experience"                  → work_experience_years >= 2

Seniority levels (use these numeric thresholds):
  "fresher" / "fresh graduate" / "fresh grad" / "new grad" / "0 experience" / "no experience"
    → (r.work_experience_years IS NULL OR r.work_experience_years < 1)
  "entry level" / "entry-level" / "entrylevel"                     → r.work_experience_years < 2
  "junior" / "jr"                                                  → r.work_experience_years < 3
  "mid level" / "mid-level" / "intermediate" / "associate"         → r.work_experience_years BETWEEN 2 AND 5
  "senior" / "sr" / "seasoned" / "experienced professional"        → r.work_experience_years >= 5
  "lead" / "tech lead" / "team lead"                               → r.work_experience_years >= 7
  "principal" / "staff engineer" / "staff"                         → r.work_experience_years >= 8
  "architect"                                                       → r.work_experience_years >= 8

### Role-type queries (engineer, developer, designer, etc.)
- Never produce a bare full-scan when the query mentions a role type
- Search work_experience_text or objectives with ILIKE:
    r.work_experience_text ILIKE '%software engineer%'
- Combine with experience filter when mentioned

### Free-text search rules
- Use ILIKE '%keyword%' on text columns — always case-insensitive
- Multi-column search with OR when concept could appear in multiple sections:
    (r.work_experience_text ILIKE '%k%' OR r.projects ILIKE '%k%' OR r.skills && ARRAY['K'])
- For domain/industry queries (fintech, healthcare, banking, etc.):
    (r.work_experience_text ILIKE '%fintech%' OR r.projects ILIKE '%fintech%')

### OR expansion for abbreviations / acronyms — CRITICAL RULE
When a term has a well-known abbreviation AND full form, ALWAYS include BOTH in ILIKE and ARRAY.
Never rely on only one form — resumes use both interchangeably.

Mandatory expansions (always generate all forms shown):
  ML / machine learning   →  ILIKE '%machine learning%' OR ILIKE '%ML%' OR ILIKE '%ml%'
                              skills: ARRAY['Machine Learning','ML','Deep Learning']
  NLP / natural language  →  ILIKE '%natural language processing%' OR ILIKE '%NLP%'
                              skills: ARRAY['NLP','Natural Language Processing']
  CV / computer vision    →  ILIKE '%computer vision%' OR ILIKE '%CV%' OR ILIKE '%image recognition%'
                              skills: ARRAY['Computer Vision','CV']
  AI / artificial intel.  →  ILIKE '%artificial intelligence%' OR ILIKE '%AI%'
                              skills: ARRAY['AI','Artificial Intelligence','Machine Learning']
  DL / deep learning      →  ILIKE '%deep learning%' OR ILIKE '%DL%' OR ILIKE '%neural network%'
                              skills: ARRAY['Deep Learning','Neural Networks','TensorFlow','PyTorch']
  DevOps                  →  ILIKE '%devops%' OR ILIKE '%DevOps%' OR ILIKE '%dev ops%'
                              skills: ARRAY['DevOps','CI/CD','Docker','Kubernetes']
  SRE                     →  ILIKE '%site reliability%' OR ILIKE '%SRE%'
  CI/CD                   →  ILIKE '%CI/CD%' OR ILIKE '%continuous integration%' OR ILIKE '%continuous delivery%'
  data science            →  ILIKE '%data science%' OR ILIKE '%data scientist%'
                              skills: ARRAY['Data Science','Machine Learning','Python','Statistics']
  data engineering        →  ILIKE '%data engineer%' OR ILIKE '%ETL%' OR ILIKE '%data pipeline%'
                              skills: ARRAY['Spark','Kafka','Airflow','ETL','Data Engineering']
  API / REST / backend    →  ILIKE '%API%' OR ILIKE '%REST%' OR ILIKE '%backend%'
  frontend / UI           →  ILIKE '%frontend%' OR ILIKE '%front-end%' OR ILIKE '%UI%'
  full stack              →  ILIKE '%full stack%' OR ILIKE '%fullstack%' OR ILIKE '%full-stack%'

General rule: if the query term is an acronym or has a common short form, add BOTH to the OR clause.

### JOIN rules
- Join users ONLY when filtering on name, email, phone, or location
- Syntax: JOIN users u ON u.user_id = r.user_id

### Negative skill queries
- "without X" / "no X" / "don't know X" / "not X":
    AND NOT (r.skills @> ARRAY['X'] OR r.work_experience_text ILIKE '%X%')

### FAANG / big-tech pattern
- "FAANG" / "big tech" / "top tech":
    (r.work_experience_text ILIKE '%Google%' OR r.work_experience_text ILIKE '%Amazon%'
     OR r.work_experience_text ILIKE '%Apple%' OR r.work_experience_text ILIKE '%Meta%'
     OR r.work_experience_text ILIKE '%Netflix%' OR r.work_experience_text ILIKE '%Microsoft%')

### Compound / stack patterns
- "full stack" / "fullstack" / "full-stack":
    (r.work_experience_text ILIKE '%full stack%' OR r.work_experience_text ILIKE '%fullstack%'
     OR r.skills && ARRAY['React','Node.js'])
- "MERN": r.work_experience_text ILIKE '%MERN%' OR r.skills && ARRAY['MongoDB','Express.js','React','Node.js']
- "MEAN": similar with Angular
- "cloud": r.skills && ARRAY['AWS','GCP','Azure']
- "mobile": r.skills && ARRAY['iOS','Android','React Native','Flutter'] OR r.work_experience_text ILIKE '%mobile%'

### Certification queries
- "AWS certified": r.achievements ILIKE '%AWS%' OR r.skills && ARRAY['AWS']
- "Google certified": r.achievements ILIKE '%Google%'
- Any certification: check achievements column first, then skills

### ORDER BY — always add to SQL for better RDS-only ranking
- Experience-based queries: ORDER BY r.work_experience_years DESC NULLS LAST
- Name / exploratory queries: ORDER BY r.created_at DESC
- Skills-only queries: ORDER BY r.created_at DESC
- Location queries: ORDER BY r.work_experience_years DESC NULLS LAST

────────────────────────────────────────────────────────────
STEP 3 — VECTOR SEARCH ROUTING RULES
────────────────────────────────────────────────────────────

Set needs_vector = true when the query is semantically open-ended OR uses vague qualifiers:
  ✓ Vague quality adjectives: "best", "strong", "top", "great", "talented", "innovative"
  ✓ Concept/domain matching: "built NLP products", "led cross-functional teams", "startup experience"
  ✓ Soft skills: "problem solver", "communicator", "team player", "self-starter"
  ✓ Vague experience phrases: "hands on", "hands-on", "practical experience", "exposure to",
    "worked with", "dealt with", "familiar with", "used", "involved in"
  ✓ Broad domain without exact skill: "automation", "testing", "scripting", "infrastructure",
    "data work", "cloud work", "backend work", "AI work"
  ✓ Any query where the concept spans many synonyms you cannot exhaustively list in SQL
  ✓ When the reason you are writing mentions "vague" or "semantic" — that means needs_vector=true

CRITICAL RULE: if your reason mentions the query is vague or needs semantic ranking,
you MUST set needs_vector=true. Never contradict yourself — reason and needs_vector must agree.

Set needs_vector = false (RDS-only) ONLY when:
  ✗ Explicit year / seniority filter is present AND no vague qualifier → RDS only
  ✗ Explicit named skills with no vague qualifier → RDS only
  ✗ Location, company name, education institution → RDS only
  ✗ Certification names → RDS only
  ✗ Name search → RDS only
  ✗ Full scan (show all) → RDS only

────────────────────────────────────────────────────────────
OUTPUT FORMAT — respond with ONLY this JSON (no markdown):
────────────────────────────────────────────────────────────
{
  "sql": "<SQL>",
  "needs_vector": <true|false>,
  "reason": "<one sentence>"
}

────────────────────────────────────────────────────────────
EXAMPLES
────────────────────────────────────────────────────────────

## Explicit experience + skill
Query: "candidates with 4+ years experience in Python"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 4 AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Explicit years and skill — RDS only."}

Query: "Python and React developers with 3+ years"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 3 AND r.skills @> ARRAY['Python','React'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Explicit skills and years — RDS only."}

Query: "Java developers with more than 5 years"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years > 5 AND r.skills @> ARRAY['Java'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Explicit skill and strict years filter — RDS only."}

Query: "candidates with exactly 3 years experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years = 3 ORDER BY r.created_at DESC","needs_vector":false,"reason":"Exact years filter — RDS only."}

Query: "engineers with 2 to 5 years experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years BETWEEN 2 AND 5 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Year range filter — RDS only."}

## Time unit conversions
Query: "developers with 6 months experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 0.5 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"6 months = 0.5 years — RDS only."}

Query: "candidates with 18 months of experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 1.5 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"18 months = 1.5 years — RDS only."}

Query: "engineers with a couple of years experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 2 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Couple of years = >= 2 — RDS only."}

## Vague experience words
Query: "engineers with some experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years > 0 AND r.work_experience_text ILIKE '%engineer%' ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'Some experience' = > 0 years; role keyword filtered via ILIKE — RDS only."}

Query: "developers with decent experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 2 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'Decent experience' = >= 2 years — RDS only."}

Query: "developers with a few years of experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 1 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'Few years' = >= 1 year — RDS only."}

Query: "candidates with several years of experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 3 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'Several years' = >= 3 years — RDS only."}

Query: "candidates with around 4 years experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 3.5 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'Around 4 years' = >= 3.5 years — RDS only."}

## Seniority levels
Query: "freshers with no experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_years IS NULL OR r.work_experience_years < 1) ORDER BY r.created_at DESC","needs_vector":false,"reason":"Fresher = < 1 year, include NULL — RDS only."}

Query: "fresh graduate developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_years IS NULL OR r.work_experience_years < 1) ORDER BY r.created_at DESC","needs_vector":false,"reason":"Fresh graduate = < 1 year — RDS only."}

Query: "junior engineers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years < 3 AND r.work_experience_text ILIKE '%engineer%' ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Junior = < 3 years — RDS only."}

Query: "mid level Python developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years BETWEEN 2 AND 5 AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Mid-level = 2–5 years, explicit skill — RDS only."}

Query: "senior developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 5 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Senior = >= 5 years — RDS only."}

Query: "tech leads with Java experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 7 AND r.skills @> ARRAY['Java'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Tech lead = >= 7 years — RDS only."}

Query: "principal engineers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 8 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Principal = >= 8 years — RDS only."}

## Typo correction examples
Query: "pyhton developers with 3 years"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 3 AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'pyhton' corrected to Python — RDS only."}

Query: "candidates who know javascrpt and recat"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['JavaScript','React'] ORDER BY r.created_at DESC","needs_vector":false,"reason":"'javascrpt'→JavaScript, 'recat'→React corrected — RDS only."}

Query: "ml enginer with 2+ yrs in tensorfow"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 2 AND (r.skills && ARRAY['Machine Learning','ML','TensorFlow'] OR r.work_experience_text ILIKE '%machine learning%' OR r.work_experience_text ILIKE '%ML%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"ml→Machine Learning (both abbreviation+full in ARRAY), tensorfow→TensorFlow — RDS only."}

Query: "data science candidates with nlp experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.skills && ARRAY['Data Science','Machine Learning','NLP','Natural Language Processing'] OR r.work_experience_text ILIKE '%data science%' OR r.work_experience_text ILIKE '%data scientist%' OR r.work_experience_text ILIKE '%natural language processing%' OR r.work_experience_text ILIKE '%NLP%' OR r.projects ILIKE '%NLP%' OR r.projects ILIKE '%natural language processing%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"data science + NLP both expanded with full form and abbreviation in ILIKE and skills — RDS only."}

Query: "computer vision or CV engineers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.skills && ARRAY['Computer Vision','CV','Image Recognition','OpenCV'] OR r.work_experience_text ILIKE '%computer vision%' OR r.work_experience_text ILIKE '%image recognition%' OR r.projects ILIKE '%computer vision%' OR r.projects ILIKE '%CV%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"CV/computer vision both forms expanded in ILIKE and skills — RDS only."}

Query: "nodejs devloper with postgress exp"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['Node.js','PostgreSQL'] ORDER BY r.created_at DESC","needs_vector":false,"reason":"Typos corrected: nodejs→Node.js, devloper→developer, postgress→PostgreSQL — RDS only."}

Query: "k8s and dokcer enginer 3+ yrs xp"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 3 AND r.skills @> ARRAY['Kubernetes','Docker'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"k8s→Kubernetes, dokcer→Docker, yrs xp→years experience — RDS only."}

Query: "reactnative or fluttr developer"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills && ARRAY['React Native','Flutter'] ORDER BY r.created_at DESC","needs_vector":false,"reason":"reactnative→React Native, fluttr→Flutter — RDS only."}

Query: "springboot devloper with 4 yrs"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 4 AND r.skills @> ARRAY['Spring Boot'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"springboot→Spring Boot, devloper→developer — RDS only."}

## Location + city abbreviations
Query: "candidates from Bangalore"
{"sql":"SELECT r.resume_id FROM resumes r JOIN users u ON u.user_id = r.user_id WHERE r.is_active = TRUE AND u.location ILIKE '%Bangalore%' ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Location filter — RDS only."}

Query: "Python devs from blore"
{"sql":"SELECT r.resume_id FROM resumes r JOIN users u ON u.user_id = r.user_id WHERE r.is_active = TRUE AND u.location ILIKE '%Bangalore%' AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'blore'→Bangalore location filter + explicit skill — RDS only."}

Query: "engineers in hyd with 2+ years"
{"sql":"SELECT r.resume_id FROM resumes r JOIN users u ON u.user_id = r.user_id WHERE r.is_active = TRUE AND u.location ILIKE '%Hyderabad%' AND r.work_experience_years >= 2 ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'hyd'→Hyderabad location filter + years — RDS only."}

## Special characters (C++, C#, .NET, Go)
Query: "C++ developers with 5 years"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 5 AND (r.skills @> ARRAY['C++'] OR r.work_experience_text ILIKE '%C++%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"C++ skill with years filter — RDS only."}

Query: "C# and .NET developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.skills @> ARRAY['C#'] OR r.skills @> ARRAY['.NET'] OR r.work_experience_text ILIKE '%C#%' OR r.work_experience_text ILIKE '%.NET%') ORDER BY r.created_at DESC","needs_vector":false,"reason":"C# and .NET special character skills — RDS only."}

Query: "Go language developers with 3 years"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 3 AND (r.skills @> ARRAY['Go'] OR r.work_experience_text ILIKE '%Golang%' OR r.work_experience_text ILIKE '% Go %') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Go language with alias Golang — RDS only."}

## Compound / stack patterns
Query: "full stack developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%full stack%' OR r.work_experience_text ILIKE '%fullstack%' OR r.skills && ARRAY['React','Node.js','JavaScript']) ORDER BY r.created_at DESC","needs_vector":false,"reason":"Full stack — text + skill search, RDS only."}

Query: "MERN stack developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%MERN%' OR r.skills && ARRAY['MongoDB','Express.js','React','Node.js']) ORDER BY r.created_at DESC","needs_vector":false,"reason":"MERN stack — component skills — RDS only."}

Query: "cloud engineers with AWS or GCP"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills && ARRAY['AWS','GCP','Google Cloud Platform'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Explicit cloud platform skills — RDS only."}

Query: "mobile developers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.skills && ARRAY['iOS','Android','React Native','Flutter'] OR r.work_experience_text ILIKE '%mobile%') ORDER BY r.created_at DESC","needs_vector":false,"reason":"Mobile — multi-platform skill search — RDS only."}

## Negative queries
Query: "developers without Java"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND NOT (r.skills @> ARRAY['Java'] OR r.work_experience_text ILIKE '%Java%') ORDER BY r.created_at DESC","needs_vector":false,"reason":"Negative Java filter — RDS only."}

Query: "Python engineers who don't know PHP"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['Python'] AND NOT (r.skills @> ARRAY['PHP'] OR r.work_experience_text ILIKE '%PHP%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Skill include + negative exclude — RDS only."}

## Domain / industry queries
Query: "candidates with fintech experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%fintech%' OR r.work_experience_text ILIKE '%finance%' OR r.projects ILIKE '%fintech%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Domain keyword across text columns — RDS only."}

Query: "engineers with healthcare or medical domain experience"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%healthcare%' OR r.work_experience_text ILIKE '%medical%' OR r.projects ILIKE '%healthcare%') ORDER BY r.created_at DESC","needs_vector":false,"reason":"Domain keyword search — RDS only."}

Query: "FAANG experience candidates"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%Google%' OR r.work_experience_text ILIKE '%Amazon%' OR r.work_experience_text ILIKE '%Apple%' OR r.work_experience_text ILIKE '%Meta%' OR r.work_experience_text ILIKE '%Netflix%' OR r.work_experience_text ILIKE '%Microsoft%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"FAANG companies — keyword search — RDS only."}

## Certification queries
Query: "AWS certified candidates"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.achievements ILIKE '%AWS%' OR r.skills && ARRAY['AWS']) ORDER BY r.created_at DESC","needs_vector":false,"reason":"AWS certification — achievements + skill check — RDS only."}

Query: "PMP certified project managers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.achievements ILIKE '%PMP%' ORDER BY r.created_at DESC","needs_vector":false,"reason":"PMP certification in achievements — RDS only."}

## Education queries
Query: "IIT graduates with Python skills"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.education ILIKE '%IIT%' AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Education institution + skill — RDS only."}

Query: "candidates from IIT or NIT"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.education ILIKE '%IIT%' OR r.education ILIKE '%NIT%') ORDER BY r.created_at DESC","needs_vector":false,"reason":"Education institution filter — RDS only."}

## Name search
Query: "what is sravan's experience"
{"sql":"SELECT r.resume_id FROM resumes r JOIN users u ON u.user_id = r.user_id WHERE r.is_active = TRUE AND u.name ILIKE '%sravan%' ORDER BY r.created_at DESC","needs_vector":false,"reason":"Name search — RDS only."}

## Company search
Query: "candidates who have worked at Google"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_text ILIKE '%Google%' ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Company name keyword — RDS only."}

Query: "candidates from googel or mircosoft"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%Google%' OR r.work_experience_text ILIKE '%Microsoft%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"'googel'→Google, 'mircosoft'→Microsoft company search — RDS only."}

## Conversational / question-form queries
Query: "who knows Python?"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Question-form → skill filter — RDS only."}

Query: "do we have any Java developers?"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['Java'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Conversational query → Java skill filter — RDS only."}

Query: "I need someone for a React role"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['React'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Intent → React skill filter — RDS only."}

Query: "can you show me all data scientists?"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%data scientist%' OR r.skills && ARRAY['Data Science','Machine Learning','Python']) ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Role text + skill search — RDS only."}

## Full scan
Query: "show all candidates"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE ORDER BY r.created_at DESC","needs_vector":false,"reason":"Full scan — RDS only."}

Query: "list everyone"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE ORDER BY r.created_at DESC","needs_vector":false,"reason":"Full scan — RDS only."}

## Semantic / vector-needed queries — vague qualifier examples
Query: "show me candidates with experience and hands on with automation"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%automation%' OR r.work_experience_text ILIKE '%scripting%' OR r.work_experience_text ILIKE '%CI/CD%' OR r.work_experience_text ILIKE '%devops%' OR r.projects ILIKE '%automation%' OR r.skills && ARRAY['Automation','DevOps','CI/CD','Selenium','Ansible','Jenkins','Bash','Python']) ORDER BY r.created_at DESC","needs_vector":true,"reason":"'Hands on' is a vague qualifier — SQL broadens automation to related terms, vector re-ranks semantically."}

Query: "candidates who have practical experience with cloud"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.skills && ARRAY['AWS','GCP','Azure','Cloud'] OR r.work_experience_text ILIKE '%cloud%') ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":true,"reason":"'Practical experience' is vague — SQL filters cloud, vector re-ranks by depth of experience."}

Query: "someone familiar with backend development"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%backend%' OR r.work_experience_text ILIKE '%back-end%' OR r.work_experience_text ILIKE '%server%' OR r.skills && ARRAY['Node.js','Django','FastAPI','Go','Java','Spring Boot']) ORDER BY r.created_at DESC","needs_vector":true,"reason":"'Familiar with' is a vague qualifier — SQL expands backend, vector re-ranks semantically."}

Query: "show me people with good exposure to testing"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%test%' OR r.work_experience_text ILIKE '%QA%' OR r.work_experience_text ILIKE '%quality assurance%' OR r.projects ILIKE '%test%' OR r.skills && ARRAY['Testing','QA','Selenium','Jest','Pytest','JUnit']) ORDER BY r.created_at DESC","needs_vector":true,"reason":"'Good exposure' is vague — SQL expands testing to related terms, vector re-ranks."}

## Semantic / vector-needed queries
Query: "engineers who worked on computer vision projects"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.projects ILIKE '%computer vision%' OR r.work_experience_text ILIKE '%computer vision%' OR r.projects ILIKE '%CV%' OR r.skills && ARRAY['Computer Vision','CV','Image Recognition']) ORDER BY r.created_at DESC","needs_vector":true,"reason":"Semantic concept — computer vision in work or projects."}

Query: "strong backend engineers who built scalable systems"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%backend%' OR r.work_experience_text ILIKE '%back-end%' OR r.skills && ARRAY['Node.js','Django','FastAPI','Go']) ORDER BY r.created_at DESC","needs_vector":true,"reason":"Quality/vague description — vector re-ranks SQL results."}

Query: "candidates who led cross-functional teams"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%led%' OR r.work_experience_text ILIKE '%cross-functional%' OR r.achievements ILIKE '%team lead%') ORDER BY r.created_at DESC","needs_vector":true,"reason":"Leadership concept — vector re-ranks SQL results."}

Query: "find me the best Python developer"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":true,"reason":"'Best' is subjective — SQL filters Python, vector re-ranks by quality."}

Query: "startup experience engineers"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_text ILIKE '%startup%' ORDER BY r.created_at DESC","needs_vector":true,"reason":"Startup culture is semantic — vector re-ranks."}

Query: "innovative problem solvers who are passionate about technology"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE ORDER BY r.created_at DESC","needs_vector":true,"reason":"Purely semantic — vector re-ranks full pool."}

Query: "candidates who built NLP or ML products from scratch"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND (r.work_experience_text ILIKE '%natural language processing%' OR r.work_experience_text ILIKE '%NLP%' OR r.work_experience_text ILIKE '%machine learning%' OR r.work_experience_text ILIKE '%ML%' OR r.projects ILIKE '%natural language processing%' OR r.projects ILIKE '%NLP%' OR r.projects ILIKE '%machine learning%' OR r.projects ILIKE '%ML%' OR r.skills && ARRAY['NLP','Natural Language Processing','Machine Learning','ML']) ORDER BY r.created_at DESC","needs_vector":true,"reason":"NLP/ML both expanded — 'built from scratch' is semantic, vector re-ranks."}

## RDS-only examples
Query: "candidates with 4+ years experience in Python"
{"sql":"SELECT r.resume_id FROM resumes r WHERE r.is_active = TRUE AND r.work_experience_years >= 4 AND r.skills @> ARRAY['Python'] ORDER BY r.work_experience_years DESC NULLS LAST","needs_vector":false,"reason":"Explicit years and skill — RDS only."}

────────────────────────────────────────────────────────────
## Now generate JSON for this query:
{query}\
"""


@dataclass
class QueryClassification:
    sql: str
    needs_vector: bool
    reason: str


def classify_and_generate_sql(query: str) -> QueryClassification:
    """
    Use Gemini Flash to convert a recruiter's natural language query into:
      1. A PostgreSQL SELECT statement returning resume_ids
      2. A routing decision (RDS-only vs RDS + vector search)

    Returns QueryClassification(sql, needs_vector, reason).
    """
    client = genai.Client()
    prompt = _CLASSIFY_PROMPT.replace("{query}", query.strip())

    logger.info("Classifying query=%r", query)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    raw = response.text.strip()
    # Strip markdown fences if model wraps output
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
        sql = data["sql"].strip().rstrip(";")
        needs_vector = bool(data.get("needs_vector", False))
        reason = str(data.get("reason", ""))

        logger.info("SQL=%s | needs_vector=%s | reason=%s", sql, needs_vector, reason)
        return QueryClassification(sql=sql, needs_vector=needs_vector, reason=reason)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("JSON parse failed (%s) on: %r", exc, raw[:300])
        # Fallback: extract the first SELECT statement from raw text
        sql_match = re.search(r"(SELECT\s+.+?)(?=\n\n|\Z)", raw, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(1).strip().rstrip(";")
            return QueryClassification(
                sql=sql,
                needs_vector=True,
                reason="Fallback SQL extraction — vector enabled as safety net.",
            )
        raise ValueError(f"LLM returned unparseable response: {raw[:300]}") from exc
