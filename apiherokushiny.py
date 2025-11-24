from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import re

app = FastAPI()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

technical_terms = [
    "python", "r", "sql", "machine learning", "estatística", "bi", "data science",
    "power bi", "dashboard", "hadoop", "big data", "etl", "pipeline", "modelagem",
    "regressão", "clusterização", "classificação", "algoritmos", "pyspark", "dados",
    "excel", "analise", "visualização", "tableau", "storytelling", "business", "aws",
    "azure", "google cloud", "governança", "engenharia de dados", "cientista de dados"
]
stopwords = set("""
empresa vaga projeto dados de da do para por na no é que um uma as os como ou se mais menos ter ser toda todo nos nossa nossas novo nova essa este também tempo até bem boa desde junto tudo nosso nossos dias anos desenvolvimento area foco perfil acompanhe acordo
""".split())

class ProfileData(BaseModel):
    perfil_text: str
    vaga_text: str
    skills: str
    portfolio: str

def extract_keywords_prof(text, max_n=10):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9çãõáéíóúâêôü_ ]', ' ', text)
    words = [w for w in text.split() if len(w) > 2 and not w.isdigit()]
    terms = [w for w in words if w not in stopwords]
    tb = {}
    for w in terms:
        tb[w] = tb.get(w, 0) + 1
    sorted_terms = sorted(tb.items(), key=lambda x: -x[1])
    tech_terms = [w for w, _ in sorted_terms if w in technical_terms]
    if len(tech_terms) >= max_n:
        return tech_terms[:max_n]
    extra_terms = [w for w, _ in sorted_terms if w not in technical_terms]
    return (tech_terms + extra_terms)[:max_n]

def calc_compat_prof(profile_keywords, critical_terms):
    match_tech = sum(1 for w in critical_terms if w in technical_terms and w in profile_keywords) * 2
    match_other = sum(1 for w in critical_terms if w not in technical_terms and w in profile_keywords) * 1
    compat = round(100 * (match_tech + match_other) / (2 * len(critical_terms))) if critical_terms else 0
    compat = min(compat, 100)
    return compat

def match_user_skills(skills, technical_terms):
    skills = [s.strip().lower() for s in re.split("[,;\n]+", skills)]
    return [s for s in skills if s in technical_terms]

@app.post("/analyze/")
def analyze(data: ProfileData):
    emb1 = model.encode(data.perfil_text + " " + data.skills + " " + data.portfolio)
    emb2 = model.encode(data.vaga_text)
    sim = float((emb1 @ emb2) / ((emb1 ** 2).sum() ** 0.5 * (emb2 ** 2).sum() ** 0.5))
    sim_percent = round(float(sim * 100), 1)

    termos_criticos = extract_keywords_prof(data.vaga_text, 10)
    perfil_kw = list(set(
        extract_keywords_prof(data.perfil_text, 15) +
        extract_keywords_prof(data.skills, 15) +
        extract_keywords_prof(data.portfolio, 15)
    ))
    compat_terms = calc_compat_prof(perfil_kw, termos_criticos)

    skills_batem = match_user_skills(data.skills, technical_terms)

    return {
        "similarity": sim_percent,
        "termos_criticos": termos_criticos,
        "compat_terms": compat_terms,
        "skills_batem": skills_batem
    }
