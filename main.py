from fastapi import FastAPI, HTTPException
from .schemas import TriageRequest, TriageResponse
from .logic import QUESTIONS, check_red_flags, build_summary, classify, HOME_ADVICE_LIBRARY

app = FastAPI(title="Pediatric Triage API")

@app.get("/questions")
def get_questions():
    return QUESTIONS

@app.post("/triage", response_model=TriageResponse)
def perform_triage(request: TriageRequest):
    answers = request.answers
    
    # 1. Check Red Flags
    red_flag = check_red_flags(answers)
    if red_flag:
        res = red_flag
    else:
        # 2. AI Classification
        summary = build_summary(answers)
        res = classify(summary)
    
    # Enrich with translated advice texts
    advice_texts = []
    lang = request.language if request.language in ["en", "ml"] else "en"
    for key in res.get("home_advice", []):
        advice_item = HOME_ADVICE_LIBRARY.get(key)
        if advice_item:
            advice_texts.append(advice_item[lang])
    
    res["advice_texts"] = advice_texts
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
