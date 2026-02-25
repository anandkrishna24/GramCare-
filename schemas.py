from pydantic import BaseModel
from typing import Dict, List, Optional, Union

class TriageRequest(BaseModel):
    answers: Dict[str, Union[str, int, float]]
    language: str = "en"

class TriageResponse(BaseModel):
    triage_level: str
    reasoning: str
    confidence: str
    home_advice: List[str]
    advice_texts: Optional[List[str]] = []
