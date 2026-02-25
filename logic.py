import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# CONFIG & MODEL LOADING
# -----------------------------
MODEL_NAME = "google/medgemma-4b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

# Global instances (initialized on first use or app start)
tokenizer = None
model = None

def get_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    return tokenizer, model

# -----------------------------
# DATA STRUCTURES
# -----------------------------

# -----------------------------
# QUESTION DATA STRUCTURE
# -----------------------------
QUESTIONS = {
    "General": {
        "Q1": {
            "en": "What is the child’s age?", 
            "ml": "കുട്ടിയുടെ വയസ് എത്രയാണ്?", 
            "type": "number", "min": 6, "max": 12,
            "context": lambda v: f"The child is {v} years old."
        },
        "Q2": {
            "en": "How long has the problem been present?", 
            "ml": "ഈ പ്രശ്നം എത്ര ദിവസമായി തുടരുന്നു?", 
            "type": "radio", 
            "options": {
                "< 1 day": {"en": "< 1 day", "ml": "1 ദിവസത്തിൽ താഴെ"},
                "1–2 days": {"en": "1–2 days", "ml": "1-2 ദിവസം"},
                "3+ days": {"en": "3+ days", "ml": "3 ദിവസത്തിലധികം"}
            },
            "context": {
                "< 1 day": "The symptoms started recently (less than 24 hours ago).",
                "1–2 days": "The symptoms have been present for 1 to 2 days.",
                "3+ days": "The symptoms have persisted for more than 3 days."
            }
        },
        "Q3": {
            "en": "Is the child unusually drowsy, confused, or not responding normally?", 
            "ml": "കുട്ടി അസാധാരണമായി ഉറക്കമുള്ളതോ പ്രതികരിക്കാത്തതോ ആണോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "The child is showing signs of altered consciousness, unusual drowsiness, or confusion.",
                "No": "The child is alert and responding normally to surroundings."
            }
        },
        "Q4": {
            "en": "Is the child able to drink and keep fluids down?", 
            "ml": "കുട്ടിക്ക് വെള്ളം കുടിക്കാനും നിലനിർത്താനും കഴിയുന്നുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child is able to tolerate oral fluids and maintain hydration.",
                "No": "The child is unable to drink or keep any fluids down, risking dehydration."
            }
        },
        "Q5": {
            "en": "Does the child have asthma, diabetes, heart disease, or other chronic illness?", 
            "ml": "കുട്ടിക്ക് ആസ്ത്മ, പ്രമേഹം, ഹൃദ്രോഗം തുടങ്ങിയ ദീർഘകാല രോഗങ്ങളുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child has a pre-existing chronic medical condition (e.g., asthma, diabetes).",
                "No": "The child has no known chronic underlying illnesses."
            }
        },
    },
    "Cold / Cough / Fever": {

        "Q6": {
            "en": "How does the fever feel?", 
            "ml": "പനി എങ്ങനെ തോന്നുന്നു?", 
            "type": "radio", 
            "options": {
                "Warm but child active": {"en": "Warm but child active", "ml": "ചെറിയ പനി, കുട്ടി ഉന്മേഷവാനാണ്"},
                "Hot and uncomfortable": {"en": "Hot and uncomfortable", "ml": "ശരീരം നന്നായി ചൂടുണ്ട്, ആസ്വസ്ഥതയുണ്ട്"},
                "Very hot and child weak": {"en": "Very hot and child weak", "ml": "കഠിനമായ പനി, കുട്ടി വളരെ അവശനാണ്"}
            },
            "context": {
                "Warm but child active": "Mild fever with normal activity.",
                "Hot and uncomfortable": "Moderate fever.",
                "Very hot and child weak": "High fever affecting the child."
            }
        },
        "Q7": {
            "en": "Has the fever lasted more than 3 days?", 
            "ml": "പനി 3 ദിവസത്തിലധികമായി തുടരുന്നുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The fever has been prolonged, lasting more than 72 hours.",
                "No": "The fever is recent and has lasted less than 3 days."
            }
        },
        "Q8": {
            "en": "Is there a rash on the body?", 
            "ml": "ശരീരത്തിൽ ചൊറിച്ചിലോ ചർമ്മത്തിൽ പാടുകളോ ഉണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "A new skin rash or spots have appeared on the child's body.",
                "No": "There is no visible rash on the skin."
            }
        },
        "Q9": {
            "en": "Is the child unable to bend the neck forward?", 
            "ml": "കുട്ടിക്ക് കഴുത്ത് മുന്നോട്ട് കുനിക്കാനാകുന്നില്ലേ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "The child is experiencing neck stiffness (meningismus), unable to touch chin to chest.",
                "No": "The child has normal neck mobility."
            }
        },
    },
    "Stomach Pain / Diarrhea / Vomiting": {
        "Q10": {
            "en": "How many times has the child vomited in the last 24 hours?", 
            "ml": "കഴിഞ്ഞ 24 മണിക്കൂറിൽ എത്ര തവണ ഛർദ്ദിച്ചു?", 
            "type": "radio", 
            "options": {
                "None": {"en": "None", "ml": "ഒന്നുമില്ല"},
                "1-3": {"en": "1-3", "ml": "1-3 തവണ"},
                "4+": {"en": "4+", "ml": "4 തവണയിൽ കൂടുതൽ"}
            },
            "context": {
                "None": "The child has not vomited in the last 24 hours.",
                "1-3": "The child has vomited 1 to 3 times recently.",
                "4+": "The child is experiencing frequent/excessive vomiting (4 or more times)."
            }
        },
        "Q11": {
            "en": "Is there blood in vomit or stool?", 
            "ml": "ഛർദ്ദിയിലോ മലത്തിലോ രക്തം ഉണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "Visible blood is present in the child's vomit or bowel movements.",
                "No": "There is no blood observed in vomit or stool."
            }
        },
        "Q12": {
            "en": "Is the stomach pain severe and constant?", 
            "ml": "വയറുവേദന കഠിനവും സ്ഥിരവുമാണോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child is reporting intense, continuous abdominal pain.",
                "No": "Abdominal pain is absent or only mild/intermittent."
            }
        },
        "Q13": {
            "en": "Has the child passed urine in the last 8 hours?", 
            "ml": "കഴിഞ്ഞ 8 മണിക്കൂറിൽ കുട്ടി മൂത്രമൊഴിച്ചിട്ടുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child has normal urine output.",
                "No": "The child has not urinated for over 8 hours, indicating potential dehydration."
            }
        },
    },
    "Breathing Problem": {
        "Q14": {
            "en": "Is the child breathing faster than usual?", 
            "ml": "കുട്ടി സാധാരണയേക്കാൾ വേഗത്തിൽ ശ്വസിക്കുന്നുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child is showing tachypnea (increased breathing rate).",
                "No": "The child's breathing rate is within the normal range."
            }
        },
        "Q15": {
            "en": "Is the chest pulling in while breathing?", 
            "ml": "ശ്വസിക്കുമ്പോൾ നെഞ്ച് ഉള്ളിലേക്ക് വലിക്കപ്പെടുന്നുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "The child has chest retractions (respiratory distress), where the skin pulls in around the ribs.",
                "No": "The child is breathing easily without visible retractions."
            }
        },
        "Q16": {
            "en": "Are the lips or face turning bluish?", 
            "ml": "ചുണ്ടുകളോ മുഖമോ നീല നിറത്തിലാകുന്നുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "Cyanosis is present: The child's lips or face have a blue tint, indicating low oxygen.",
                "No": "The child has normal skin/lip coloration."
            }
        },
        "Q17": {
            "en": "Is the child unable to speak full sentences due to breathlessness?", 
            "ml": "ശ്വാസം മുട്ടലാൽ കുട്ടിക്ക് പൂർണ്ണ വാചകം പറയാനാകുന്നില്ലേ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child is showing severe breathlessness, unable to speak in full sentences.",
                "No": "The child can speak normally without significant shortness of breath."
            }
        },
    },
    "Body Pain / Headache": {
        "Q18": {
            "en": "Is the headache or body pain severe ?", 
            "ml": "തലവേദനയോ ശരീരവേദനയോ കൂടുതലാണോ ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child is in severe headache or body pain.",
                "No": "The child is experiencing mild to moderate pain."
            }
        },
        "Q19": {
            "en": "Did the child have a head injury recently?", 
            "ml": "കുട്ടിക്ക് അടുത്തിടെ തലക്ക് പരിക്കുണ്ടായിട്ടുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "There is a history of recent trauma or injury to the head.",
                "No": "There has been no recent head injury."
            }
        },
        "Q20": {
            "en": "Is there repeated vomiting with headache?", 
            "ml": "തലവേദനയോടൊപ്പം ആവർത്തിച്ച ഛർദ്ദിയുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "context": {
                "Yes": "The child has a headache accompanied by persistent vomiting.",
                "No": "The headache is not associated with vomiting."
            }
        },
    },
    "Critical Red-Flags": {
        "Q21": {
            "en": "Has the child had a seizure?", 
            "ml": "കുട്ടിക്ക് അപസ്മാരം ഉണ്ടായിട്ടുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "The child has experienced a seizure or convulsion.",
                "No": "The child has had no seizures."
            }
        },
        "Q22": {
            "en": "Has the child fainted or become unconscious?", 
            "ml": "കുട്ടി ബോധരഹിതനായിട്ടുണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "The child has experienced loss of consciousness or fainting.",
                "No": "The child has remained conscious throughout."
            }
        },
        "Q23": {
            "en": "Is there a severe injury or heavy bleeding?", 
            "ml": "ഗുരുതരമായ പരിക്കോ രക്തസ്രാവമോ ഉണ്ടോ?", 
            "type": "radio", 
            "options": {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}}, 
            "is_critical": True,
            "context": {
                "Yes": "The child has sustained a major injury or is actively bleeding heavily.",
                "No": "There is no severe injury or heavy bleeding noted."
            }
        },
    }
}
# -----------------------------
# HOME CARE ADVICE LIBRARY
# -----------------------------
HOME_ADVICE_LIBRARY = {
    "REST": {
        "en": "Ensure the child gets adequate rest.",
        "ml": "കുട്ടിക്ക് ആവശ്യത്തിന് വിശ്രമം നൽകുക."
    },
    "FLUIDS": {
        "en": "Encourage frequent intake of clean fluids like water or coconut water.",
        "ml": "വെള്ളം, ഇളനീർ തുടങ്ങിയ പാനീയങ്ങൾ ധാരാളം നൽകുക."
    },
    "LIGHT_DIET": {
        "en": "Provide light, easily digestible food.",
        "ml": "ലഘുവായതും എളുപ്പത്തിൽ ദഹിക്കുന്നതുമായ ഭക്ഷണം നൽകുക."
    },
    "HYGIENE": {
        "en": "Maintain proper hand hygiene to prevent spread of infection.",
        "ml": "അണുബാധ പടരാതിരിക്കാൻ കൈകൾ വൃത്തിയായി സൂക്ഷിക്കുക."
    },
    "MONITOR_SYMPTOMS": {
        "en": "Monitor symptoms closely for any worsening.",
        "ml": "ലക്ഷണങ്ങൾ കൂടുന്നുണ്ടോ എന്ന് ശ്രദ്ധാപൂർവ്വം നിരീക്ഷിക്കുക."
    },
    "TEMPERATURE_CHECK": {
        "en": "Check temperature periodically if fever is present.",
        "ml": "പനി ഉണ്ടെങ്കിൽ കൃത്യസമയത്ത് താപനില പരിശോധിക്കുക."
    }
}

# -----------------------------
# HARD RED FLAG CHECK
# -----------------------------
def check_red_flags(answers):
    critical_data = {
        "triage_level": "RED",
        "reasoning": "Immediate medical attention required for life-threatening symptoms flagged by critical clinical rules.",
        "confidence": "High (Rule-based Override)",
        "home_advice": []
    }
    
    # Rule 1: Direct Critical Questions (Q21, Q22, Q23)
    if any(answers.get(q) == "Yes" for q in ["Q21", "Q22", "Q23"]):
        return critical_data
    # Rule 2: Q3 (Not responding)
    if answers.get("Q3") == "Yes":
        return critical_data
    # Rule 3: Q9 (Neck Stiffness standalone)
    if answers.get("Q9") == "Yes":
        return critical_data
    # Rule 4: Q11 (Blood in vomit/stool)
    if answers.get("Q11") == "Yes":
        return critical_data
    # Rule 5: Q15/Q16 (Chest pulling / Bluish)
    if answers.get("Q15") == "Yes" or answers.get("Q16") == "Yes":
        return critical_data
    # Rule 6: Head injury + Vomiting (Q19 + Q20)
    if answers.get("Q19") == "Yes" and (answers.get("Q20") == "Yes" or answers.get("Q10", "") == "4+"):
        return critical_data
    
    # Rule 7: Chronic illness + Breathing distress (Q5 + Q14)
    if answers.get("Q5") == "Yes" and answers.get("Q14") == "Yes":
        return critical_data
    
    # Rule 8: Dehydration Risk (No urine + Vomiting) (Q13 + Q10)
    if answers.get("Q13") == "No" and answers.get("Q10", "None") != "None":
        return critical_data
        
    # Rule 9: Severe Pain + Fever Combination (Q18 + Q6)
    if answers.get("Q18") == "Yes" and answers.get("Q6") is not None:
        return critical_data
        
    return None

# -----------------------------
# BUILD STRUCTURED SUMMARY 
# -----------------------------
def build_summary(answers):
    summary = "Pediatric Clinical Assessment (Age 6-12):\n"
    for cat, qs in QUESTIONS.items():
        cat_summary = ""
        for q_id, q_data in qs.items():
            val = answers.get(q_id)
            if val is not None:
                # Use context mapping if available
                if "context" in q_data:
                    if callable(q_data["context"]):
                        desc = q_data["context"](val)
                    elif isinstance(q_data["context"], dict):
                        desc = q_data["context"].get(val, f"{q_data['en']}: {val}")
                    else:
                        desc = f"{q_data['en']}: {val}"
                else:
                    desc = f"{q_data['en']}: {val}"
                cat_summary += f"- {desc}\n"
        
        if cat_summary:
            summary += f"\n### {cat}\n{cat_summary}"
    return summary

# -----------------------------
# MEDGEMMA CLASSIFICATION & EXTRACTION
# -----------------------------
REQUIRED_KEYS = {"triage_level", "reasoning", "confidence", "home_advice"}
VALID_ADVICE = {
    "REST",
    "FLUIDS",
    "LIGHT_DIET",
    "HYGIENE",
    "MONITOR_SYMPTOMS",
    "TEMPERATURE_CHECK"
}
VALID_TRIAGE = {"RED", "YELLOW", "GREEN"}

def extract_json_response(response_text: str):
    """
    Safely extract first valid JSON object using balanced brace counting.
    """
    # 1️⃣ Remove markdown wrappers if present
    cleaned = response_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # 2️⃣ Find first balanced JSON block
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    brace_count = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            brace_count += 1
        elif cleaned[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            json_str = cleaned[start:i+1]
            break
    else:
        raise ValueError("Incomplete JSON object")

    return json.loads(json_str)

def classify(summary_text):
    tokenizer, model = get_model()
    prompt = f"""<start_of_turn>user
You are an expert pediatric triage assistant. 
Analyze the following clinical observations for a child aged 6-12 and classify the triage level.

URGENCY LEVELS:
RED – Emergency: Immediate hospital/ER required. Life-threatening or unstable symptoms.
YELLOW – Urgent: Visit a doctor/clinic within 24 hours. Symptoms are worsening but currently stable.
GREEN – Observation: Home care and monitoring. Minor symptoms.

INSTRUCTIONS:
1. Carefully review the "Clinical Observations" provided.
2. Provide your findings only in a valid JSON format.
3. Select the most relevant advice keys from the ADVICE_LIBRARY below (select 2-3 items).

ADVICE_LIBRARY:
- REST: Ensure the child gets adequate rest.
- FLUIDS: Encourage frequent intake of clean fluids.
- LIGHT_DIET: Provide light, easily digestible food.
- HYGIENE: Maintain proper hand hygiene.
- MONITOR_SYMPTOMS: Monitor symptoms closely for any worsening.
- TEMPERATURE_CHECK: Check temperature periodically if fever is present.

Return format:
{{
  "triage_level": "RED/YELLOW/GREEN",
  "reasoning": "A concise clinical explanation focusing on the severity and combination of symptoms provided.",
  "confidence": "High/Medium/Low",
  "home_advice": ["KEY1", "KEY2"]
}}

Clinical Observations:
{summary_text}<end_of_turn>
<start_of_turn>model
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Try robust JSON extraction
    try:
        res = extract_json_response(response)
        # Confidence auto-bump for RED
        if res.get("triage_level") == "RED":
            res["confidence"] = "High (Model + Structured Assessment)"
        return res
    except Exception as e:
        # Fallback if AI fails JSON        
        return {
            "triage_level": "YELLOW",
            "reasoning": f"AI analysis error. Precautionary triage applied.",
            "confidence": "Low",
            "home_advice": []
        }