# 🩺 AI Patient Health Analyzer

> **Educational prototype. Not a medical device. It does not diagnose and must not be used for real patient care.**
> Use only synthetic or fully de-identified data. Every output needs review by a qualified clinician.

**Scenario:** a hospital wants to improve diagnosis and treatment planning. This app analyses patient information, identifies potential
health risks, suggests follow-up points for the clinician, and evaluates the benefits and limitations of AI-assisted healthcare.

## What it does
**1. Single patient**
- Calculates, with plain readable code (no AI): **NEWS2** early warning score, **BMI** (selectable cut-offs), **blood pressure** category,
  **glucose / HbA1c** category, **lipid** categories, **eGFR** (CKD-EPI 2021), **anaemia** screen, smoking and family-history flags,
  combined cardiometabolic risk, and **red-flag symptoms** found in free text (with simple negation, so "denies chest pain" is not flagged).
- Every flag shows its **rule, value and threshold**, and non-prescriptive follow-up points "for clinician consideration".
- Lists **data gaps** and catches **implausible values** (for example a temperature typed in Fahrenheit).
- Optional **AI summary** (Groq): it may use only the computed findings. Its output is checked: finding references must exist, numbers must
  come from the data, and sentences about medicines or doses are withheld.

**2. Hospital cohort (CSV upload and manual entry together)**
- Build one patient list from three sources: **upload a CSV**, **add patients by hand** (analyze a patient in tab 1, then click
  *Add this patient to the hospital cohort*), and/or add 60 **synthetic** patients.
- The combined list gives a **review order** (with a Source column), NEWS2 band chart, priority chart and **risk patterns** across the group.
- Remove hand-entered patients at any time. Open any patient for the full analysis.

**3. Benefits and limitations**
- A structured evaluation of AI-assisted healthcare (benefits, limitations, mitigations in this prototype, what a hospital still needs,
  and how to evaluate such a tool before use).

**4. Rules and self-check**
- All thresholds with their sources, and a self-check that reproduces published examples (for example NEWS2 worked examples and eGFR).

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Python packages (streamlit, pandas, groq) |
| `.gitignore` | Keeps secrets and patient data out of GitHub |

## Groq API key (optional)
Everything except the AI summary works without a key.
1. Create a free key at https://console.groq.com (API Keys → Create API Key).
2. On Streamlit Community Cloud: app **Settings → Secrets**, paste one line with straight quotes:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Save and **Reboot app**.

Never put the key in `app.py`. Only send synthetic or de-identified data to the AI.

## Try it
- Tab 1: choose **Example B: acute deterioration**, click **Analyze patient** (NEWS2 16, urgent). Try Examples A and C too.
- Tab 2: click **Create a synthetic cohort**. One patient has a temperature entered in Fahrenheit to show the data check.
- Tab 4: click **Run self-check**.

## CSV columns (Tab 2)
`patient_id, age, sex, sbp, dbp, hr, rr, spo2, temp_c, on_oxygen, consciousness, height_cm, weight_kg, glucose_fasting_mgdl, hba1c_pct,
total_chol_mgdl, ldl_mgdl, hdl_mgdl, triglycerides_mgdl, creatinine_mgdl, hemoglobin_gdl, smoker, known_diabetes, known_hypertension,
known_ckd, family_history_cvd, symptoms`. Download the template inside the app. Units: mmHg, per minute, %, degrees C, cm, kg, mg/dL (glucose,
lipids, creatinine), g/dL.

## Key limitations (see tab 3 for the full evaluation)
- Not clinically validated; thresholds are guideline categories, not diagnoses. A single snapshot hides trends.
- No medicines, pregnancy, children, NEWS2 Scale 2, imaging or 10-year risk scores.
- Rules and tools can be biased by the populations they came from (for example BMI cut-offs; pulse oximeters can overestimate oxygen
  levels in people with darker skin).
- Language models can invent or distort information, so the AI only summarises computed findings and is filtered.
- Real use needs privacy and legal review, regulatory approval, local validation, governance and clinician training.

## Sources for the rules
Royal College of Physicians NEWS2 (2017) · ACC/AHA 2017 blood pressure guideline · American Diabetes Association Standards of Care ·
NKF CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021) · WHO and Asia-Pacific BMI classifications · NCEP ATP III lipid categories ·
WHO haemoglobin thresholds. Check current local guidelines, as thresholds change.

## Run on your computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```
