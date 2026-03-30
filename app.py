"""
Aarogya AI — Disease Risk Prediction Web App
Models load in background after server starts.
"""
import os, math, pickle, threading
import numpy as np
from flask import Flask, render_template, request, jsonify

# ── sklearn version compatibility patch ──────────────────────────────────────
import sklearn.impute._base as _sib
_orig = _sib.SimpleImputer.transform
def _patched(self, X):
    if not hasattr(self, '_fill_dtype') and hasattr(self, '_fit_dtype'):
        self._fill_dtype = self._fit_dtype
    return _orig(self, X)
_sib.SimpleImputer.transform = _patched

app = Flask(__name__)

# ── Models loaded in background ───────────────────────────────────────────────
MODELS = {}
MODELS_READY = False

def load_models():
    global MODELS, MODELS_READY
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            path = os.path.join(MODEL_DIR, fname)
            with open(path, "rb") as f:
                obj = pickle.load(f)
            MODELS[obj["key"]] = obj
            print(f"✅ {obj['key']} ({len(obj['features'])} features)", flush=True)
    MODELS_READY = True
    print("✅ All models loaded.", flush=True)

# Start loading immediately in background thread
threading.Thread(target=load_models, daemon=True).start()

# ── Gender weights ────────────────────────────────────────────────────────────
GENDER_WEIGHTS = {
    "female": {"breast":1.0,"pcos":1.0,"thyroid":0.9,"heart":0.7,"liver":0.65,"diabetes":0.65,"prostate":0.0},
    "male":   {"prostate":1.0,"heart":0.95,"liver":0.85,"diabetes":0.75,"thyroid":0.6,"breast":0.0,"pcos":0.0}
}

DISEASE_META = {
    "breast":   {"name":"Breast Cancer",  "icon":"🎗️"},
    "pcos":     {"name":"PCOS",            "icon":"🔵"},
    "thyroid":  {"name":"Thyroid Disease", "icon":"🦋"},
    "prostate": {"name":"Prostate Cancer", "icon":"🔷"},
    "heart":    {"name":"Heart Disease",   "icon":"❤️"},
    "diabetes": {"name":"Diabetes",        "icon":"💉"},
    "liver":    {"name":"Liver Disease",   "icon":"🟤"},
}

def risk_level(score):
    if score < 20:   return ("LOW",      "#22c55e", "🟢")
    elif score < 45: return ("MODERATE", "#f59e0b", "🟡")
    elif score < 65: return ("HIGH",     "#ef4444", "🔴")
    else:            return ("CRITICAL", "#dc2626", "🚨")


def build_features(f):
    gender       = f.get("gender","female")
    age          = max(18, int(f.get("age", 35)))
    height_cm    = float(f.get("height", 160))
    weight_kg    = float(f.get("weight", 65))
    waist_in     = float(f.get("waist", 32))
    bmi          = round(weight_kg / ((height_cm/100)**2), 1)

    smoking      = int(f.get("smoking", 0))
    alcohol      = int(f.get("alcohol", 0))
    activity     = int(f.get("activity", 2))
    diet         = int(f.get("diet", 0))
    family       = int(f.get("family_hx", 0))
    stress       = int(f.get("stress", 0))
    sleep_hrs    = float(f.get("sleep_hrs", 7))

    bp_level     = int(f.get("bp_level", 0))
    sugar_level  = int(f.get("sugar_level", 0))
    chol_level   = int(f.get("chol_level", 0))
    known_diab   = int(f.get("known_diabetes", 0))
    known_bp     = int(f.get("known_bp", 0))

    fatigue      = int(f.get("fatigue", 0))
    weakness     = int(f.get("weakness", 0))
    weightgain   = int(f.get("weightgain", 0))
    weightloss   = int(f.get("weightloss", 0))
    fever        = int(f.get("fever", 0))
    nausea       = int(f.get("nausea", 0))
    appetite     = int(f.get("appetite", 0))
    swelling     = int(f.get("swelling", 0))

    chest_pain   = int(f.get("chest_pain", 0))
    chest_tight  = int(f.get("chest_tight", 0))
    palpitations = int(f.get("palpitations", 0))
    breathless   = int(f.get("breathless", 0))
    exercise_cp  = int(f.get("exercise_cp", 0))
    irregular_hb = int(f.get("irregular_hb", 0))

    frequent_urine= int(f.get("frequent_urine", 0))
    excess_thirst = int(f.get("excess_thirst", 0))
    blurred_vision= int(f.get("blurred_vision", 0))
    slow_healing  = int(f.get("slow_healing", 0))
    tingling      = int(f.get("tingling", 0))
    pregnant      = int(f.get("pregnant", 0))

    jaundice     = int(f.get("jaundice", 0))
    abdominal    = int(f.get("abdominal", 0))
    dark_urine   = int(f.get("dark_urine", 0))
    pale_stool   = int(f.get("pale_stool", 0))
    itching      = int(f.get("itching", 0))
    liver_tender = int(f.get("liver_tender", 0))

    urine_less   = int(f.get("urine_less", 0))
    urine_blood  = int(f.get("urine_blood", 0))
    urine_foam   = int(f.get("urine_foam", 0))
    back_pain    = int(f.get("back_pain", 0))
    leg_cramps   = int(f.get("leg_cramps", 0))

    cough        = int(f.get("cough", 0))
    cough_blood  = int(f.get("cough_blood", 0))
    wheezing     = int(f.get("wheezing", 0))
    sore_throat  = int(f.get("sore_throat", 0))

    dark_skin    = int(f.get("dark_skin", 0))
    hairloss     = int(f.get("hairloss", 0))
    pcos_acne    = int(f.get("pcos_acne", 0))
    skin_tags    = int(f.get("skin_tags", 0))
    dry_skin     = int(f.get("dry_skin", 0))

    irreg_cycle  = int(f.get("irreg_cycle", 0))
    heavy_period = int(f.get("heavy_period", 0))
    facial_hair  = int(f.get("facial_hair", 0))
    breast_lump  = int(f.get("breast_lump", 0))
    breast_pain  = int(f.get("breast_pain", 0))
    nipple_disc  = int(f.get("nipple_disc", 0))
    abortions    = int(f.get("abortions", 0))
    marriage_yrs = int(f.get("marriage_yrs", 3))

    thyroid_swell= int(f.get("thyroid_swell", 0))
    cold_intol   = int(f.get("cold_intol", 0))
    heat_intol   = int(f.get("heat_intol", 0))
    voice_change = int(f.get("voice_change", 0))
    radiotherap  = int(f.get("radiotherap", 0))

    urination_diff= int(f.get("urination_diff", 0))
    blood_semen  = int(f.get("blood_semen", 0))
    nocturia     = int(f.get("nocturia", 0))
    pelvic_pain  = int(f.get("pelvic_pain", 0))
    erectile_d   = int(f.get("erectile_d", 0))

    is_male = 1 if gender == "male" else 0

    bp_sys  = 110 + known_bp*20 + bp_level*12 + stress*3 + (age-30)*0.3
    bp_dia  = 70  + known_bp*12 + bp_level*8
    glucose = 85 + known_diab*55 + sugar_level*18 + frequent_urine*10 + excess_thirst*8 + diet*5 + bmi*0.3
    glucose = round(min(400, glucose), 1)
    chol = 150 + chol_level*25 + diet*12 + smoking*15 + (age-30)*0.8 + alcohol*8
    chol = round(min(400, chol), 1)
    hb = round(max(6.0, 14.5 - fatigue*0.6 - weakness*0.5 - (1-is_male)*1.2 - jaundice*0.3), 1)
    h_rate = int(min(200, 68 + breathless*8 + palpitations*10 + fever*5 + chest_pain*4))
    insulin = round(min(250, max(15, 60 + (glucose-85)*0.8 + bmi*0.5)), 1)
    skin_th = int(min(60, 15 + bmi*0.5 + (1-activity/3)*5))
    dpf = round(0.25 + family*0.25 + known_diab*0.3, 2)
    creat   = round(max(0.5, 0.9 + urine_less*0.35 + bp_level*0.1 + (age-30)*0.005), 2)
    egfr    = max(10, int(120 - (creat-0.9)*45 - (age-30)*0.5))
    urea    = round(20 + urine_less*12 + bp_level*6, 1)
    albumin = round(max(2.0, 4.2 - jaundice*0.5 - abdominal*0.15 - urine_foam*0.2), 2)
    tbili   = round(0.5 + jaundice*2.5 + alcohol*0.6 + dark_urine*1.0, 2)
    dbili   = round(tbili*0.35, 2)
    alk_ph  = int(75 + jaundice*65 + alcohol*25 + abdominal*10)
    alt     = int(20 + jaundice*50 + alcohol*35 + liver_tender*20)
    ast     = int(18 + jaundice*45 + alcohol*30 + liver_tender*15)
    t_prot  = round(max(3.5, 7.5 - jaundice*0.6 - alcohol*0.2), 2)
    agr     = round(albumin / max(0.1, t_prot-albumin), 2)
    cp_val  = min(3, chest_pain + chest_tight)
    oldpeak = round(max(0, chest_pain*0.7 + exercise_cp*0.5), 1)
    ca_val  = min(3, int((chest_pain+bp_level)/2))
    fbs     = 1 if glucose > 120 else 0
    restecg = 1 if (bp_level >= 2 or irregular_hb) else 0
    exang   = 1 if exercise_cp >= 2 else 0
    slope   = 1 if chest_pain >= 2 else 0
    thal    = 3 if family else (2 if chest_pain >= 2 else 2)
    tsh = round(max(0.05, 2.5 + cold_intol*4.5 - heat_intol*1.8 + thyroid_swell*1.5), 2)
    if cold_intol and not heat_intol:
        thyroid_fn_hypo = 1; thyroid_fn_eu = 0; thyroid_fn_subhyper = 0; thyroid_fn_subhypo = 0
    elif heat_intol and not cold_intol:
        thyroid_fn_hypo = 0; thyroid_fn_eu = 0; thyroid_fn_subhyper = 1; thyroid_fn_subhypo = 0
    elif tsh > 5:
        thyroid_fn_hypo = 0; thyroid_fn_eu = 0; thyroid_fn_subhyper = 0; thyroid_fn_subhypo = 1
    else:
        thyroid_fn_hypo = 0; thyroid_fn_eu = 1; thyroid_fn_subhyper = 0; thyroid_fn_subhypo = 0
    risk_low  = 1 if not (smoking or family or thyroid_swell or radiotherap) else 0
    risk_inter= 1 - risk_low
    lcavol  = round(max(0.1, 0.3 + urination_diff*0.9 + blood_semen*1.5 + pelvic_pain*0.5), 2)
    gleason = 6 + int(urination_diff >= 2) + int(blood_semen) + int(pelvic_pain >= 2)
    pgg45   = min(100, int(urination_diff*18 + blood_semen*30 + pelvic_pain*12 + family*10))
    lbph    = round(max(-1.4, 0.1 + (age-40)*0.018 + nocturia*0.3), 2)
    svi     = 1 if (blood_semen and urination_diff >= 2) else 0
    lcp     = round(max(-1.4, lcavol - 0.6), 2)
    lweight = round(3.5 + (bmi-25)*0.025 + (age-40)*0.005, 2)
    hip_in  = int(waist_in + 8 + bmi*0.25)
    whr     = round(waist_in / max(1, hip_in), 2)
    lh_val  = round(max(1, 5.0 + irreg_cycle*2.5 + facial_hair*3.0 + (bmi-22)*0.2), 1)
    fsh_val = round(max(1, 8.0 + heavy_period*1.5 + (3-activity)*1.5), 1)
    amh_val = round(max(0.1, 3.5 + irreg_cycle*2.5 + facial_hair*2.0), 1)
    foll_n  = int(max(4, 5 + irreg_cycle*2.5 + facial_hair*2))
    cycle_ri= (irreg_cycle + 2) if gender == "female" else 2
    cycle_len=int(max(3, 5 + irreg_cycle))
    vit_d3  = max(5, 35 - (3-activity)*6 - (1-is_male)*3)
    lump_factor = breast_lump * 3.0 + breast_pain * 0.5 + nipple_disc * 2.0 + family * 1.5
    radius_mean       = round(10.0 + lump_factor*1.2, 2)
    texture_mean      = round(15.0 + lump_factor*1.0, 2)
    perimeter_mean    = round(65.0 + lump_factor*8.0, 2)
    area_mean         = round(320.0 + lump_factor*60.0, 2)
    smoothness_mean   = round(0.08 + lump_factor*0.004, 4)
    compactness_mean  = round(0.06 + lump_factor*0.012, 4)
    concavity_mean    = round(0.04 + lump_factor*0.015, 4)
    concave_pts_mean  = round(0.02 + lump_factor*0.01, 4)
    symmetry_mean     = round(0.17 + lump_factor*0.005, 4)
    frac_dim_mean     = round(0.063 + lump_factor*0.001, 4)
    radius_se         = round(0.25 + lump_factor*0.08, 3)
    texture_se        = round(0.9 + lump_factor*0.15, 3)
    perimeter_se      = round(1.8 + lump_factor*0.6, 3)
    area_se           = round(20.0 + lump_factor*10.0, 2)
    smooth_se         = round(0.006 + lump_factor*0.001, 4)
    compact_se        = round(0.015 + lump_factor*0.005, 4)
    concav_se         = round(0.015 + lump_factor*0.006, 4)
    concavpt_se       = round(0.008 + lump_factor*0.003, 4)
    sym_se            = round(0.018 + lump_factor*0.002, 4)
    fracdim_se        = round(0.003 + lump_factor*0.0003, 5)
    radius_worst      = round(13.0 + lump_factor*2.0, 2)
    texture_worst     = round(22.0 + lump_factor*2.0, 2)
    perimeter_worst   = round(85.0 + lump_factor*12.0, 2)
    area_worst        = round(560.0 + lump_factor*100.0, 2)
    smooth_worst      = round(0.12 + lump_factor*0.008, 4)
    compact_worst     = round(0.20 + lump_factor*0.03, 4)
    concav_worst      = round(0.20 + lump_factor*0.04, 4)
    concavpt_worst    = round(0.10 + lump_factor*0.015, 4)
    sym_worst         = round(0.27 + lump_factor*0.01, 4)
    fracdim_worst     = round(0.075 + lump_factor*0.002, 4)

    return {
        "Pregnancies":              0 if is_male else max(0, pregnant + abortions),
        "Glucose":                  glucose,
        "BloodPressure":            bp_dia,
        "SkinThickness":            skin_th,
        "Insulin":                  insulin,
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                      age,
        "age":      age,
        "sex":      is_male,
        "cp":       cp_val,
        "trestbps": bp_sys,
        "chol":     chol,
        "fbs":      fbs,
        "restecg":  restecg,
        "thalach":  h_rate,
        "exang":    exang,
        "oldpeak":  oldpeak,
        "slope":    slope,
        "ca":       ca_val,
        "thal":     thal,
        "Gender":                    is_male,
        "Total_Bilirubin":           tbili,
        "Direct_Bilirubin":          dbili,
        "Alkaline_Phosphotase":      alk_ph,
        "Alamine_Aminotransferase":  alt,
        "Aspartate_Aminotransferase":ast,
        "Total_Protiens":            t_prot,
        "Albumin":                   albumin,
        "Albumin_and_Globulin_Ratio":agr,
        "lcavol":  lcavol,
        "lweight": lweight,
        "lbph":    lbph,
        "svi":     svi,
        "lcp":     lcp,
        "gleason": gleason,
        "pgg45":   pgg45,
        "radius_mean":           radius_mean,
        "texture_mean":          texture_mean,
        "perimeter_mean":        perimeter_mean,
        "area_mean":             area_mean,
        "smoothness_mean":       smoothness_mean,
        "compactness_mean":      compactness_mean,
        "concavity_mean":        concavity_mean,
        "concave points_mean":   concave_pts_mean,
        "symmetry_mean":         symmetry_mean,
        "fractal_dimension_mean":frac_dim_mean,
        "radius_se":             radius_se,
        "texture_se":            texture_se,
        "perimeter_se":          perimeter_se,
        "area_se":               area_se,
        "smoothness_se":         smooth_se,
        "compactness_se":        compact_se,
        "concavity_se":          concav_se,
        "concave points_se":     concavpt_se,
        "symmetry_se":           sym_se,
        "fractal_dimension_se":  fracdim_se,
        "radius_worst":          radius_worst,
        "texture_worst":         texture_worst,
        "perimeter_worst":       perimeter_worst,
        "area_worst":            area_worst,
        "smoothness_worst":      smooth_worst,
        "compactness_worst":     compact_worst,
        "concavity_worst":       concav_worst,
        "concave points_worst":  concavpt_worst,
        "symmetry_worst":        sym_worst,
        "fractal_dimension_worst":fracdim_worst,
        " Age (yrs)":           age,
        "Weight (Kg)":          weight_kg,
        "Height(Cm) ":          height_cm,
        "Blood Group":          11,
        "Pulse rate(bpm) ":     h_rate,
        "RR (breaths/min)":     int(14 + breathless*3),
        "Hb(g/dl)":             hb,
        "Cycle(R/I)":           cycle_ri,
        "Cycle length(days)":   cycle_len,
        "Marraige Status (Yrs)":marriage_yrs,
        "Pregnant(Y/N)":        pregnant,
        "No. of abortions":     abortions,
        "  I   beta-HCG(mIU/mL)":5.0,
        "II    beta-HCG(mIU/mL)":5.0,
        "FSH(mIU/mL)":          fsh_val,
        "LH(mIU/mL)":           lh_val,
        "FSH/LH":               round(fsh_val/max(0.1, lh_val), 2),
        "Hip(inch)":            hip_in,
        "Waist(inch)":          waist_in,
        "Waist:Hip Ratio":      whr,
        "TSH (mIU/L)":          tsh,
        "AMH(ng/mL)":           amh_val,
        "PRL(ng/mL)":           20.0,
        "Vit D3 (ng/mL)":       vit_d3,
        "PRG(ng/mL)":           0.5,
        "RBS(mg/dl)":           glucose,
        "Weight gain(Y/N)":     weightgain,
        "hair growth(Y/N)":     facial_hair,
        "Skin darkening (Y/N)": dark_skin,
        "Hair loss(Y/N)":       hairloss,
        "Pimples(Y/N)":         1 if pcos_acne >= 2 else 0,
        "Fast food (Y/N)":      1 if diet >= 2 else 0,
        "Reg.Exercise(Y/N)":    1 if activity >= 2 else 0,
        "BP _Systolic (mmHg)":  bp_sys,
        "BP _Diastolic (mmHg)": bp_dia,
        "Follicle No. (L)":     foll_n,
        "Follicle No. (R)":     foll_n,
        "Avg. F size (L) (mm)": 14.0,
        "Avg. F size (R) (mm)": 14.0,
        "Endometrium (mm)":     8.0,
        "Gender_M":             is_male,
        "Smoking_Yes":          smoking,
        "Hx Smoking_Yes":       smoking,
        "Hx Radiothreapy_Yes":  radiotherap,
        "Thyroid Function_Clinical Hypothyroidism":     thyroid_fn_hypo,
        "Thyroid Function_Euthyroid":                   thyroid_fn_eu,
        "Thyroid Function_Subclinical Hyperthyroidism": thyroid_fn_subhyper,
        "Thyroid Function_Subclinical Hypothyroidism":  thyroid_fn_subhypo,
        "Physical Examination_Multinodular goiter":     1 if thyroid_swell >= 1 and voice_change else 0,
        "Physical Examination_Normal":                  1 if not thyroid_swell else 0,
        "Physical Examination_Single nodular goiter-left":  0,
        "Physical Examination_Single nodular goiter-right": 1 if thyroid_swell else 0,
        "Adenopathy_Extensive":  0,
        "Adenopathy_Left":       0,
        "Adenopathy_No":         1 if not thyroid_swell else 0,
        "Adenopathy_Posterior":  0,
        "Adenopathy_Right":      0,
        "Pathology_Hurthel cell":0,
        "Pathology_Micropapillary":0,
        "Pathology_Papillary":   1,
        "Focality_Uni-Focal":    1,
        "Risk_Intermediate":     risk_inter,
        "Risk_Low":              risk_low,
        "T_T1b":0,"T_T2":1,"T_T3a":0,"T_T3b":0,"T_T4a":0,"T_T4b":0,
        "N_N1a":0,"N_N1b":0,"M_M1":0,
        "Stage_II":0,"Stage_III":0,"Stage_IVA":0,"Stage_IVB":0,
        "Response_Excellent":1,"Response_Indeterminate":0,
        "Response_Structural Incomplete":0,
    }


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/lung-xray")
def lung_xray():
    return render_template("lung_xray.html")

@app.route("/ready")
def ready():
    return jsonify({"ready": MODELS_READY, "models": list(MODELS.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_READY:
        return jsonify({"success": False, "error": "Models still loading, please wait a moment and try again."}), 503

    try:
        form     = request.form
        gender   = form.get("gender", "female")
        name     = form.get("name", "Patient").strip() or "Patient"
        feats    = build_features(form)
        weights  = GENDER_WEIGHTS.get(gender, GENDER_WEIGHTS["female"])

        results = []
        total_w = total_wv = 0.0

        for key, mobj in MODELS.items():
            wt = weights.get(key, 0.0)
            if wt == 0.0:
                continue
            X    = np.array([[feats.get(c, 0) for c in mobj["features"]]], dtype=float)
            prob = mobj["model"].predict_proba(X)[0][1]
            pct  = round(prob * 100, 1)
            lvl, col, em = risk_level(pct)
            meta = DISEASE_META.get(key, {"name": key, "icon": "🔬"})
            results.append({
                "key": key, "name": meta["name"], "icon": meta["icon"],
                "risk": pct, "level": lvl, "color": col, "emoji": em, "weight": wt,
            })
            total_wv += pct * wt
            total_w  += wt

        results.sort(key=lambda r: r["risk"], reverse=True)
        overall = round(total_wv / max(1, total_w), 1)
        o_lvl, o_col, o_em = risk_level(overall)

        return jsonify({
            "success": True, "patient": name, "gender": gender,
            "overall": overall, "o_level": o_lvl, "o_color": o_col, "o_emoji": o_em,
            "results": results,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
