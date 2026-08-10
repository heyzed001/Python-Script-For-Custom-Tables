# =========================
# recodes.py
# =========================
# Add any recode function here.
# Rules:
#   - Each function takes a single value (one cell) and returns the recoded label
#   - Always handle NaN/None at the top
#   - Always return a string (the readable label, not a code)
#   - Import this file in your main script: from recodes import *
# =========================

import pandas as pd


# =========================
# REGION RECODE
# =========================
def recode_region(x):
    """
    Groups specific cities into broader regions.
    """
    if pd.isna(x):
        return "Others"

    val = str(x).strip().upper()

    if val == "LAGOS":
        return "Lagos"
    elif val == "IBADAN":
        return "South West"
    elif val in ["BENIN", "ENUGU", "OWERRI"]:
        return "E & S Central"
    elif val in ["KANO", "ABUJA", "KADUNA"]:
        return "North"
    else:
        return "Others"


# =========================
# AGE GROUP RECODE
# =========================
def recode_age_group(x):
    """
    Collapses individual age bands into broader groups.
    """
    if pd.isna(x):
        return "Other"

    x = str(x).strip()

    if x in ["18 - 24 years", "25 - 29 years", "30 - 34 years"]:
        return "18 - 34 years"
    elif x in ["35 - 39 years", "40 - 50 years"]:
        return "35 - 50 years"
    else:
        return "Other"


# =========================
# ADD YOUR RECODES BELOW
# =========================
def recode_outlet_type(x):
    if pd.isna(x): return "Other"
    val = str(x).strip()
    if val in ["Distributor","Wholesaler"]: 
        return "D & W"
    elif val in ["Retailer"]:
        return "Ret Only"
    else:
        return "Other"
    
def recode_perform(x):
    if pd.isna(x): return "Other"
    val = str(x).strip().lower()
    if val in ["excellent","good"]: 
        return "Positive"
    elif val in ["fair"]:
        return "MID"
    elif val in ["very poor","poor"]:
        return "Negative"
    else:
        return "Other"

