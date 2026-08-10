# =========================
# IMPORTS
# =========================
import pandas as pd
import pyreadstat as py
import re
import numpy as np
# =========================
# LOAD DATA
# =========================
df, meta = py.read_sav("glowCC.sav")
df.columns = df.columns.str.strip()

# =========================
# MULTI MAPS
# =========================
multi_maps = {
    "C1a": {
        "1": "Attachment",
        "2": "Wig",
        "3": "Weave-on",
        "4": "Wool",
        "5": "Clip On",
        "6": "Crochet",
        "98": "Others",
        "99": "None"
    },
    "C1b": {
        "1": "Attachment",
        "2": "Wig",
        "3": "Weave-on",
        "4": "Wool",
        "5": "Clip On",
        "6": "Crochet",
        "98": "Others",
        "99": "None"
    },
    "QSf":{
        "1": "Braiding",
        "2": "Wig Making or Revamping",
        "3": "Weave-on Fixing",
        "4": "Clip On",
        "5": "Crochet Installation",
        "6": "Others (Please specify)",
        "10": "Cutting",
        "11": "Styling",
        "12": "Chemical Treatments",
        "13": "Relaxing or Retouching",
        "20": "Bridal styling services",
        "21": "Special occasion styling services",
        "22": "Hair care Consultations",
        "23": "Pedicure",
        "24": "Manicure",
        "30": "Makeup Services",
        "31": "Nail Services",
        "32": "Facial",
        "33": "Skin Consultations",
        "40": "Sell Hair Care products like shampoos, conditioner, etc",
        "41": "Sell Skin products like creams, serums, etc",
        "42": "Sell Hair extensions like Attachments, Crochets, Wigs, etc.",
        "43": "Sell Perfumes",
        "44": "Sell devices like Hair dryers, Straighteners, etc"
    },
    "BAU3a.9":{
        "1" :"Television",
        "2": "Radio",
        "3": "Bus Branding",
        "4" :"Newspaper/Magazine",
        "5":"Electronic Board/Billboards",
        "6" :"Point of Sales Materials",
        "7" :"Facebook",
        "8": "Twitter",
        "9": "Salon / Spa",
        "10": "Youtube",
        "11" :"Instagram",
        "12": "Events/Market Promos/Fashion Show",
        "13" :"University/NYSC promotions",
        "14" :"Brand Ambassadors",
        "15" :"Wall / Building /Shop branding",
        "16" :"Open Market Gate",
        "17" :"BBN (Big Brother Naija)",
        "18" :"Others (please specify)",
        "19" :"Can’t recall",
    }
}

# =========================
# APPLY LABELS
# =========================
def apply_labels(df, meta, var):

    label_name = meta.variable_to_label.get(var)
    labels_dict = meta.value_labels.get(label_name)

    col = f"{var}_label"

    if labels_dict is not None:
        df[var] = df[var].astype("Int64")
        df[col] = df[var].map(labels_dict)
    else:
        df[col] = df[var].astype(str)

    return df, labels_dict


# =========================
# COLUMN % FUNCTION
# =========================
def to_col_percent(table):

    col_sum = table.sum(axis=0)
    table = table.div(col_sum, axis=1) * 100

    return table.round(1)


# =========================
# SINGLE TABLE (COL %)
def single_table(df, meta, row, var):

    df, labels_dict = apply_labels(df, meta, var)

    col_dummy = pd.get_dummies(df[f"{var}_label"])

    table = col_dummy.groupby(df[f"{row}_label"]).sum()

    if labels_dict is not None:
        table = table.reindex(columns=list(labels_dict.values()), fill_value=0)

    # ✅ FIXED COLUMN %
    col_base = table.sum(axis=0)

    table = table.div(col_base, axis=1) * 100
    table = table.fillna(0).round(1)

    table.columns = pd.MultiIndex.from_product([[var], table.columns])

    return table

# =========================
# MULTI TABLE (COL %)
# =========================
def multi_table(df, row, var, multi_map):

    split_df = df[var].astype(str).str.get_dummies(sep=" ")
    split_df = split_df.rename(columns=multi_map)
    split_df = split_df.reindex(columns=list(multi_map.values()), fill_value=0)

    table = split_df.groupby(df[f"{row}_label"]).sum()

    table = to_col_percent(table)

    table.columns = pd.MultiIndex.from_product([[var], table.columns])

    return table


# =========================
# MASTER FUNCTION
# =========================
def create_table(df, meta, row, single_cols, multi_cols, multi_maps,header_dict=None):

    df, row_labels = apply_labels(df, meta, row)

    tables = []

    for var in single_cols:
        tables.append(single_table(df, meta, row, var))

    for var in multi_cols:
        var_map = multi_maps.get(var, {})
        tables.append(multi_table(df, row, var, var_map))

    final_table = pd.concat(tables, axis=1)

    if row_labels is not None:
        final_table = final_table.reindex(list(row_labels.values()))
    
    # BASE ROW
    # =========================
    base_row_vals = final_table.replace("-", 0).astype(float).sum(axis=0)
    base_row = pd.DataFrame([base_row_vals], index=["Base"])

    # Keep the custom header on the base row index
    base_row.index.name = final_table.index.name

    # CONCAT
    final_table = pd.concat([base_row, final_table])

    # =========================
    # TOTAL COLUMN (LEFT)
    # =========================
    # We grab the first group to calculate the overall Total
    first_group = final_table.columns.get_level_values(0)[0]
    group_df = final_table[first_group]

    total_col = group_df.replace("-", 0).astype(float).sum(axis=1)

    # Convert to MultiIndex column to match the rest of the table
    total_df = pd.DataFrame(total_col)
    total_df.columns = pd.MultiIndex.from_tuples([("", "Total")])

    # Insert at the leftmost position
    final_table = pd.concat([total_df, final_table], axis=1)
    
    # Final check: Ensure the index axis carries the label
    final_table.rename_axis(index=row_label, inplace=True)


    return final_table


# =========================
def multi_row_table(df, meta, row_var, col_vars, multi_maps):
    """
    Multi-response as ROW with support for:
    ✔ single columns (fixed)
    ✔ multi columns
    ✔ correct column %
    """

    # =========================
    # ROW → MULTI RESPONSE
    # =========================
    row_map = multi_maps.get(row_var, {})

    split_row = df[row_var].astype(str).str.get_dummies(sep=" ")
    split_row = split_row.rename(columns=row_map)
    split_row = split_row.reindex(columns=list(row_map.values()), fill_value=0)

    tables = []

    # =========================
    # LOOP THROUGH COLUMNS
    # =========================
    for col in col_vars:

        # =====================
        # SINGLE COLUMN (FIXED)
        # =====================
        if col not in multi_maps:

            df, labels_dict = apply_labels(df, meta, col)

            # 👉 convert to dummy (CRITICAL FIX)
            col_dummy = pd.get_dummies(df[f"{col}_label"])

            # ensure all labels exist
            if labels_dict is not None:
                col_dummy = col_dummy.reindex(
                    columns=list(labels_dict.values()),
                    fill_value=0
                )

            # 👉 TRUE cross-tab
            temp = split_row.T.dot(col_dummy)

            # 👉 COLUMN BASE (correct)
            col_base = col_dummy.sum(axis=0)
            
            col_base = col_base.replace(0, np.nan)

            temp = temp.div(col_base, axis=1) * 100
            temp = temp.fillna(0).round(1)

        # =====================
        # MULTI COLUMN
        # =====================
        else:

            col_map = multi_maps[col]

            split_col = df[col].astype(str).str.get_dummies(sep=" ")
            split_col = split_col.rename(columns=col_map)
            split_col = split_col.reindex(
                columns=list(col_map.values()),
                fill_value=0
            )

            # 👉 TRUE cross-tab
            temp = split_row.T.dot(split_col)

            # 👉 COLUMN BASE (correct)
            col_base = split_col.sum(axis=0)

            col_base = col_base.replace(0, np.nan)

            temp = temp.div(col_base, axis=1) * 100
            temp = temp.fillna(0).round(1)

        # =====================
        # ADD HEADER
        # =====================
        temp.columns = pd.MultiIndex.from_product([[col], temp.columns])

        tables.append(temp)

    # =========================
    # MERGE ALL TABLES
    # =========================
    final_table = pd.concat(tables, axis=1)

    return final_table


def spss_multi_table(df, meta, prefix, col_vars, multi_maps, specific_cols=None):
    # =========================
    # 1. FIND MULTI COLUMNS
    # =========================
    if specific_cols:
        relevant_cols = specific_cols
    else:
        regex_pattern = re.compile(f"^{prefix}[._]\\d+$")
        relevant_cols = [c for c in df.columns if regex_pattern.match(c)]

    if not relevant_cols:
        print(f"No columns found for {prefix}")
        return None

    extracted_map = {}
    main_question = ""

    # =========================
    # 2. EXTRACT LABELS
    # =========================
    for c in relevant_cols:
        label = meta.column_names_to_labels.get(c, "")
        if label:
            if ":" in label:
                parts = label.split(":", 1)
                if not main_question:
                    main_question = parts[0].strip()
                option_text = parts[1].strip()
            elif "?" in label:
                parts = label.split("?", 1)
                if not main_question:
                    main_question = parts[0].strip() + "?"
                option_text = parts[1].strip()
            else:
                option_text = label

            extracted_map[c] = option_text

    # =========================
    # 3. BUILD MERGED STRING (ONLY YES)
    # =========================
    merged_name = f"{prefix}_Merged"

    def get_selected(row):
        selected = []
        for col in relevant_cols:
            val = str(row[col]).strip().lower()
            
            # 🔥 FIX: Ignores zeros and blanks, catching 1, 2, 3, 4, 'yes', the label itself, etc.
            if val not in ["0", "0.0", "no", "false", "nan", "none", ""]:
                selected.append(col)
                
        return " ".join(selected)

    df[merged_name] = df[relevant_cols].apply(get_selected, axis=1)

    # ❌ DO NOT remove empty responses here. It breaks the column base for percentages.

    # =========================
    # 4. ADD TO MULTI MAPS
    # =========================
    multi_maps = multi_maps.copy() 
    multi_maps[merged_name] = extracted_map

    # =========================
    # 5. CALL YOUR MULTI ROW FUNCTION
    # =========================
    return multi_row_table(
        df,
        meta,
        row_var=merged_name,
        col_vars=col_vars,
        multi_maps=multi_maps
    )

custom_headers = {
   
}
# =========================
# EXAMPLE USAGE
# =========================
"""
table1 = create_table(
    df,
    meta,
    row="Q2.C3dc",
    single_cols=["region","Gender","Age_cal",],
    multi_cols=["C1a"],
    multi_maps=multi_maps
)

print("\n===== COLUMN % TABLE (NORMAL) =====\n")


table2 = multi_row_table(
    df,
    meta,
    row_var="BAU3a.9",
    col_vars=["region","Age_cal","SALON_TYPE","C1b"],
    multi_maps=multi_maps,
)

print("\n===== COLUMN % TABLE (MULTI ROW) =====\n")


table3 = spss_multi_table(
    df, 
    meta, 
    prefix="BAU3a.9", 
    col_vars=["region","Age_cal","SALON_TYPE","C1b"], 
    multi_maps=multi_maps
)
print(table1)"""
