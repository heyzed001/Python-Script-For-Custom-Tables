# =========================
# IMPORTS
# =========================
import pandas as pd
import pyreadstat as py
import re
import numpy as np
from pathlib import Path


File_path=Path.home()/"Desktop"/"Python check"/"DATASET"/"Spss"
retail_path=File_path/"Retail_Cleaned.sav"


#IT RECODE NOTHING SPECIAL
# =========================
# LOAD DATA
# =========================
df, meta = py.read_sav(retail_path)
df.columns = df.columns.str.strip()
# =========================
# MULTI MAPS
# =========================
custom_headers = {
   
}

multi_maps = {
    "OBS4":{
        "0":"No",
        "1":"Yes"
    }
}

def apply_recode(df, meta, source_var, new_var, recode_func, label=None):
    """
    Applies a recode function to an existing column and creates a new column.
    The new column is registered in meta so get_var_label() and all table
    functions work with it exactly like any other variable.

    Parameters
    ----------
    df          : your dataframe
    meta        : pyreadstat meta object
    source_var  : existing column to recode from  (e.g. "region")
    new_var     : name for the new recoded column  (e.g. "region_grp")
    recode_func : function from recodes.py         (e.g. recode_region)
    label       : question label shown in tables.
                  If None, uses the label of source_var from meta.

    Returns
    -------
    df, meta    : updated dataframe and meta (always unpack both)

    Usage
    -----
    df, meta = apply_recode(df, meta, "region",   "region_grp",  recode_region,  "Region")
    df, meta = apply_recode(df, meta, "age_band", "age_grp",     recode_age_group)

    Then use new_var ("region_grp", "age_grp") directly in any table call:
        create_table(df, meta, row="region_grp", single_cols=["s1","s2"], ...)
        multi_row_table(df, meta, row_var="OBS4", col_vars=["region_grp","s1"])
    """

    # 1. Resolve the source column to readable text before recoding.
    #    SPSS variables store numeric codes (1.0, 2.0 ...) not the visible text.
    #    If value labels exist, map codes -> text first so the recode function
    #    receives "Distributor" not "1.0", "Retailer" not "2.0", etc.
    label_name  = meta.variable_to_label.get(source_var)
    labels_dict = meta.value_labels.get(label_name) if label_name else None

    if labels_dict is not None:
        # Map numeric code -> readable text, then apply recode on the text
        source_series = df[source_var].map(labels_dict)
    else:
        # No value labels — column is already text (e.g. an open-ended string)
        source_series = df[source_var]

    # 2. Apply recode — each cell goes through the function row by row
    df[new_var] = source_series.apply(recode_func)

    # 3. Register a question label in meta so get_var_label() returns it correctly
    if label is None:
        label = meta.column_names_to_labels.get(source_var, source_var)
    meta.column_names_to_labels[new_var] = label

    # 4. No value labels needed — recode functions return readable text directly.
    #    apply_labels() handles columns with no value labels gracefully (uses raw str).

    return df, meta

def get_var_label(meta, var):
    return meta.column_names_to_labels.get(var, var)


# =========================
# RESOLVE MULTI MAP
# =========================
def resolve_multi_map(meta, var, multi_maps=None):
    """
    Reads the choice/option labels for a multi-response variable.

    Priority:
      1. Explicitly passed multi_maps argument
      2. Global multi_maps dict at the top of the file
      3. SPSS value labels via meta.variable_to_label -> meta.value_labels
      4. Returns None if nothing found

    SPSS stores codes as floats (0.0, 1.0).
    get_dummies(sep=" ") produces plain int strings ("0", "1").
    Cast: float 0.0 -> int 0 -> str "0" so they match.
    """
    # 1. Explicitly passed multi_maps takes priority
    if multi_maps and var in multi_maps:
        return multi_maps[var]

    # 2. Fall back to global multi_maps
    global_maps = globals().get("multi_maps", {})
    if global_maps and var in global_maps:
        return global_maps[var]

    # 3. Auto-read from SPSS value labels
    label_name = meta.variable_to_label.get(var)
    if label_name:
        labels_dict = meta.value_labels.get(label_name)
        if labels_dict:
            return {str(int(float(k))): v for k, v in labels_dict.items()}

    return None

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

def get_var_label(meta, var):
    return meta.column_names_to_labels.get(var, var)

# =========================
# MASTER FUNCTION
# =========================

def create_table(df, meta, row, single_cols, multi_cols, multi_maps, stats_var=None, header_dict=None):
    # 1. Apply labels to the row variable
    df, row_labels = apply_labels(df, meta, row)
    row_label = get_var_label(meta, row)

    tables = []

    # =========================
    # 1. GENERATE PERCENTAGE TABLES
    # =========================
    for var in single_cols:
        tables.append(single_table(df, meta, row, var))

    for var in multi_cols:
        var_map = multi_maps.get(var, {})
        tables.append(multi_table(df, row, var, var_map))

    final_table = pd.concat(tables, axis=1)

    # Reindex to ensure all row labels are present
    if row_labels is not None:
        final_table = final_table.reindex(list(row_labels.values()))

    # Calculate Base row
    base_row_vals = final_table.replace("-", 0).astype(float).sum(axis=0)
    base_row = pd.DataFrame([base_row_vals], index=["Base"])

    # Merge Base and Percentages
    final_table = pd.concat([base_row, final_table])

    # =========================
    # 2. ADD STATS (MEAN/SD/SE)
    # =========================
    if stats_var:
        df[stats_var] = pd.to_numeric(df[stats_var], errors="coerce")
        stats_blocks = []
        groups = final_table.columns.get_level_values(0).unique()

        for group in groups:
            sub_cols = final_table[group].columns
            temp_stats = pd.DataFrame(index=["Mean", "Standard deviation", "Standard error"], columns=sub_cols)

            # FIX 1: group is the raw variable name, so compare directly to single_cols
            if group in single_cols:
                real_var = group
                df, _ = apply_labels(df, meta, real_var)

                for col in sub_cols:
                    sub_data = df.loc[df[f"{real_var}_label"].astype(str) == str(col), stats_var].dropna()

                    if not sub_data.empty:
                        temp_stats.loc["Mean", col] = sub_data.mean()
                        temp_stats.loc["Standard deviation", col] = sub_data.std()
                        if len(sub_data) > 0:
                            temp_stats.loc["Standard error", col] = sub_data.std() / (len(sub_data) ** 0.5)

            # FIX 2: compare m_var directly to group (both are raw variable names)
            else:
                for m_var, m_map in multi_maps.items():
                    if m_var == group:
                        split_df = df[m_var].astype(str).str.get_dummies(sep=" ")
                        split_df = split_df.rename(columns=m_map)
                        for col in sub_cols:
                            if col in split_df.columns:
                                sub_data = df.loc[split_df[col] == 1, stats_var].dropna()
                                if not sub_data.empty:
                                    temp_stats.loc["Mean", col] = sub_data.mean()
                                    temp_stats.loc["Standard deviation", col] = sub_data.std()
                                    if len(sub_data) > 0:
                                        temp_stats.loc["Standard error", col] = sub_data.std() / (len(sub_data) ** 0.5)

            temp_stats = temp_stats.astype(float).round(3)
            temp_stats.columns = pd.MultiIndex.from_product([[group], temp_stats.columns])
            stats_blocks.append(temp_stats)

        stats_full = pd.concat(stats_blocks, axis=1)
        final_table = pd.concat([final_table, stats_full])

    # =========================
    # 3. TOTAL COLUMN (LEFT)
    # =========================
    total_dist = (df[f"{row}_label"].value_counts(normalize=True) * 100).round(0)
    total_col = total_dist.reindex(final_table.index)
    total_col.loc["Base"] = 100.0

    if stats_var:
        all_stats_data = df[stats_var].dropna()
        if not all_stats_data.empty:
            total_col.loc["Mean"] = round(all_stats_data.mean(), 3)
            total_col.loc["Standard deviation"] = round(all_stats_data.std(), 3)
            total_col.loc["Standard error"] = round(all_stats_data.std() / (len(all_stats_data) ** 0.5), 3)

    total_df = pd.DataFrame(total_col)
    total_df.columns = pd.MultiIndex.from_tuples([("", "Total")])

    final_table = pd.concat([total_df, final_table], axis=1)

    # Final Formatting
    final_table.rename_axis(index=row_label, inplace=True)
    final_table = final_table.fillna("-").replace(0, "-")

    return final_table
# =========================
def multi_row_table(df, meta, row_var, col_vars, multi_maps,header_dict=None):
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

            # TRUE cross-tab
            temp = split_row.T.dot(col_dummy)

            #  COLUMN BASE (correct)
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
            if val not in ["0", "0.0", "no", "false", "nan", "none", ""]:
                selected.append(col)
        return " ".join(selected)
 
    df[merged_name] = df[relevant_cols].apply(get_selected, axis=1)
 
    # =========================
    # 4. ADD TO MULTI MAPS
    # =========================
    multi_maps = multi_maps.copy()
    multi_maps[merged_name] = extracted_map
 
    # =========================
    # 5. CALL MULTI ROW TABLE
    # =========================
    final_table = multi_row_table(
        df,
        meta,
        row_var=merged_name,
        col_vars=col_vars,
        multi_maps=multi_maps
    )
 
    # =========================
    # 6. BASE ROW (top)
    # Sum of each column across all option rows
    # =========================
    base_vals = final_table.replace("-", 0).astype(float).sum(axis=0)
    base_row  = pd.DataFrame([base_vals], index=["Base"])
    base_row.index.name = main_question if main_question else prefix
 
    final_table = pd.concat([base_row, final_table])
 
    # =========================
    # 7. TOTAL COLUMN (left)
    # For each option row: sum its % values across all column groups
    # =========================
    total_vals = final_table.replace("-", 0).astype(float).sum(axis=1)
    total_df   = pd.DataFrame(total_vals)
    total_df.columns = pd.MultiIndex.from_tuples([("", "Total")])
 
    final_table = pd.concat([total_df, final_table], axis=1)
    final_table.index.name = main_question if main_question else prefix
 
    return final_table
 

# =========================
# EXAMPLE USAGE
