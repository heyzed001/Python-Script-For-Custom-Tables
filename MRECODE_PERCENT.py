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
# MULTI MAPS (MULTIPLE)
# =========================
# Add variables here ONLY if they have no value labels in the SPSS file.
# Variables that have SPSS value labels are read automatically.
multi_maps = {
    "OBS4": {"0": "No", "1": "Yes"}
}


# Custom headers dictionary
custom_headers = {
    # --- Professional Background & Experience ---
}


# =========================
# APPLY RECODE
# =========================
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


# to return label instead of variable
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
# SINGLE TABLE
# =========================
def single_table(df, meta, row, var):

    df, labels_dict = apply_labels(df, meta, var)

    col_dummy = pd.get_dummies(df[f"{var}_label"])

    table = col_dummy.groupby(df[f"{row}_label"]).sum()

    if labels_dict is not None:
        table = table.reindex(columns=list(labels_dict.values()), fill_value=0)

    # col% — divide each cell by its column total, round to nearest integer
    col_base = table.sum(axis=0).replace(0, np.nan)
    table = (table.div(col_base, axis=1) * 100).fillna(0).round(0).astype(int)

    var_label = get_var_label(meta, var)
    table.columns = pd.MultiIndex.from_product([[var_label], table.columns])
    row_label = get_var_label(meta, row)
    table.index.name = row_label

    return table


# =========================
# MULTI TABLE (COLUMN MODE)
# =========================
def multi_table(df, row, var, multi_map):

    split_df = df[var].astype(str).str.get_dummies(sep=" ")
    split_df = split_df.rename(columns=multi_map)
    split_df = split_df.reindex(columns=list(multi_map.values()), fill_value=0)

    table = split_df.groupby(df[f"{row}_label"]).sum()

    # col% — divide each cell by its column total, round to nearest integer
    col_base = table.sum(axis=0).replace(0, np.nan)
    table = (table.div(col_base, axis=1) * 100).fillna(0).round(0).astype(int)

    var_label = get_var_label(meta, var)
    table.columns = pd.MultiIndex.from_product([[var_label], table.columns])
    row_label = get_var_label(meta, row)
    table.index.name = row_label

    return table


# =========================
# MASTER FUNCTION (NORMAL)
# =========================
def create_table(df, meta, row, single_cols, multi_cols, multi_maps, stats_var=None):

    df, row_labels = apply_labels(df, meta, row)

    row_label = get_var_label(meta, row)

    tables = []

    for var in single_cols:
        tables.append(single_table(df, meta, row, var))

    for var in multi_cols:
        var_map = multi_maps.get(var, {})
        tables.append(multi_table(df, row, var, var_map))

    final_table = pd.concat(tables, axis=1)

    if row_labels is not None:
        final_table = final_table.reindex(list(row_labels.values()))

    # BASE ROW — 100 for every column (= 100% of column respondents)
    base_row = pd.DataFrame(
        [[100] * len(final_table.columns)],
        columns=final_table.columns,
        index=["Base"]
    )
    base_row.index.name = row_label
    final_table = pd.concat([base_row, final_table])

    # MEAN / SD / SE
    if stats_var:

        df[stats_var] = pd.to_numeric(df[stats_var], errors="coerce")
        stats_blocks = []
        var_label_map = {get_var_label(meta, v): v for v in single_cols}

        for group in final_table.columns.get_level_values(0).unique():

            if group == "":
                continue

            sub_cols = final_table[group].columns
            temp_stats = pd.DataFrame(
                index=["Mean", "Standard deviation", "Standard error"],
                columns=sub_cols
            )

            if group in var_label_map:
                real_var = var_label_map[group]
                df, labels_dict = apply_labels(df, meta, real_var)
                grouped = df.groupby(f"{real_var}_label")[stats_var]
                mean  = grouped.mean()
                std   = grouped.std()
                count = grouped.count()
                se    = std / (count ** 0.5)
                temp_stats.loc["Mean"]               = mean.reindex(sub_cols)
                temp_stats.loc["Standard deviation"] = std.reindex(sub_cols)
                temp_stats.loc["Standard error"]     = se.reindex(sub_cols)
            else:
                for var, mp in multi_maps.items():
                    if get_var_label(meta, var) == group:
                        split_df = df[var].astype(str).str.get_dummies(sep=" ")
                        split_df = split_df.rename(columns=mp)
                        for col in sub_cols:
                            if col in split_df.columns:
                                sub = df.loc[split_df[col] == 1, stats_var]
                            else:
                                sub = pd.Series()
                            sub = sub.dropna()
                            if len(sub) > 0:
                                mean = sub.mean(); std = sub.std()
                                se   = std / (len(sub) ** 0.5)
                            else:
                                mean, std, se = 0, 0, 0
                            temp_stats.loc["Mean", col]               = round(mean, 2)
                            temp_stats.loc["Standard deviation", col] = round(std,  2)
                            temp_stats.loc["Standard error", col]     = round(se,   2)

            temp_stats.columns = pd.MultiIndex.from_product([[group], temp_stats.columns])
            stats_blocks.append(temp_stats)

        stats_full = pd.concat(stats_blocks, axis=1)
        stats_full = stats_full.reindex(columns=final_table.columns, fill_value="")
        final_table = pd.concat([final_table, stats_full])

    # TOTAL COLUMN (left)
    # Overall col% of the row variable = distribution across all respondents
    total_dist = (df[f"{row}_label"].value_counts(normalize=True) * 100).round(0).astype(int)

    total_values = []

    if stats_var:
        stats_data = df[stats_var].dropna()

    for idx in final_table.index:
        if stats_var and idx == "Mean":
            total_values.append(round(stats_data.mean(), 2))
        elif stats_var and idx == "Standard deviation":
            total_values.append(round(stats_data.std(), 2))
        elif stats_var and idx == "Standard error":
            total_values.append(round(stats_data.std() / (len(stats_data) ** 0.5), 2))
        elif idx == "Base":
            total_values.append(100)
        else:
            total_values.append(int(total_dist.get(idx, 0)))

    total_df = pd.DataFrame(total_values, index=final_table.index)
    total_df.columns = pd.MultiIndex.from_tuples([("", "Total")])
    final_table = pd.concat([total_df, final_table], axis=1)
    final_table.index.name = row_label

    final_table = final_table.fillna("-").replace(0, "-")

    return final_table


# =========================
# MULTI AS ROW
# =========================
def multi_row_table(df, meta, row_var, col_vars, multi_maps):
    """
    Multi-response as ROW with:
    ✔ single columns  — correct col%
    ✔ multi columns   — correct col%
    ✔ Base row        — 100 for every column (= 100% of column respondents)
    ✔ Total column    — overall col% per row option across all respondents
    ✔ Integer rounding — no decimals
    """

    # =========================
    # ROW → MULTI RESPONSE
    # =========================
    row_map = multi_maps.get(row_var, {})

    split_row = df[row_var].astype(str).str.get_dummies(sep=" ")
    split_row = split_row.rename(columns=row_map)
    split_row = split_row.reindex(columns=list(row_map.values()), fill_value=0)

    tables = []

    for col in col_vars:

        # =====================
        # SINGLE COLUMN
        # =====================
        if col not in multi_maps:

            df, labels_dict = apply_labels(df, meta, col)

            col_dummy = pd.get_dummies(df[f"{col}_label"])

            if labels_dict is not None:
                col_dummy = col_dummy.reindex(
                    columns=list(labels_dict.values()), fill_value=0
                )

            # count: how many selected each row option per col category
            temp = split_row.T.dot(col_dummy)

            # col base = total respondents in each col category
            col_base = col_dummy.sum(axis=0).replace(0, np.nan)

            # col % — rounded to nearest integer
            temp = (temp.div(col_base, axis=1) * 100).fillna(0).round(0).astype(int)

        # =====================
        # MULTI COLUMN
        # =====================
        else:

            col_map = multi_maps[col]

            split_col = df[col].astype(str).str.get_dummies(sep=" ")
            split_col = split_col.rename(columns=col_map)
            split_col = split_col.reindex(
                columns=list(col_map.values()), fill_value=0
            )

            # count: co-selections between row options and col options
            temp = split_row.T.dot(split_col)

            # col base = total respondents who selected each col option
            col_base = split_col.sum(axis=0).replace(0, np.nan)

            # col % — rounded to nearest integer
            temp = (temp.div(col_base, axis=1) * 100).fillna(0).round(0).astype(int)

        col_label = get_var_label(meta, col) or col
        temp.columns = pd.MultiIndex.from_product([[col_label], temp.columns])
        tables.append(temp)

    # =========================
    # MERGE ALL TABLES
    # =========================
    final_table = pd.concat(tables, axis=1)

    # =========================
    # BASE ROW (top)
    # Every column base = 100  (represents 100% of that column's respondents)
    # =========================
    base_row = pd.DataFrame(
        [[100] * len(final_table.columns)],
        columns=final_table.columns,
        index=["Base"]
    )
    final_table = pd.concat([base_row, final_table])

    # =========================
    # TOTAL COLUMN (left)
    # Overall col% per row option = selected by that option / all respondents * 100
    # Base in Total = 100
    # =========================
    n_total = len(df)
    option_pct = (split_row.sum(axis=0) / n_total * 100).round(0).astype(int)

    total_series = option_pct.reindex(final_table.index, fill_value=0)
    total_series["Base"] = 100

    total_df = pd.DataFrame({"Total": total_series})
    total_df.columns = pd.MultiIndex.from_tuples([("", "Total")])

    final_table = pd.concat([total_df, final_table], axis=1)
    final_table.index.name = get_var_label(meta, row_var)

    return final_table


# =========================
# AUTO MULTI TABLE
# =========================
def auto_multi_table(df, meta, prefix, col_vars, header_dict):
    relevant_cols = [c for c in df.columns if c.startswith(prefix)]
    extracted_map = {
        str(i+1): c.split(':')[-1].strip()
        for i, c in enumerate(relevant_cols)
    }
    merged_name = f"{prefix}_Merged"
    df[merged_name] = df[relevant_cols].apply(
        lambda x: " ".join([str(i+1) for i, v in enumerate(x) if str(v).lower() in ['yes', '1', '1.0']]),
        axis=1
    )
    return multi_row_table(
        df, meta,
        row_var=merged_name,
        col_vars=col_vars,
        multi_maps={merged_name: extracted_map},
        header_dict=header_dict
    )


# =========================
# SPSS MULTI TABLE
# =========================
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

    # DO NOT remove empty responses — it breaks column base for percentages

    # =========================
    # 4. ADD TO MULTI MAPS
    # =========================
    multi_maps = multi_maps.copy()
    multi_maps[merged_name] = extracted_map

    # =========================
    # 5. CALL MULTI ROW TABLE
    # col%, Base (=100), Total, and integer rounding all handled inside multi_row_table
    # =========================
    final_table = multi_row_table(
        df,
        meta,
        row_var=merged_name,
        col_vars=col_vars,
        multi_maps=multi_maps
    )

    # Set question label as index name
    final_table.index.name = main_question if main_question else prefix

    return final_table


# =========================
# FORMAT EXCEL TABLE
# =========================
"""def format_excel_table(writer, sheet_name, current_row, df):
    workbook  = writer.book
    worksheet = workbook[sheet_name]

    header_fill   = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    header_font   = Font(color="FFFFFF", bold=True, size=11)
    base_row_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    border_side = Side(style='thin', color="000000")
    border      = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)

    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align   = Alignment(horizontal="left",   vertical="center", wrap_text=True)

    header_depth  = df.columns.nlevels
    index_width   = df.index.nlevels
    total_cols    = len(df.columns) + index_width
    data_start_row = current_row + 1 + header_depth

    for r in range(current_row + 1, data_start_row):
        for c in range(1, total_cols + 1):
            cell = worksheet.cell(row=r, column=c)
            cell.fill      = header_fill
            cell.font      = header_font
            cell.alignment = center_align
            cell.border    = border

    for c in range(1, total_cols + 1):
        cell = worksheet.cell(row=data_start_row, column=c)
        cell.fill      = base_row_fill
        cell.font      = Font(bold=True)
        cell.alignment = center_align if c > index_width else left_align
        cell.border    = border

    for r in range(data_start_row + 1, data_start_row + len(df)):
        for c in range(1, total_cols + 1):
            cell = worksheet.cell(row=r, column=c)
            cell.border    = border
            cell.alignment = center_align if c > index_width else left_align


# =========================
# WRITE TABLE
# =========================
def write_table(writer, df, sheet, current_row, title=None, table_num=None):
    workbook = writer.book

    if sheet not in workbook.sheetnames:
        workbook.create_sheet(sheet)

    worksheet = workbook[sheet]

    if title is not None:
        label = f"Table {table_num}: {title}" if table_num is not None else title
        worksheet.cell(row=current_row + 1, column=1).value = label
        current_row += 1

    df.to_excel(writer, sheet_name=sheet, startrow=current_row)
    format_excel_table(writer, sheet, current_row, df)

    return current_row + df.columns.nlevels + len(df) + 3
"""

# =========================
# APPLY RECODES HERE
# =========================
# Call apply_recode() once after loading data, before any table calls.
# Syntax:
#   df, meta = apply_recode(df, meta, "source_col", "new_col", recode_func, "Table Label")
#
# Examples:
#   df, meta = apply_recode(df, meta, "region",   "region_grp",  recode_region,    "Region")
#   df, meta = apply_recode(df, meta, "age_band", "age_grp",     recode_age_group, "Age Group")
#
# Then use "region_grp" or "age_grp" anywhere in your table calls:
#   create_table(df, meta, row="region_grp",  single_cols=["s1","s2"], ...)
#   create_table(df, meta, row="s1", single_cols=["region_grp", "age_grp"], ...)
#   multi_row_table(df, meta, row_var="OBS4", col_vars=["region_grp","s1"])
#   spss_multi_table(df, meta, "Q24", col_vars=["region_grp","s1"], ...)


# =========================
# TABLE CALLS

