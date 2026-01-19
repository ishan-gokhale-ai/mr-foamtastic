import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from io import BytesIO

APP_VERSION = "V6.6.0"

# --- 0. LOOK AND FEEL ---
st.markdown("""
    <style>
    [data-testid="stSidebarContent"] {
        padding-top: 0rem !important;
        margin-top: -35px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.75rem !important;
    }
    [data-testid="stSidebar"] label {
        margin-bottom: -5px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #31333F !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99999;
    }
    [data-testid="stDecoration"], [data-testid="stToolbar"] {
        display: none;
    }
    [data-testid="stTab"] {
        font-weight: 700 !important;
    }
    [data-testid="stTableSummary"] + div div[role="grid"] div[role="row"] div[role="gridcell"]:nth-child(1) {
        background-color: #f0f2f6 !important;
        font-style: italic;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. PASSWORD GATE ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.set_page_config(page_title="Mr. Foamtastic Login", page_icon="logo1.png")
        st.title(f"🔒 Mr. Foamtastic {APP_VERSION}")
        st.text_input("Enter Team Password", type="password", on_change=password_entered, key="password")
        st.info("Contact Ishan Gokhale for access.")
        return False
    elif not st.session_state["password_correct"]:
        st.set_page_config(page_title="Mr. Foamtastic Login", page_icon="logo1.png")
        st.text_input("Enter Team Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop()

# --- 2. CONFIGURATION & STATE ---
st.set_page_config(layout="wide", page_title="Mr. Foamtastic", page_icon="logo1.png")

if 'export_basket' not in st.session_state:
    st.session_state['export_basket'] = []
if 'explore_stage' not in st.session_state:
    st.session_state['explore_stage'] = []

# --- 3. DATA LOADING ---
@st.cache_data
def load_data(uploaded_file=None):
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("foams.csv")
        if 'Conductive' in df.columns:
            df['Conductive'] = df['Conductive'].astype(str).str.upper().map({'YES': True, 'NO': False})
        if 'PSA' in df.columns:
            df['PSA'] = df['PSA'].astype(str).str.upper().map({'YES': True, 'NO': False})
        df['Search_Label'] = df['Manufacturer'] + " | " + df['Model']
        return df
    except Exception:
        return None

df = load_data()

if df is None:
    with st.sidebar:
        st.warning("⚠️ 'foams.csv' not found.")
        uploaded_csv = st.file_uploader("Upload Foams Database", type="csv")
        if uploaded_csv:
            df = load_data(uploaded_csv)
            st.rerun()
        else:
            st.stop()

# --- 4. CORE CALCULATION ENGINE ---
def get_value_with_status(row, gap, mode, area_val, allow_extrapolation=False):
    thickness_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
    x = np.array([float(c) for c in thickness_cols])
    y = row[thickness_cols].values.astype(float)
    valid_mask = y > 0
    x_valid, y_valid = x[valid_mask], y[valid_mask]

    if len(x_valid) < 2: return 0.0, "No Data"
    if gap <= 0: return 0.0, "Interpolated"

    is_extrapolated = False
    if gap < x_valid.min() or gap > x_valid.max():
        if not allow_extrapolation: return 0.0, "Out of Range"
        is_extrapolated = True

    f = interp1d(x_valid[np.argsort(x_valid)], y_valid[np.argsort(x_valid)], kind='linear', fill_value="extrapolate")
    stress = float(f(gap))
    val = stress if mode == "Stress (MPa)" else (stress * area_val)
    return val, ("Extrapolated" if is_extrapolated else "Interpolated")

# --- 5. HELPER FUNCTIONS ---
def update_explore_stage():
    selection = st.session_state.get('search_widget', [])
    selected_models = [s.split(" | ")[1] for s in selection]
    new_stage = [item for item in st.session_state['explore_stage'] if item['model'] in selected_models]
    existing_models = [item['model'] for item in new_stage]
    for s in selection:
        model = s.split(" | ")[1]
        if model not in existing_models:
            row_match = df[df['Model'] == model]
            if not row_match.empty:
                new_stage.append({"custom_name": model, "model": model, "row": row_match.iloc[0]})
    st.session_state['explore_stage'] = new_stage

def clear_stage():
    st.session_state['explore_stage'] = []
    st.session_state['search_widget'] = []

# --- 6. SIDEBAR ---
with st.sidebar:
    st.caption(f"{APP_VERSION} | #Slack Ishan Gokhale")
    st.divider()
    unit_mode = st.radio("Output Type", ["Stress (MPa)", "Force (N)"])
    unit_label = "MPa" if unit_mode == "Stress (MPa)" else "N"
    
    if unit_mode == "Force (N)":
        area = st.number_input("Area (mm²)", value=20.0, step=1.0, format="%.2f")
        v_def_min, v_def_max = 1.60, 2.40
    else:
        area = 1.0 
        v_def_min, v_def_max = 0.080, 0.120

    with st.expander("📂 Filters", expanded=True):
        mfr_options = sorted(df['Manufacturer'].unique().tolist())
        sel_mfrs = st.multiselect("Vendors", mfr_options, default=mfr_options)
        want_cond = st.checkbox("Conductive")
        want_psa = st.checkbox("With PSA")

    with st.expander("🛠️ Constraints", expanded=True):
        t_min_in = st.number_input("Min Thk", value=float(df['thickness'].min()), format="%.3f")
        t_max_in = st.number_input("Max Thk", value=float(df['thickness'].max()), format="%.3f")
        v_min_in = st.number_input("Min Target", value=v_def_min, format="%.3f")
        v_max_in = st.number_input("Max Target", value=v_def_max, format="%.3f")
        c_min_in = st.number_input("Min % Comp", value=20)
        c_max_in = st.number_input("Max % Comp", value=70)

# --- 7. TABS ---
col1, col2, col3 = st.columns([1, 3, 1]) 
with col2: st.image("banner.png", use_container_width=True)

tab_select, tab_explore, tab_export = st.tabs(["SELECT", "EXPLORE", "EXPORT"])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --- TAB : SELECT ---
with tab_select:
    s_col1, s_col2, s_col3 = st.columns([1, 1, 1])
    with s_col1: s_gap = st.number_input("Nominal Gap (mm)", value=0.400, step=0.010, format="%.3f", key="sgap_s")
    with s_col2: s_tol = st.number_input("Gap Tolerance (± mm)", value=0.100, step=0.010, format="%.3f", key="stol_s")
    
    mask = (df['Manufacturer'].isin(sel_mfrs)) & (df['thickness'] >= t_min_in) & (df['thickness'] <= t_max_in)
    if want_cond: mask &= (df['Conductive'] == True)
    if want_psa: mask &= (df['PSA'] == True)
    filtered_df = df[mask]
    
    mode_label = unit_mode.split()[0]
    results = []
    for _, row in filtered_df.iterrows():
        vn, sn = get_value_with_status(row, s_gap, unit_mode, area, False)
        comp = ((row['thickness'] - s_gap) / row['thickness']) * 100
        if sn == "Interpolated" and v_min_in <= vn <= v_max_in and c_min_in <= comp <= c_max_in:
            v_min, s_min = get_value_with_status(row, s_gap - s_tol, unit_mode, area, True)
            v_max, s_max = get_value_with_status(row, s_gap + s_tol, unit_mode, area, True)
            results.append({
                "Foam Name": "Enter custom name...", 
                "Vendor": row['Manufacturer'], "Model": row['Model'], "Thk": row['thickness'],
                f"Nom {mode_label}": vn, 
                "Min Gap Val": f"⚠️ {v_min:.3f}" if s_min == "Extrapolated" else f"{v_min:.3f}", 
                "Max Gap Val": f"⚠️ {v_max:.3f}" if s_max == "Extrapolated" else f"{v_max:.3f}", 
                "row_ref": row, "Add to Export": False
            })

    if results:
        all_thks = [r['Thk'] for r in results]
        x_start, x_end = 0.0, max(all_thks) * 1.2
        px_range = np.linspace(x_start, x_end, 200)

        fig_sel = go.Figure()
        fig_sel.add_vrect(x0=s_gap - s_tol, x1=s_gap + s_tol, fillcolor="rgba(100,100,100,0.1)", line_width=0)
        
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            hover_labels = []
            for tx in px_range:
                model_base = r['Model']
                if abs(tx - s_gap) < 0.005: status = " (NOMINAL)"
                elif abs(tx - (s_gap - s_tol)) < 0.005: status = " (MIN TOL)"
                elif abs(tx - (s_gap + s_tol)) < 0.005: status = " (MAX TOL)"
                else: status = ""
                hover_labels.append(f"{model_base}{status}")

            py = [get_value_with_status(r['row_ref'], tx, unit_mode, area, False)[0] for tx in px_range]
            fig_sel.add_trace(go.Scatter(
                x=px_range, y=[y if y >= 0 else None for y in py], 
                name=r['Model'], line=dict(color=color),
                customdata=hover_labels,
                hovertemplate="<b>%{customdata}</b><br>Gap: %{x:.3f} mm<br>"+f"{mode_label}: %{{y:.3f}} {unit_label}<extra></extra>"
            ))
        
        # ADD NOMINAL VERTICAL LINE
        fig_sel.add_vline(x=s_gap, line_dash="dash", line_color="#888", annotation_text="NOMINAL", annotation_position="top")

        fig_sel.update_layout(
            template="plotly_white", height=400, 
            xaxis=dict(range=[x_start, x_end], title="Compressed thickness (mm)"), 
            yaxis=dict(rangemode="tozero", title=unit_mode), 
            hovermode="x unified"
        )
        st.plotly_chart(fig_sel, use_container_width=True)

        display_df = pd.DataFrame(results).drop(columns=['row_ref'])
        display_df['Thk'] = display_df['Thk'].map('{:.3f}'.format)
        display_df[f"Nom {mode_label}"] = display_df[f"Nom {mode_label}"].map('{:.3f}'.format)

        edited_df = st.data_editor(display_df, column_config={"Foam Name": st.column_config.TextColumn("✏️ Foam Name", width="medium"), "Add to Export": st.column_config.CheckboxColumn("Add")}, disabled=["Vendor", "Model", "Thk", f"Nom {mode_label}", "Min Gap Val", "Max Gap Val"], use_container_width=True, hide_index=True, key="selection_editor")

        for i, row in edited_df.iterrows():
            r_orig = results[i]
            add_key = f"added_{r_orig['Model']}_{i}"
            if row["Add to Export"] and not st.session_state.get(add_key, False):
                raw_name = str(row.get("Foam Name", ""))
                final_name = raw_name if raw_name not in ["Enter custom name...", "", "None"] else ""
                st.session_state['export_basket'].append({
                    "Foam": final_name, "Vendor": r_orig['Vendor'], "Model": r_orig['Model'], "Thk (mm)": round(r_orig['Thk'], 3),
                    "Nom Gap (mm)": round(s_gap, 3), f"Nom {unit_mode}": round(r_orig[f"Nom {mode_label}"], 3),
                    "Min Gap (mm)": round(s_gap - s_tol, 3), f"Min Gap {unit_mode}": r_orig['Min Gap Val'], 
                    "Max Gap (mm)": round(s_gap + s_tol, 3), f"Max Gap {unit_mode}": r_orig['Max Gap Val']
                })
                st.session_state[add_key] = True
            elif not row["Add to Export"]: st.session_state[add_key] = False

# --- TAB : EXPLORE ---
with tab_explore:
    c_gap, c_tol, c_clear = st.columns([1, 1, 1])
    with c_gap: e_gap = st.number_input("Nominal Gap (mm)", value=0.400, step=0.010, format="%.3f", key="egap_global")
    with c_tol: e_tol = st.number_input("Tolerance (± mm)", value=0.100, step=0.005, format="%.3f", key="etol_global")
    with c_clear: st.button("🗑️ Clear Stage", on_click=clear_stage, use_container_width=True)
    
    st.multiselect("Foam Search", sorted(df['Search_Label'].unique().tolist()), key="search_widget", on_change=update_explore_stage)

    if st.session_state['explore_stage']:
        ex_thks = [item['row']['thickness'] for item in st.session_state['explore_stage']]
        ex_start, ex_end = 0.0, max(ex_thks) * 1.2
        px_range_ex = np.linspace(ex_start, ex_end, 200)

        fig_exp = go.Figure()
        fig_exp.add_vrect(x0=e_gap - e_tol, x1=e_gap + e_tol, fillcolor="rgba(100,100,100,0.1)", line_width=0)
        
        explore_results = []
        for i, item in enumerate(st.session_state['explore_stage']):
            color = colors[i % len(colors)]
            item_row = item['row']
            vn, _ = get_value_with_status(item_row, e_gap, unit_mode, area, False)
            v_min, s_min = get_value_with_status(item_row, e_gap - e_tol, unit_mode, area, True)
            v_max, s_max = get_value_with_status(item_row, e_gap + e_tol, unit_mode, area, True)
            
            explore_results.append({
                "Foam Name": "Enter custom name...", 
                "Vendor": item_row['Manufacturer'], "Model": item['model'], "Thk": item_row['thickness'],
                f"Nom {mode_label}": vn, 
                "Min Gap Val": f"⚠️ {v_min:.3f}" if s_min == "Extrapolated" else f"{v_min:.3f}", 
                "Max Gap Val": f"⚠️ {v_max:.3f}" if s_max == "Extrapolated" else f"{v_max:.3f}", 
                "row_ref": item_row, "Add to Export": False
            })
            
            hover_labels_ex = []
            for tx in px_range_ex:
                model_base = item['model']
                if abs(tx - e_gap) < 0.005: status = " (NOMINAL)"
                elif abs(tx - (e_gap - e_tol)) < 0.005: status = " (MIN TOL)"
                elif abs(tx - (e_gap + e_tol)) < 0.005: status = " (MAX TOL)"
                else: status = ""
                hover_labels_ex.append(f"{model_base}{status}")

            py = [get_value_with_status(item_row, tx, unit_mode, area, False)[0] for tx in px_range_ex]
            fig_exp.add_trace(go.Scatter(
                x=px_range_ex, y=[y if y >= 0 else None for y in py], 
                name=item['model'], line=dict(color=color),
                customdata=hover_labels_ex,
                hovertemplate="<b>%{customdata}</b><br>Gap: %{x:.3f} mm<br>"+f"{mode_label}: %{{y:.3f}} {unit_label}<extra></extra>"
            ))

        # ADD NOMINAL VERTICAL LINE
        fig_exp.add_vline(x=e_gap, line_dash="dash", line_color="#888", annotation_text="NOMINAL", annotation_position="top")

        fig_exp.update_layout(
            template="plotly_white", 
            height=450, 
            xaxis=dict(range=[ex_start, ex_end], title="Compressed thickness (mm)"), 
            yaxis=dict(rangemode="tozero", title=unit_mode), 
            hovermode="x unified"
        )
        st.plotly_chart(fig_exp, use_container_width=True)
        
        display_exp_df = pd.DataFrame(explore_results).drop(columns=['row_ref'])
        display_exp_df['Thk'] = display_exp_df['Thk'].map('{:.3f}'.format)
        display_exp_df[f"Nom {mode_label}"] = display_exp_df[f"Nom {mode_label}"].map('{:.3f}'.format)

        edited_exp_df = st.data_editor(display_exp_df, column_config={"Foam Name": st.column_config.TextColumn("✏️ Foam Name", width="medium"), "Add to Export": st.column_config.CheckboxColumn("Add")}, disabled=["Vendor", "Model", "Thk", f"Nom {mode_label}", "Min Gap Val", "Max Gap Val"], use_container_width=True, hide_index=True, key="explore_editor")

        for i, row in edited_exp_df.iterrows():
            r_orig = explore_results[i]
            add_key = f"exp_added_{r_orig['Model']}_{i}"
            if row["Add to Export"] and not st.session_state.get(add_key, False):
                raw_name = str(row.get("Foam Name", ""))
                final_name = raw_name if raw_name not in ["Enter custom name...", "", "None"] else ""
                st.session_state['export_basket'].append({
                    "Foam": final_name, "Vendor": r_orig['Vendor'], "Model": r_orig['Model'], "Thk (mm)": round(r_orig['Thk'], 3),
                    "Nom Gap (mm)": round(e_gap, 3), f"Nom {unit_mode}": round(r_orig[f"Nom {mode_label}"], 3),
                    "Min Gap (mm)": round(e_gap - e_tol, 3), f"Min Gap {unit_mode}": r_orig['Min Gap Val'], 
                    "Max Gap (mm)": round(e_gap + e_tol, 3), f"Max Gap {unit_mode}": r_orig['Max Gap Val']
                })
                st.session_state[add_key] = True
            elif not row["Add to Export"]: st.session_state[add_key] = False

# --- TAB : EXPORT ---
with tab_export:
    if st.session_state['export_basket']:
        edited_basket = st.data_editor(pd.DataFrame(st.session_state['export_basket']), use_container_width=True, hide_index=True, num_rows="dynamic", key="export_editor_final")
        if len(edited_basket) != len(st.session_state['export_basket']):
            st.session_state['export_basket'] = edited_basket.to_dict('records')
            st.rerun()
        st.divider()
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer: pd.DataFrame(st.session_state['export_basket']).to_excel(writer, index=False)
        st.download_button("📥 Download Excel Report", output.getvalue(), "Foam_Report.xlsx", type="primary")
        if st.button("🗑️ Clear Entire Basket"):
            st.session_state['export_basket'] = []
            st.rerun()
    else: st.info("Basket is empty.")