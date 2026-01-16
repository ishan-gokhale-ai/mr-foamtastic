import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from io import BytesIO

APP_VERSION = "V6.5.0"

# --- 0. LOOK AND FEEL ---

st.markdown("""
    <style>
    /* 1. Reset the sidebar padding and pull content up */
    [data-testid="stSidebarContent"] {
        padding-top: 0rem !important;
        margin-top: -35px !important; /* Pulls content way up */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* 2. Vertical gap between widgets */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.75rem !important;
    }

    /* 3. Style labels */
    [data-testid="stSidebar"] label {
        margin-bottom: -5px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #31333F !important;
    }

    /* 4. Divider lines */
    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* 5. Radio Button alignment */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] + div {
        margin-top: 5px !important;
    }

    /* 6. Header spacing */
    [data-testid="stSidebar"] h2 {
        padding-top: 0.5rem !important;
        margin-bottom: 0rem !important;
    }
    
    /* 7. Main page padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        margin-top: 0rem !important;
    }
    
    /* 8. CLEAN HEADER (Fixes missing arrow) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99999 !important; /* Forces the arrow to sit ABOVE the content */
    }
    
    /* Ensure the arrow button is specifically visible and colored */
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        color: #31333F !important;
    }
    
    /* Hide the colored top decoration line */
    [data-testid="stDecoration"] {
        display: none;
    }

    /* Hide the "Deploy" and "..." menu on the right */
    [data-testid="stToolbar"] {
        display: none;
    }
    
    /* 9. Tighten Title Spacing */
    h1 {
        padding-top: 0rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* 10. Tab styling */
    [data-testid="stTab"] {
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* 11. Force Headers Left */
    [data-testid="stDataFrame"] th,
    [data-testid="stDataEditor"] th,
    div[data-testid="stColumnHeader"] {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. PASSWORD GATE ---
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password in state
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.set_page_config(page_title="Mr. Foamtastic Login", page_icon="🔒")
        st.title(f"🔒 Mr. Foamtastic {APP_VERSION}")
        st.text_input("Enter Team Password", type="password", on_change=password_entered, key="password")
        st.info("Contact Ishan Gokhale for access.")
        return False
    elif not st.session_state["password_correct"]:
        st.set_page_config(page_title="Mr. Foamtastic Login", page_icon="🔒")
        st.text_input("Enter Team Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- 2. CONFIGURATION & STATE ---
st.set_page_config(layout="wide", page_title="Mr. Foamtastic", page_icon="logo1.png")

if 'export_basket' not in st.session_state:
    st.session_state['export_basket'] = []
if 'explore_stage' not in st.session_state:
    st.session_state['explore_stage'] = []

# --- 3. DATA LOADING (With Fallback) ---
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            # Try local file
            df = pd.read_csv("foams.csv")
            
        # Clean data
        if 'Conductive' in df.columns:
            df['Conductive'] = df['Conductive'].astype(str).str.upper().map({'YES': True, 'NO': False})
        if 'PSA' in df.columns:
            df['PSA'] = df['PSA'].astype(str).str.upper().map({'YES': True, 'NO': False})
        return df
    except FileNotFoundError:
        return None 
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Try loading local first
df = load_data()

# If local failed, ask for upload in Sidebar
if df is None:
    with st.sidebar:
        st.warning("⚠️ 'foams.csv' not found.")
        uploaded_csv = st.file_uploader("Upload Foams Database", type="csv")
        if uploaded_csv:
            df = load_data(uploaded_csv)
        else:
            st.stop() # Stop execution until file is provided

# --- 4. CORE CALCULATION ENGINE ---
def get_value_with_status(row, gap, mode, area_val, allow_extrapolation=False):
    thickness_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
    x = np.array([float(c) for c in thickness_cols])
    y = row[thickness_cols].values.astype(float)
    valid_mask = y > 0
    x_valid, y_valid = x[valid_mask], y[valid_mask]

    if len(x_valid) < 2: return 0.0, "No Data"
    
    is_extrapolated = False
    if gap < x_valid.min() or gap > x_valid.max():
        if not allow_extrapolation: return 0.0, "Out of Range"
        is_extrapolated = True

    f = interp1d(x_valid[np.argsort(x_valid)], y_valid[np.argsort(x_valid)], kind='linear', fill_value="extrapolate")
    stress = float(f(gap))
    val = stress if mode == "Stress (MPa)" else (stress * area_val)
    return val, ("Extrapolated" if is_extrapolated else "Interpolated")

# --- 5. SIDEBAR ---
with st.sidebar:
    # 1. Header & Branding
    # st.image("logo.png", use_container_width=True)
    st.caption(f"{APP_VERSION} | #Slack Ishan Gokhale")
    st.divider()
    st.write("")

    # 2. Global Calculation Settings
    with st.expander("Stress vs Force Mode", expanded=False):
        unit_mode = st.radio(
            "Select Output Type", 
            ["Stress (MPa)", "Force (N)"],
            help="Stress is area-independent; Force requires contact area."
        )
        
        if unit_mode == "Force (N)":
            area = st.number_input("Contact area (mm²)", value=20.0, step=1.0, format="%.2f")
            v_def_min, v_def_max = 1.60, 2.40
        else:
            area = 1.0 
            v_def_min, v_def_max = 0.080, 0.120

    # 3. Database Filters (Material Properties)
    with st.expander("📂 Material Database Filters", expanded=True):
        mfr_options = sorted(df['Manufacturer'].unique().tolist()) if not df.empty else []
        sel_mfrs = st.multiselect("Vendors", mfr_options, default=mfr_options)
        
        want_cond = st.checkbox("Conductive", help="Filter for EMI/Grounding foams")
        want_psa = st.checkbox("With PSA", help="Filter for foams with inbuilt adhesive")

    # 4. Refinement Filters (Physical Limits)
    with st.expander("🛠️ Performance Constraints", expanded=True):
        st.markdown("### Thickness (mm)")
        t_min_in = st.number_input("Min Thk", value=float(df['thickness'].min() if not df.empty else 0.0), format="%.3f", label_visibility="collapsed")
        t_max_in = st.number_input("Max Thk", value=float(df['thickness'].max() if not df.empty else 5.0), format="%.3f", label_visibility="collapsed")
        
        st.markdown(f"### Target {unit_mode.split()[0]}")
        v_min_in = st.number_input(f"Min {unit_mode.split()[0]}", value=v_def_min, format="%.3f" if unit_mode == "Stress (MPa)" else "%.2f")
        v_max_in = st.number_input(f"Max {unit_mode.split()[0]}", value=v_def_max, format="%.3f" if unit_mode == "Stress (MPa)" else "%.2f")
        
        st.markdown("### Compression %")
        c_col1, c_col2 = st.columns(2)
        c_min_in = c_col1.number_input("Min %", value=20, step=1)
        c_max_in = c_col2.number_input("Max %", value=70, step=1)

# --- 6. TABS (SELECT, EXPLORE, EXPORT) ---

# Banner
col1, col2, col3 = st.columns([1, 3, 1]) 
with col2:
    st.image("banner.png", use_container_width=True)

tab_select, tab_explore, tab_export = st.tabs(["SELECT", "EXPLORE", "EXPORT"])

# --- TAB :SELECT ---
with tab_select:
    st.caption("Enter key design parameters for foam recommendations")
    
    # --- 1. Top Input Row ---
    s_col1, s_col2, s_col3 = st.columns([1, 1, 1])
    with s_col1:
        s_gap = st.number_input("Nominal Gap (mm)", value=0.400, step=0.010, format="%.3f", key="sgap_s")
    with s_col2:
        s_tol = st.number_input("Gap Tolerance (± mm)", value=0.100, step=0.010, format="%.3f", key="stol_s")
    with s_col3:
        st.metric("Calculation Mode", unit_mode.split()[0], delta="Active")

    # --- 2. Filter & Calculation Logic ---
    mask = (df['Manufacturer'].isin(sel_mfrs)) & (df['thickness'] >= t_min_in) & (df['thickness'] <= t_max_in)
    if want_cond: mask &= (df['Conductive'] == True)
    if want_psa: mask &= (df['PSA'] == True)
    filtered_df = df[mask]
    
    # Helper for dynamic headers
    mode_label = unit_mode.split()[0]  # "Stress" or "Force"
    unit_label = unit_mode.split('(')[1].replace(')', '')  # "MPa" or "N"
    
    results = []
    for _, row in filtered_df.iterrows():
        vn, sn = get_value_with_status(row, s_gap, unit_mode, area, False)
        comp = ((row['thickness'] - s_gap) / row['thickness']) * 100
        
        # Only suggest foams where the NOMINAL value is within range and interpolated
        if sn == "Interpolated" and v_min_in <= vn <= v_max_in and c_min_in <= comp <= c_max_in:
            v_min, s_min = get_value_with_status(row, s_gap - s_tol, unit_mode, area, True)
            v_max, s_max = get_value_with_status(row, s_gap + s_tol, unit_mode, area, True)
            
            # Format display strings with ⚠️ warning for extrapolation
            d_min = f"⚠️ {v_min:.3f}" if s_min == "Extrapolated" else f"{v_min:.3f}"
            d_max = f"⚠️ {v_max:.3f}" if s_max == "Extrapolated" else f"{v_max:.3f}"

            results.append({
                "Foam Name": row['Model'], 
                "Vendor": row['Manufacturer'], 
                "Model": row['Model'], 
                "Thk": row['thickness'],
                f"Nom {mode_label}": vn,  # Dynamic key for table mapping
                "Min Gap Val": d_min, 
                "Max Gap Val": d_max, 
                "row_ref": row,
                "Add to Export": False 
            })

    if results:
        # --- 3. Full-Width Performance Chart (With Markers) ---
        fig_sel = go.Figure()
        
        # Tolerance Zone
        fig_sel.add_vrect(
            x0=s_gap - s_tol, x1=s_gap + s_tol, 
            fillcolor="rgba(100,100,100,0.1)", line_width=0, 
            annotation_text="Gap tolerance", annotation_position="top left"
        )
        
        px_range = np.linspace(0.1, 4.0, 200)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            row_data = r['row_ref']
            
            # Curve
            py = [get_value_with_status(row_data, tx, unit_mode, area, False)[0] for tx in px_range]
            fig_sel.add_trace(go.Scatter(
                x=px_range, y=[y if y > 0 else None for y in py], 
                name=r['Model'], mode='lines', line=dict(color=color)
            ))

            # Markers (Min, Nom, Max)
            p_nom, _ = get_value_with_status(row_data, s_gap, unit_mode, area, True)
            p_min, _ = get_value_with_status(row_data, s_gap - s_tol, unit_mode, area, True)
            p_max, _ = get_value_with_status(row_data, s_gap + s_tol, unit_mode, area, True)

            fig_sel.add_trace(go.Scatter(
                x=[s_gap],
                y=[p_nom],
                mode='markers',
                marker=dict(size=8, symbol=['circle'], color=color),
                showlegend=False, hoverinfo='skip'
            ))
        
        fig_sel.update_layout(
            template="plotly_white", height=400, 
            margin=dict(l=10, r=10, t=30, b=10), 
            hovermode="x unified",
            xaxis_title="Gap (mm)",
            yaxis_title=unit_mode,
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_sel, use_container_width=True)

        # --- 4. Interactive Data Table (Left Aligned via Strings) ---
        st.caption("Values marked ⚠️ are extrapolated")
        
        # Display DF: Convert numbers to strings for Left Alignment
        display_df = pd.DataFrame(results).drop(columns=['row_ref'])
        display_df['Thk'] = display_df['Thk'].map('{:.3f}'.format)
        display_df[f"Nom {mode_label}"] = display_df[f"Nom {mode_label}"].map('{:.3f}'.format)
        
        edited_df = st.data_editor(
            display_df, 
            column_config={
                "Foam Name": st.column_config.TextColumn("Foam name (Editable)", width="medium",default="",help="enter foam descriptor"),
                "Thk": st.column_config.TextColumn("Thickness (mm)"),
                f"Nom {mode_label}": st.column_config.TextColumn(f"{mode_label} at Nom gap ({unit_label})"),
                "Min Gap Val": st.column_config.TextColumn(f"{mode_label} at Min gap ({unit_label})"),
                "Max Gap Val": st.column_config.TextColumn(f"{mode_label} at Max gap ({unit_label})"),
                "Add to Export": st.column_config.CheckboxColumn("Add", default=False)
            },
            disabled=["Vendor", "Model", "Thk", f"Nom {mode_label}", "Min Gap Val", "Max Gap Val"],
            use_container_width=True,
            hide_index=True,
            key="selection_editor"
        )

        # --- 5. State-Safe Add Logic ---
        for i, row in edited_df.iterrows():
            # Use results[i] to get the original FLOAT data, not the string display data
            r_orig = results[i]
            add_key = f"added_{r_orig['Model']}_{i}"
            
            if row["Add to Export"] and not st.session_state.get(add_key, False):
                if not any(item['Model'] == r_orig['Model'] and item['Foam'] == row['Foam Name'] for item in st.session_state['export_basket']):
                    st.session_state['export_basket'].append({
                        "Foam": row["Foam Name"],
                        "Vendor": r_orig['Vendor'],
                        "Model": r_orig['Model'],
                        "Thk (mm)": round(r_orig['Thk'], 3),
                        "Nom Gap (mm)": round(s_gap, 3),
                        f"Nom {unit_mode}": round(r_orig[f"Nom {mode_label}"], 3),
                        "Min Gap (mm)": round(s_gap - s_tol, 3),
                        f"Min Gap {unit_mode}": r_orig['Min Gap Val'], 
                        "Max Gap (mm)": round(s_gap + s_tol, 3),
                        f"Max Gap {unit_mode}": r_orig['Max Gap Val']
                    })
                    st.session_state[add_key] = True
                    st.toast(f"✅ Added {row['Foam Name']}!")
            
            elif not row["Add to Export"]:
                st.session_state[add_key] = False

    else:
        st.info("No foams match your current search criteria. Try adjusting the sidebar filters.")

# --- HELPER FUNCTIONS ---
def update_explore_stage():
    """Syncs the search widget with the explore_stage data."""
    selection = st.session_state['search_widget']
    selected_models = [s.split(" | ")[1] for s in selection]
    
    # Filter existing
    new_stage = [item for item in st.session_state['explore_stage'] if item['model'] in selected_models]
    existing_models = [item['model'] for item in new_stage]
    
    # Add new
    for s in selection:
        model = s.split(" | ")[1]
        if model not in existing_models:
            row = df[df['Model'] == model].iloc[0]
            current_gap = st.session_state.get('egap_global', 1.0)
            new_stage.append({
                "custom_name": model, "model": model, "gap": current_gap, "row": row
            })
    st.session_state['explore_stage'] = new_stage

def clear_stage():
    """Callback to safely clear everything BEFORE the page reruns."""
    st.session_state['explore_stage'] = []
    st.session_state['search_widget'] = []

# --- TAB: EXPLORE ---
with tab_explore:

    st.caption("Search for specific foams to compare them side-by-side against a common gap target.")
    
    # --- 0. PREPARE SEARCH DATA ---
    df['Search_Label'] = df['Manufacturer'] + " | " + df['Model']
    search_options = sorted(df['Search_Label'].unique().tolist())
    
    if 'search_widget' not in st.session_state:
        st.session_state['search_widget'] = [
            f"{item['row']['Manufacturer']} | {item['model']}" 
            for item in st.session_state['explore_stage']
        ]

    # --- 1. TOP INPUTS ---
    
    c_gap, c_tol, c_mode, c_clear = st.columns([1, 1, 1, 1])
    with c_gap:
        e_gap = st.number_input("Nominal Gap (mm)", value=0.400, step=0.010, format="%.3f", key="egap_global")
    with c_tol:
        e_tol = st.number_input("Tolerance (± mm)", value=0.100, step=0.005, format="%.3f", key="etol_global")
    with c_mode:
        st.metric("Calculation Mode", unit_mode.split()[0], delta="Active")
    with c_clear:
        st.write("") 
        st.button("🗑️ Clear Stage", use_container_width=True, on_click=clear_stage)

        st.write("")

        st.multiselect(
        "Foam name", 
        search_options, 
        key="search_widget", 
        on_change=update_explore_stage,
        placeholder="Type vendor or model (e.g. 'Poron' or '4701')..."
    )

    # --- 2. VISUALIZATION & RESULTS ---
    if st.session_state['explore_stage']:
        
        # --- A. Performance Chart (With Markers) ---
        
        fig_exp = go.Figure()
        fig_exp.add_vrect(
            x0=e_gap - e_tol, x1=e_gap + e_tol, 
            fillcolor="rgba(100,100,100,0.1)", line_width=0,
            annotation_text="Gap tolerance", annotation_position="top left"
        )

        px_range = np.linspace(0.1, 4.0, 200)
        explore_results = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, item in enumerate(st.session_state['explore_stage']):
            color = colors[i % len(colors)]
            row = item['row']
            vn, sn = get_value_with_status(row, e_gap, unit_mode, area, False)
            v_min, s_min = get_value_with_status(row, e_gap - e_tol, unit_mode, area, True)
            v_max, s_max = get_value_with_status(row, e_gap + e_tol, unit_mode, area, True)
            
            d_min = f"⚠️ {v_min:.3f}" if s_min == "Extrapolated" else f"{v_min:.3f}"
            d_max = f"⚠️ {v_max:.3f}" if s_max == "Extrapolated" else f"{v_max:.3f}"

            explore_results.append({
                "Foam Name": item['custom_name'],
                "Vendor": row['Manufacturer'], "Model": row['Model'], "Thk": row['thickness'],
                f"Nom {mode_label}": vn, "Min Gap Val": d_min, "Max Gap Val": d_max,
                "Add to Export": False, "row_ref": row
            })

            # Curve
            py = [get_value_with_status(row, tx, unit_mode, area, False)[0] for tx in px_range]
            fig_exp.add_trace(go.Scatter(
                x=px_range, y=[y if y > 0 else None for y in py], 
                name=item['custom_name'], mode='lines', line=dict(color=color)
            ))

            # Markers
            fig_exp.add_trace(go.Scatter(
                x=[e_gap],
                y=[vn],
                mode='markers',
                marker=dict(size=8, symbol=['circle'], color=color),
                showlegend=False, hoverinfo='skip'
            ))

        fig_exp.update_layout(
            template="plotly_white", height=450, margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified", xaxis_title="Gap (mm)", yaxis_title=unit_mode,
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_exp, use_container_width=True)
        
        # --- B. Results Table (Left Aligned via Strings) ---
        
        exp_display_df = pd.DataFrame(explore_results).drop(columns=['row_ref'])
        exp_display_df['Thk'] = exp_display_df['Thk'].map('{:.3f}'.format)
        exp_display_df[f"Nom {mode_label}"] = exp_display_df[f"Nom {mode_label}"].map('{:.3f}'.format)

        edited_exp_df = st.data_editor(
            exp_display_df,
            column_config={
                "Foam Name": st.column_config.TextColumn("Foam Name", width="medium"),
                "Thk": st.column_config.TextColumn("Thk (mm)"),
                f"Nom {mode_label}": st.column_config.TextColumn(f"{mode_label} @ {e_gap:.2f}mm"),
                "Min Gap Val": st.column_config.TextColumn(f"{mode_label} @ {e_gap - e_tol:.2f}mm"),
                "Max Gap Val": st.column_config.TextColumn(f"{mode_label} @ {e_gap + e_tol:.2f}mm"),
                "Add to Export": st.column_config.CheckboxColumn("Add", default=False)
            },
            disabled=["Vendor", "Model", "Thk", f"Nom {mode_label}", "Min Gap Val", "Max Gap Val"],
            use_container_width=True, hide_index=True, key="explore_editor"
        )

        # --- C. Export Logic ---
        for i, row in edited_exp_df.iterrows():
            r_orig = explore_results[i] # Grab raw data
            add_key = f"exp_added_{r_orig['Model']}_{i}"
            
            if row["Add to Export"] and not st.session_state.get(add_key, False):
                if not any(item['Model'] == r_orig['Model'] and item['Foam'] == row['Foam Name'] for item in st.session_state['export_basket']):
                    st.session_state['export_basket'].append({
                        "Foam": row["Foam Name"], "Vendor": r_orig['Vendor'], "Model": r_orig['Model'],
                        "Thk (mm)": round(r_orig['Thk'], 3), "Nom Gap (mm)": round(e_gap, 3),
                        f"Nom {unit_mode}": round(r_orig[f"Nom {mode_label}"], 3),
                        "Min Gap (mm)": round(e_gap - e_tol, 3), f"Min Gap {unit_mode}": r_orig['Min Gap Val'], 
                        "Max Gap (mm)": round(e_gap + e_tol, 3), f"Max Gap {unit_mode}": r_orig['Max Gap Val']
                    })
                    st.session_state[add_key] = True
                    st.toast(f"✅ Added {row['Foam Name']} to basket!")
            elif not row["Add to Export"]:
                st.session_state[add_key] = False
    else:
        st.info("Use the search bar to add foams for comparison.")

# --- TAB : EXPORT ---
with tab_export:
    st.caption("Export foams in the table below to a .xlsx file.")
    
    if st.session_state['export_basket']:
        edited_basket = st.data_editor(
            pd.DataFrame(st.session_state['export_basket']),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic", 
            key="export_editor_final"
        )

        if len(edited_basket) != len(st.session_state['export_basket']):
            st.session_state['export_basket'] = edited_basket.to_dict('records')
            st.rerun()

        st.divider()
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(st.session_state['export_basket']).to_excel(writer, index=False)
        
        st.download_button("📥 Download Excel Report", output.getvalue(), "Foam_Report.xlsx", type="primary")
        
        if st.button("🗑️ Clear Entire Basket"):
            st.session_state['export_basket'] = []
            st.rerun()
    else:
        st.info("Basket is empty.")