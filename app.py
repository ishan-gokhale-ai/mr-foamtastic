import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from io import BytesIO

# --- 0. LOOK AND FEEL ---

st.markdown("""
    <style>
    /* 1. Reset the sidebar padding to be clean but not zero */
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* 2. Set a consistent, small vertical gap between widgets */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.75rem !important;
    }

    /* 3. Style the labels to be closer to their inputs without overlapping */
    [data-testid="stSidebar"] label {
        margin-bottom: -5px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #31333F !important;
    }

    /* 4. Tighten the Divider lines */
    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* 5. Fix the Radio Button alignment */
    /* This ensures the 'Calculation Mode' doesn't stack on top of its label */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] + div {
        margin-top: 5px !important;
    }

    /* 6. Improve the Logo/Header spacing at the very top */
    [data-testid="stSidebar"] h2 {
        padding-top: 0.5rem !important;
        margin-bottom: 0rem !important;
    }
    /* 7. Remove the massive gap at the top of the main page */
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 2rem !important; /* Standard professional padding */
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    /* 8. Hide the default Streamlit header bar (removes top white strip) */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
        height: 0rem !important;
    }

    /* 9. Tighten the Tab styling */
    [data-testid="stTab"] {
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* 10. Fix sidebar logo alignment so it sits flush with the top */
    [data-testid="stSidebarContent"] {
        padding-top: 1rem !important;
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
        # First run, show input for password.
        st.set_page_config(page_title="Mr. Foamtastic Login", page_icon="🔒")
        st.title("🔒 Mr. Foamtastic V6.3")
        st.text_input("Enter Team Password", type="password", on_change=password_entered, key="password")
        st.info("Contact Ishan Gokhale for access.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.set_page_config(page_title="Mr. Foamtastic Login", page_icon="🔒")
        st.text_input("Enter Team Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
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
def load_data():
    try:
        # Simple path because GitHub and Streamlit Cloud keep files together
        df = pd.read_csv("foams.csv")
        df['Conductive'] = df['Conductive'].astype(str).str.upper().map({'YES': True, 'NO': False})
        df['PSA'] = df['PSA'].astype(str).str.upper().map({'YES': True, 'NO': False})
        return df
    except Exception as e:
        st.error(f"Error loading foams.csv: {e}")
        return pd.DataFrame()

df = load_data()

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

def format_val(val, status, mode):
    prec = 3 if "Stress" in mode or "mm" in mode else 2
    formatted = f"{val:.{prec}f}"
    return f"🔴 {formatted}" if status == "Extrapolated" else formatted

# --- 5. SIDEBAR ---
with st.sidebar:
    # 1. Header & Branding
    st.image("logo.png", use_container_width=True)
    st.caption("V6.3 | #Slack Ishan Gokhale")
    st.divider()

    # 2. Global Calculation Settings
    with st.expander("Stress vs Force Mode", expanded=False):
        unit_mode = st.radio(
            "Select Output Type", 
            ["Stress (MPa)", "Force (N)"],
            help="Stress is area-independent; Force requires contact area."
        )
        
        if unit_mode == "Force (N)":
            area = st.number_input("Contact area (mm²)", value=20.0, step=1.0, format="%.2f")
            v_def_min, v_def_max = 0.50, 2.00
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

tab_select, tab_explore, tab_export = st.tabs(["SELECT", "EXPLORE", "EXPORT"])

with tab_select:
    st.header("Foam Recommendation Engine")
    
    # --- 1. Top Input Row ---
    s_col1, s_col2, s_col3 = st.columns([1, 1, 1])
    with s_col1:
        s_gap = st.number_input("Target Nominal Gap (mm)", value=1.000, step=0.010, format="%.3f", key="sgap_s")
    with s_col2:
        s_tol = st.number_input("Gap Tolerance (± mm)", value=0.100, step=0.010, format="%.3f", key="stol_s")
    with s_col3:
        st.metric("Calculation Mode", unit_mode.split()[0], delta="Active")

    # --- 2. Filter & Calculation Logic ---
    mask = (df['Manufacturer'].isin(sel_mfrs)) & (df['thickness'] >= t_min_in) & (df['thickness'] <= t_max_in)
    if want_cond: mask &= (df['Conductive'] == True)
    if want_psa: mask &= (df['PSA'] == True)
    filtered_df = df[mask]
    
    results = []
    for _, row in filtered_df.iterrows():
        vn, sn = get_value_with_status(row, s_gap, unit_mode, area, False)
        comp = ((row['thickness'] - s_gap) / row['thickness']) * 100
        
        # Only suggest foams where the NOMINAL value is within range and interpolated
        if sn == "Interpolated" and v_min_in <= vn <= v_max_in and c_min_in <= comp <= c_max_in:
            v_min, s_min = get_value_with_status(row, s_gap - s_tol, unit_mode, area, True)
            v_max, s_max = get_value_with_status(row, s_gap + s_tol, unit_mode, area, True)
            
            # Format display values with warning icons for extrapolation
            d_min = f"⚠️ {v_min:.3f}" if s_min == "Extrapolated" else f"{v_min:.3f}"
            d_max = f"⚠️ {v_max:.3f}" if s_max == "Extrapolated" else f"{v_max:.3f}"

            results.append({
                "Foam Name": row['Model'], # Default name
                "Vendor": row['Manufacturer'], 
                "Model": row['Model'], 
                "Thk": row['thickness'],
                f"Nom {unit_mode.split()[0]}": vn, 
                "Min Gap Val": d_min, 
                "Max Gap Val": d_max, 
                "row_ref": row,
                "Add to Export": False 
            })

    if results:
        # --- 3. Full-Width Performance Chart ---
        st.subheader("Performance Visualization")
        fig_sel = go.Figure()
        fig_sel.add_vrect(x0=s_gap - s_tol, x1=s_gap + s_tol, fillcolor="rgba(100,100,100,0.1)", line_width=0, annotation_text="Tol Zone", annotation_position="top left")
        
        px_range = np.linspace(0.1, 4.0, 200)
        for r in results:
            py = [get_value_with_status(r['row_ref'], tx, unit_mode, area, False)[0] for tx in px_range]
            fig_sel.add_trace(go.Scatter(x=px_range, y=[y if y > 0 else None for y in py], name=r['Model'], mode='lines'))
        
        fig_sel.update_layout(template="plotly_white", height=400, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_sel, use_container_width=True)

        st.divider()

        # --- 4. Interactive Data Table ---
        st.subheader("Compatible Foams")
        st.caption("💡 Edit **Foam Name** to customize ID. Values marked ⚠️ are extrapolated from lab data.")
        
        edited_df = st.data_editor(
            pd.DataFrame(results).drop(columns=['row_ref']), 
         # Extract the base unit label (Stress or Force) for the headers
        mode_label = unit_mode.split()[0] # Result: "Stress" or "Force"

            column_config={
            "Foam Name": st.column_config.TextColumn("Foam Name (Editable)", width="medium"),
            "Thk": st.column_config.NumberColumn("Thk (mm)", format="%.3f"),
            f"Nom {mode_label}": st.column_config.NumberColumn(f"{mode_label} at Nominal Gap", format="%.3f"),
            "Min Gap Val": st.column_config.TextColumn(f"{mode_label} at Min Gap ({unit_mode.split('(')[1].replace(')', '')})"),
            "Max Gap Val": st.column_config.TextColumn(f"{mode_label} at Max Gap ({unit_mode.split('(')[1].replace(')', '')})"),
            "Add to Export": st.column_config.CheckboxColumn("Add", default=False)
            },
            disabled=["Vendor", "Model", "Thk", f"Nom {unit_mode.split()[0]}", "Min Gap Val", "Max Gap Val"],
            use_container_width=True,
            hide_index=True,
            key="selection_editor"
        )

        # --- 5. State-Safe Add Logic ---
        for i, row in edited_df.iterrows():
            # Unique key to track if THIS row in THIS specific search has been added
            row_id = f"add_state_{row['Model']}_{i}"
            
            if row["Add to Export"] and not st.session_state.get(row_id, False):
                r_orig = results[i]
                
                st.session_state['export_basket'].append({
                    "Foam": row["Foam Name"],
                    "Vendor": r_orig['Vendor'],
                    "Model": r_orig['Model'],
                    "Thk (mm)": round(r_orig['Thk'], 3),
                    "Nom Gap (mm)": round(s_gap, 3),
                    f"Nom {unit_mode}": round(r_orig[f"Nom {unit_mode.split()[0]}"], 3),
                    "Min Gap (mm)": round(s_gap - s_tol, 3),
                    f"Min Gap {unit_mode}": r_orig['Min Gap Val'], # Stores the string with ⚠️
                    "Max Gap (mm)": round(s_gap + s_tol, 3),
                    f"Max Gap {unit_mode}": r_orig['Max Gap Val']
                })
                
                st.session_state[row_id] = True
                st.toast(f"✅ Added {row['Foam Name']}!")
            
            elif not row["Add to Export"]:
                st.session_state[row_id] = False

    else:
        st.info("No foams match your search criteria. Adjust filters in the sidebar.")

with tab_explore:
    st.header("Foam Explorer")
    st.caption("Select a foam from the dropdown menus to check CFD")
    
    # --- DATA SANITIZER: Fixes the KeyError by forcing lowercase keys ---
    if st.session_state['explore_stage']:
        sanitized_stage = []
        for item in st.session_state['explore_stage']:
            # This logic looks for the old keys and maps them to the new ones
            sanitized_item = {
                "custom_name": item.get('custom_name') or item.get('Custom Name') or item.get('Model'),
                "model": item.get('model') or item.get('Model'),
                "gap": item.get('gap') or item.get('Gap') or 1.0,
                "row": item.get('row')
            }
            sanitized_stage.append(sanitized_item)
        st.session_state['explore_stage'] = sanitized_stage

    exp_col1, exp_col2 = st.columns([1, 2])
    
    with exp_col1:
        st.subheader("Stage Management")
        
        with st.expander("➕ Add New Foam to Stage", expanded=True):
            e_mfr = st.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
            e_series = st.selectbox("Series", sorted(df[df['Manufacturer'] == e_mfr]['Series'].unique()))
            e_model = st.selectbox("Model", sorted(df[(df['Manufacturer'] == e_mfr) & (df['Series'] == e_series)]['Model'].unique()))
            
            ec1, ec2 = st.columns(2)
            e_gap = ec1.number_input("Nominal Gap (mm)", value=1.000, step=0.010, format="%.3f", key="egap")
            e_tol = ec2.number_input("Tol (± mm)", value=0.100, step=0.005, format="%.3f", key="etol_input")

            if st.button("Add to Stage", use_container_width=True, type="primary"):
                row_data = df[df['Model'] == e_model].iloc[0]
                st.session_state['explore_stage'].append({
                    "custom_name": e_model, 
                    "model": e_model, 
                    "gap": e_gap, 
                    "row": row_data
                })
                st.rerun()

        st.divider()

        if st.session_state['explore_stage']:
            st.write("**Active Stage Items**")
            
            # Defensive DataFrame creation
            stage_df = pd.DataFrame(st.session_state['explore_stage'])
            
            # Ensure the columns exist before slicing to prevent the KeyError
            required_cols = ['custom_name', 'model', 'gap']
            if all(col in stage_df.columns for col in required_cols):
                editable_df = stage_df[required_cols]
                
                edited_stage = st.data_editor(
                    editable_df,
                    column_config={
                        "custom_name": st.column_config.TextColumn("Foam Name"),
                        "model": st.column_config.Column("Model", disabled=True),
                        "gap": st.column_config.NumberColumn("Gap (mm)", format="%.3f")
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="stage_editor"
                )

                # Sync changes back
                if len(edited_stage) != len(st.session_state['explore_stage']) or not edited_stage.equals(editable_df):
                    new_stage = []
                    for _, edited_row in edited_stage.iterrows():
                        model_val = edited_row.get('model')
                        if model_val and pd.notna(model_val):
                            matched_rows = df[df['Model'] == model_val]
                            if not matched_rows.empty:
                                new_stage.append({
                                    "row": matched_rows.iloc[0], 
                                    "gap": edited_row.get('gap', 1.0), 
                                    "custom_name": edited_row.get('custom_name', model_val)
                                })
                    st.session_state['explore_stage'] = new_stage
                    st.rerun()
            
            if st.button("🗑️ Clear Entire Stage", use_container_width=True):
                st.session_state['explore_stage'] = []
                st.rerun()

    with exp_col2:
        if not st.session_state['explore_stage']:
            st.info("👈 Add a foam from the 'Stage Management' panel to begin your simulation.")
        else:
            fig_exp = go.Figure()
            
            fig_exp.update_layout(
                template="plotly_white",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=80),
                xaxis_title="<b>Gap (mm)</b>",
                yaxis_title=f"<b>{unit_mode}</b>",
                colorway=["#00C9FF","#FF4B4B","#00D26A","#FF8700","#7030A0","#262730"],
                xaxis=dict(showline=True, linewidth=2, linecolor='black', gridcolor='#F0F2F6', dtick=0.2, ticks="outside"),
                yaxis=dict(showline=True, linewidth=2, linecolor='black', gridcolor='#F0F2F6', ticks="outside"),
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
            )

            # Defensive Gap access
            first_item = st.session_state['explore_stage'][0]
            ref_gap = first_item.get('gap') or first_item.get('Gap') or 1.0
            
            fig_exp.add_vrect(x0=ref_gap - e_tol, x1=ref_gap + e_tol, fillcolor="rgba(100,100,100,0.1)", line_width=0)
            
            stage_table_data = []
            px_range = np.linspace(0.1, 4.0, 200)

            for item in st.session_state['explore_stage']:
                # Re-sanitizing within loop to be safe
                row = item.get('row')
                g = item.get('gap') or item.get('Gap') or 1.0
                cname = item.get('custom_name') or item.get('Custom Name') or "Unknown"
                
                if row is not None:
                    vn, sn = get_value_with_status(row, g, unit_mode, area, False)
                    v_min, s_min = get_value_with_status(row, g - e_tol, unit_mode, area, True)
                    v_max, s_max = get_value_with_status(row, g + e_tol, unit_mode, area, True)
                    
                    stage_table_data.append({
                        "Vendor": row['Manufacturer'], 
                        "Foam": cname, 
                        "Model": row['Model'], 
                        "Thk (mm)": f"{row['thickness']:.3f}",
                        "Nom Gap (mm)": f"{g:.3f}", 
                        f"Nom {unit_mode}": format_val(vn, sn, unit_mode),
                        "Min Gap (mm)": f"{g - e_tol:.3f}", 
                        f"Min Gap {unit_mode}": format_val(v_min, s_min, unit_mode),
                        "Max Gap (mm)": f"{g + e_tol:.3f}", 
                        f"Max Gap {unit_mode}": format_val(v_max, s_max, unit_mode)
                    })
                    
                    py = [get_value_with_status(row, tx, unit_mode, area, False)[0] for tx in px_range]
                    fig_exp.add_trace(go.Scatter(
                        x=px_range, y=[y if y > 0 else None for y in py], 
                        name=f"{cname} ({row['Model']})", mode='lines'
                    ))
            
            st.plotly_chart(fig_exp, use_container_width=True)
            if stage_table_data:
                st.dataframe(pd.DataFrame(stage_table_data).rename(index=lambda x: x + 1), use_container_width=True)
            
            if st.button("📤 Add Stage to Export", use_container_width=True):
                for item in st.session_state['explore_stage']:
                    r = item.get('row')
                    g = item.get('gap') or item.get('Gap') or 1.0
                    cname = item.get('custom_name') or item.get('Custom Name') or "Unknown"
                    if r is not None:
                        v_n, s_n = get_value_with_status(r, g, unit_mode, area, False)
                        v_min, s_min = get_value_with_status(r, g - e_tol, unit_mode, area, True)
                        v_max, s_max = get_value_with_status(r, g + e_tol, unit_mode, area, True)
                        st.session_state['export_basket'].append({
                            "Foam": cname, "Vendor": r['Manufacturer'], "Model": r['Model'],
                            "Thk (mm)": round(r['thickness'], 3), "Nom Gap (mm)": round(g, 3), 
                            f"Nom {unit_mode}": round(v_n, 3),
                            "Min Gap (mm)": round(g - e_tol, 3), f"Min Gap {unit_mode}": round(v_min, 3), 
                            "Max Gap (mm)": round(g + e_tol, 3), f"Max Gap {unit_mode}": round(v_max, 3)
                        })
                st.toast("Stage added to Export Basket!")

with tab_export:
    st.header("Finalize Selection Report")
    
    if st.session_state['export_basket']:
        # We wrap the editor in a form or use a specific key to prevent instant re-runs
        # num_rows="dynamic" allows the user to select a row and hit 'Delete'
        edited_basket = st.data_editor(
            pd.DataFrame(st.session_state['export_basket']),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic", 
            key="export_editor_final"
        )

        # Update the basket only if the number of rows changed (a deletion happened)
        if len(edited_basket) != len(st.session_state['export_basket']):
            st.session_state['export_basket'] = edited_basket.to_dict('records')
            st.rerun()

        st.divider()
        
        # Download Button
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(st.session_state['export_basket']).to_excel(writer, index=False)
        
        st.download_button("📥 Download Excel Report", output.getvalue(), "Foam_Report.xlsx", type="primary")
        
        if st.button("🗑️ Clear Entire Basket"):
            st.session_state['export_basket'] = []
            st.rerun()
    else:
        st.info("Basket is empty.")