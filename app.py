import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from io import BytesIO

# --- 0. LOOK AND FEEL ---

st.set_page_config(
    layout="wide", 
    page_title="Mr. Foamtastic", 
    page_icon="logo1.jpeg"  # Ensure this file is in your GitHub root
)

st.logo("logo.jpeg", size="large")

st.markdown("""
    <style>
    /* Force the logo container to be more prominent */
    [data-testid="stLogo"] {
        height: 6rem; /* Increases the height of the logo area */
        width: auto;
    }
    
    /* Industrial headers */
    h1, h2, h3 {
        font-weight: 800 !important;
        color: #0E1117 !important;
        letter-spacing: -0.5px;
    }
    
    /* Control panel sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #E6E9EF;
    }
    
    /* Polished dataframes */
    .stDataFrame {
        border: 1px solid #F0F2F6;
        border-radius: 8px;
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
st.set_page_config(layout="wide", page_title="Mr. Foamtastic", page_icon="🧪")

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
st.sidebar.title("Mr. Foamtastic")
st.sidebar.caption("V6.3 | #Slack Ishan Gokhale")
st.sidebar.divider()

unit_mode = st.sidebar.radio("Calculation Mode", ["Stress (MPa)", "Force (N)"])
if unit_mode == "Force (N)":
    area = st.sidebar.number_input("Contact Area (mm²)", value=20.0, step=1.0)
    v_def_min, v_def_max = 0.50, 2.00
else:
    area = 1.0 
    v_def_min, v_def_max = 0.080, 0.120

st.sidebar.divider()
st.sidebar.header("Database Filters")
mfr_options = sorted(df['Manufacturer'].unique().tolist()) if not df.empty else []
sel_mfrs = st.sidebar.multiselect("Vendors", mfr_options, default=mfr_options)
want_cond = st.sidebar.checkbox("Conductive Only")
want_psa = st.sidebar.checkbox("With PSA Only")

st.sidebar.divider()
st.sidebar.header("Refinement Filters")
t_min_in = st.sidebar.number_input("Min Thickness (mm)", value=float(df['thickness'].min() if not df.empty else 0.0), format="%.3f")
t_max_in = st.sidebar.number_input("Max Thickness (mm)", value=float(df['thickness'].max() if not df.empty else 5.0), format="%.3f")
v_min_in = st.sidebar.number_input(f"Min {unit_mode}", value=v_def_min, format="%.3f" if unit_mode == "Stress (MPa)" else "%.2f")
v_max_in = st.sidebar.number_input(f"Max {unit_mode}", value=v_def_max, format="%.3f" if unit_mode == "Stress (MPa)" else "%.2f")
c_min_in = st.sidebar.number_input("Min Compression %", value=20, step=1)
c_max_in = st.sidebar.number_input("Max Compression %", value=70, step=1)

# --- 6. TABS (EXPLORE, SELECT, EXPORT) ---
tab_explore, tab_select, tab_export = st.tabs(["EXPLORE", "SELECT", "EXPORT"])

with tab_explore:
    st.header("Failure Analysis & Simulation")
    e_tol = st.number_input("System Tolerance (± mm)", value=0.100, step=0.005, format="%.3f", key="etol")
    exp_col1, exp_col2 = st.columns([1, 2])
    
    with exp_col1:
        st.subheader("Add Foam to Stage")
        e_mfr = st.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
        e_series = st.selectbox("Series", sorted(df[df['Manufacturer'] == e_mfr]['Series'].unique()))
        e_model = st.selectbox("Model", sorted(df[(df['Manufacturer'] == e_mfr) & (df['Series'] == e_series)]['Model'].unique()))
        e_gap = st.number_input("Nominal Gap (mm)", value=1.000, step=0.010, format="%.3f", key="egap")
        
        if st.button("➕ Add to Stage"):
            row = df[df['Model'] == e_model].iloc[0]
            st.session_state['explore_stage'].append({"row": row, "gap": e_gap, "custom_name": e_model})
        
        for idx, item in enumerate(st.session_state['explore_stage']):
            c_e1, c_e2, c_e3 = st.columns([2, 2, 1])
            st.session_state['explore_stage'][idx]['custom_name'] = c_e1.text_input(f"Foam Name {idx+1}", value=item['custom_name'], key=f"exp_name_{idx}")
            c_e2.write(f"({item['row']['Model']})")
            if c_e3.button("❌", key=f"rm_exp_{idx}"):
                st.session_state['explore_stage'].pop(idx); st.rerun()

    with exp_col2:
        if st.session_state['explore_stage']:
            fig_exp = go.Figure()
            
            # Graph visual improvements
            
            fig_exp.update_layout(
                template="plotly_white",  # Perfect for screenshots
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20), # Tight margins for better screenshots
                xaxis_title="<b>Gap (mm)</b>", # Bold titles pop more in reports
                yaxis_title=f"<b>{unit_mode}</b>",
                
                # Custom color palette matching your logo bubbles
                colorway=["#00C9FF","#FF4B4B","#00D26A","#FF8700","#7030A0","#262730"]
                
                xaxis=dict(
                    showline=True, linewidth=2, linecolor='black',
                    gridcolor='#F0F2F6', dtick=0.2, ticks="outside"
                ),
                yaxis=dict(
                    showline=True, linewidth=2, linecolor='black',
                    gridcolor='#F0F2F6', ticks="outside"
                ),
                legend=dict(
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="Black",
                    borderwidth=1
                )
            )

            ref_gap = st.session_state['explore_stage'][0]['gap']
            fig_exp.add_vrect(x0=ref_gap - e_tol, x1=ref_gap + e_tol, fillcolor="rgba(100,100,100,0.1)", line_width=0)
            stage_table = []
            for item in st.session_state['explore_stage']:
                row, g, cname = item['row'], item['gap'], item['custom_name']
                vn, sn = get_value_with_status(row, g, unit_mode, area, False)
                v_min, s_min = get_value_with_status(row, g - e_tol, unit_mode, area, True)
                v_max, s_max = get_value_with_status(row, g + e_tol, unit_mode, area, True)
                
                stage_table.append({
                    "Vendor": row['Manufacturer'], "Foam": cname, "Model": row['Model'], "Thk (mm)": f"{row['thickness']:.3f}",
                    "Nom Gap (mm)": f"{g:.3f}", f"Nom {unit_mode}": format_val(vn, sn, unit_mode),
                    "Min Gap (mm)": f"{g - e_tol:.3f}", f"Min Gap {unit_mode}": format_val(v_min, s_min, unit_mode),
                    "Max Gap (mm)": f"{g + e_tol:.3f}", f"Max Gap {unit_mode}": format_val(v_max, s_max, unit_mode)
                })
                px_range = np.linspace(0.1, 4.0, 200)
                py = [get_value_with_status(row, tx, unit_mode, area, False)[0] for tx in px_range]
                fig_exp.add_trace(go.Scatter(x=px_range, y=[y if y > 0 else None for y in py], name=f"{cname} ({row['Model']})", mode='lines'))
            
            st.plotly_chart(fig_exp, use_container_width=True)
            st.dataframe(pd.DataFrame(stage_table).rename(index=lambda x: x + 1), use_container_width=True)
            
            if st.button("📤 Add Stage to Export"):
                for item in st.session_state['explore_stage']:
                    r, g, cname = item['row'], item['gap'], item['custom_name']
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
                st.toast("Stage exported!")

with tab_select:
    st.header("Search & Selection")
    s_gap = st.number_input("Target Nominal Gap (mm)", value=1.000, step=0.010, format="%.3f", key="sgap_s")
    s_tol = st.number_input("Gap Tolerance (± mm)", value=0.100, step=0.010, format="%.3f", key="stol_s")

    mask = (df['Manufacturer'].isin(sel_mfrs)) & (df['thickness'] >= t_min_in) & (df['thickness'] <= t_max_in)
    if want_cond: mask &= (df['Conductive'] == True)
    if want_psa: mask &= (df['PSA'] == True)
    filtered_df = df[mask]
    
    results = []
    for _, row in filtered_df.iterrows():
        vn, sn = get_value_with_status(row, s_gap, unit_mode, area, False)
        comp = ((row['thickness'] - s_gap) / row['thickness']) * 100
        if sn == "Interpolated" and v_min_in <= vn <= v_max_in and c_min_in <= comp <= c_max_in:
            v_min, s_min = get_value_with_status(row, s_gap - s_tol, unit_mode, area, True)
            v_max, s_max = get_value_with_status(row, s_gap + s_tol, unit_mode, area, True)
            results.append({
                "Vendor": row['Manufacturer'], "Model": row['Model'], "Thk (mm)": row['thickness'],
                "Nom Gap (mm)": s_gap, f"Nom {unit_mode}": vn, "v_min": v_min, "s_min": s_min,
                "v_max": v_max, "s_max": s_max, "row_ref": row
            })

    if results:
        disp_res = []
        for r in results:
            disp_res.append({
                "Vendor": r['Vendor'], "Model": r['Model'], "Thk (mm)": f"{r['Thk (mm)']:.3f}",
                "Nom Gap (mm)": f"{r['Nom Gap (mm)']:.3f}", f"Nom {unit_mode}": format_val(r[f"Nom {unit_mode}"], "Interpolated", unit_mode),
                "Min Gap (mm)": f"{s_gap - s_tol:.3f}", f"Min Gap {unit_mode}": format_val(r['v_min'], r['s_min'], unit_mode),
                "Max Gap (mm)": f"{s_gap + s_tol:.3f}", f"Max Gap {unit_mode}": format_val(r['v_max'], r['s_max'], unit_mode)
            })
        st.dataframe(pd.DataFrame(disp_res).rename(index=lambda x: x + 1), use_container_width=True)

        fig_sel = go.Figure()
        
         # Graph visual improvements
        fig_sel.update_layout(
            template="plotly_white",  # Perfect for screenshots
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20), # Tight margins for better screenshots
            xaxis_title="<b>Gap (mm)</b>", # Bold titles pop more in reports
            yaxis_title=f"<b>{unit_mode}</b>",
            
            # Custom color palette matching your logo bubbles
            colorway=["#00C9FF","#FF4B4B","#00D26A","#FF8700","#7030A0","#262730"]
            
            xaxis=dict(
                showline=True, linewidth=2, linecolor='black',
                gridcolor='#F0F2F6', dtick=0.2, ticks="outside"
            ),
            yaxis=dict(
                showline=True, linewidth=2, linecolor='black',
                gridcolor='#F0F2F6', ticks="outside"
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="Black",
                borderwidth=1
            )
        )
        
        fig_sel.add_vrect(x0=s_gap - s_tol, x1=s_gap + s_tol, fillcolor="rgba(100,100,100,0.1)", line_width=0)
        px_range = np.linspace(0.1, 4.0, 200)
        for r in results:
            py = [get_value_with_status(r['row_ref'], tx, unit_mode, area, False)[0] for tx in px_range]
            fig_sel.add_trace(go.Scatter(x=px_range, y=[y if y > 0 else None for y in py], name=r['Model'], mode='lines'))
        st.plotly_chart(fig_sel, use_container_width=True)
        for r in results:
            col1, col2, col3 = st.columns([3, 2, 2])
            col1.write(f"**{r['Model']}**")
            foam_label = col2.text_input("Foam Name", key=f"fn_sel_{r['Model']}", placeholder="e.g. FCAM B2B")
            if col3.button("Add to Export", key=f"sb_sel_{r['Model']}"):
                st.session_state['export_basket'].append({
                    "Foam": foam_label if foam_label else r['Model'], "Vendor": r['Vendor'], "Model": r['Model'],
                    "Thk (mm)": round(r['Thk (mm)'], 3), "Nom Gap (mm)": round(r['Nom Gap (mm)'], 3), 
                    f"Nom {unit_mode}": round(r[f"Nom {unit_mode}"], 3),
                    "Min Gap (mm)": round(r['Nom Gap (mm)'] - s_tol, 3), f"Min Gap {unit_mode}": round(r['v_min'], 3),
                    "Max Gap (mm)": round(r['Nom Gap (mm)'] + s_tol, 3), f"Max Gap {unit_mode}": round(r['v_max'], 3)
                })
                st.toast(f"Added {r['Model']}!")

with tab_export:
    st.header("Finalize Selection Report")
    if st.session_state['export_basket']:
        export_df = pd.DataFrame(st.session_state['export_basket'])
        st.dataframe(export_df.rename(index=lambda x: x + 1), use_container_width=True)
        
        for idx, _ in enumerate(st.session_state['export_basket']):
            if st.button(f"❌ Remove Item {idx+1}", key=f"rm_final_{idx}"):
                st.session_state['export_basket'].pop(idx); st.rerun()
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, index=False)
        st.download_button("📥 Download Final Excel Report", output.getvalue(), "Mr_Foamtastic_Export.xlsx")
    else:
        st.info("Basket is empty.")