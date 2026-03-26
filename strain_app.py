import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from lmfit.models import GaussianModel, LinearModel
import re
import zipfile
import io

# --- 헬퍼 함수 ---
def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

st.set_page_config(page_title="UNIST 6D GIWAXS Dual Analyzer", layout="wide")
st.title("🔬 6D GIWAXS 수직/수평 Strain 동시 분석")

# --- 사이드바: 실험 셋업 ---
st.sidebar.header("1. 실험 셋업 (6D UNIST-PAL)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=11.564, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=100.0, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=78.13)

st.sidebar.divider()
st.sidebar.subheader("🎯 빔 센터(Beam Center) 설정")
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=1440.36)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=1053.49)

wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

st.sidebar.divider()
st.sidebar.header("2. 분석 파라미터")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.5420, format="%.4f")
q_min = st.sidebar.number_input("Fit 영역 시작 q", value=1.40)
q_max = st.sidebar.number_input("Fit 영역 끝 q", value=1.80)

# [추가] 수직/수평 각도 범위 설정
st.sidebar.subheader("🎯 적분 각도(Azimuth) 설정")
v_width = st.sidebar.slider("수직(Vertical) 적분 폭 (±°)", 1, 30, 10, help="수직 아래(-90°) 기준 폭")
h_width = st.sidebar.slider("수평(Horizontal) 적분 폭 (±°)", 1, 30, 10, help="수평 좌우(-180°, 0°) 기준 폭")

# --- 파일 업로드 ---
st.sidebar.subheader("📂 데이터 업로드")
use_sample_data = st.sidebar.checkbox("✅ 샘플 데이터 테스트", value=False)

uploaded_files = []
if use_sample_data:
    sample_dir = "sample_data"
    if os.path.exists(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.lower().endswith(('.tif', '.tiff')):
                fpath = os.path.join(sample_dir, fname)
                with open(fpath, "rb") as f:
                    file_obj = io.BytesIO(f.read()); file_obj.name = fname
                    uploaded_files.append(file_obj)
else:
    uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'zip_data' not in st.session_state: st.session_state.zip_data = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor_dual")

    if st.button("🚀 수직/수평 동시 분석 시작", type="primary"):
        temp_dir = "temp_giwaxs"
        os.makedirs(temp_dir, exist_ok=True)
        paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(paths[uf.name], "wb") as f: f.write(uf.getbuffer())

        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results, zip_buffer = [], io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(paths[row["파일명"]]).data
                    
                    # --- [추가] 수직 및 수평 적분 수행 ---
                    # 수직(Out-of-plane): -90도 중심 
                    q_v, I_v = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(-90-v_width, -90+v_width))
                    # 수평(In-plane): 0도 및 -180도 부근 (데이터가 있는 우측 하단 0도 기준 예시) 
                    q_h, I_h = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(-h_width, 0))

                    # 공통 피팅 함수
                    def fit_peak(q, I):
                        mask = (q >= q_min) & (q <= q_max)
                        qc, Ic = q[mask], I[mask]
                        if len(qc) < 5: return None, None
                        model = GaussianModel() + LinearModel()
                        params = model.make_params(amplitude=Ic.max()-Ic.min(), center=(q_min+q_max)/2, sigma=0.05, slope=0, intercept=Ic.min())
                        params['center'].set(min=q_min, max=q_max)
                        out = model.fit(Ic, params, x=qc)
                        return out.params['center'].value, out

                    qv_exp, out_v = fit_peak(q_v, I_v)
                    qh_exp, out_h = fit_peak(q_h, I_h)

                    strain_v = (q_bulk - qv_exp) / qv_exp * 100 if qv_exp else np.nan
                    strain_h = (q_bulk - qh_exp) / qh_exp * 100 if qh_exp else np.nan

                    results.append({
                        "파일명": row["파일명"], "입사각": row["입사각(deg)"],
                        "V_Peak": qv_exp, "V_Strain(%)": strain_v,
                        "H_Peak": qh_exp, "H_Strain(%)": strain_h
                    })

                    # 상세 결과 시각화
                    with st.expander(f"📊 {row['파일명']} 상세 (V/H 비교)"):
                        col1, col2, col3 = st.columns([1.2, 1, 1])
                        with col1:
                            # 2D 이미지 (Flip 및 q-축 적용)
                            h_px, w_px = img_data.shape
                            dq = (2*np.pi/(wavelength*1e10)) * (px_m/dist_m)
                            ext = [-dbx*dq, (w_px-dbx)*dq, -dby*dq, (h_px-dby)*dq]
                            fig2d, ax2d = plt.subplots()
                            im = ax2d.imshow(np.log1p(np.clip(np.flipud(img_data), 0, None)), cmap='jet', extent=ext)
                            ax2d.set_title("2D GIWAXS (Flipped)"); ax2d.set_xlabel(r"$q_{xy}$"); ax2d.set_ylabel(r"$q_z$")
                            st.pyplot(fig2d); plt.close(fig2d)
                        with col2:
                            # 수직 피팅 그래프
                            fig_v, ax_v = plt.subplots(); ax_v.plot(q_v, I_v, 'bo', markersize=2, label='Data (V)')
                            if out_v: ax_v.plot(q_v[(q_v>=q_min)&(q_v<=q_max)], out_v.best_fit, 'r-')
                            ax_v.set_title(f"Vertical\nStrain: {strain_v:.3f}%"); ax_v.set_xlim(q_min-0.1, q_max+0.1)
                            st.pyplot(fig_v); plt.close(fig_v)
                        with col3:
                            # 수평 피팅 그래프
                            fig_h, ax_h = plt.subplots(); ax_h.plot(q_h, I_h, 'go', markersize=2, label='Data (H)')
                            if out_h: ax_h.plot(q_h[(q_h>=q_min)&(q_h<=q_max)], out_h.best_fit, 'r-')
                            ax_h.set_title(f"Horizontal\nStrain: {strain_h:.3f}%"); ax_h.set_xlim(q_min-0.1, q_max+0.1)
                            st.pyplot(fig_h); plt.close(fig_h)

                except Exception as e: st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()

    # --- 최종 결과 트렌드 ---
    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        st.divider()
        st.subheader("📈 입사각별 수직/수평 Strain 비교")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.dataframe(res_df.style.format({"V_Strain(%)": "{:.3f}", "H_Strain(%)": "{:.3f}"}))
            st.download_button("💾 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "dual_strain_results.csv", key="dual_csv")
            
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(res_df["입사각"], res_df["V_Strain(%)"], 'ro-', label='Vertical (Out-of-plane)')
            ax.plot(res_df["입사각"], res_df["H_Strain(%)"], 'bs-', label='Horizontal (In-plane)')
            ax.set_xlabel("Incidence Angle (deg)"); ax.set_ylabel("Strain (%)")
            ax.legend(); ax.grid(True, alpha=0.5)
            st.pyplot(fig)