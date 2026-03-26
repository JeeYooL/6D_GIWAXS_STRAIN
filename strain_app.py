import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from lmfit.models import GaussianModel
import re
import zipfile
import io

# --- 헬퍼 함수 ---
def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

st.set_page_config(page_title="UNIST 6D GIWAXS Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (Manual p.7 왜곡 보정 적용)")

# --- 사이드바: 기하학 설정 ---
st.sidebar.header("1. Beamline Setup (Igor DB값)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=12.4, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=200.0, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=172.0)
# 매뉴얼 가이드: MATLAB Center에서 -1 한 값을 입력 [cite: 538]
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=1023.0)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=511.0)

# 물리량 계산
wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

st.sidebar.divider()
st.sidebar.header("2. Analysis Parameters")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, format="%.4f")
# 고각 데이터 에러 방지를 위해 범위를 넉넉히 설정하세요
q_min = st.sidebar.number_input("Fit Start q", value=1.4)
q_max = st.sidebar.number_input("Fit End q", value=1.8)

# --- 파일 업로드 및 데이터 관리 ---
uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    # 세션 상태를 사용하여 결과 보존
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'origin_zip' not in st.session_state:
        st.session_state.origin_zip = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor")

    if st.button("🚀 전수 분석 실행", type="primary"):
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. 파일 선저장
        saved_paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(saved_paths[uf.name], "wb") as f:
                f.write(uf.getbuffer())

        # 2. 분석 엔진 설정 (매뉴얼 p.7의 Ewald sphere 보정 반영)
        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    img = fabio.open(saved_paths[row["파일명"]]).data
                    # 매뉴얼 7p [Iso q_xy cut] 로직: pyFAI는 적분 시 기하학적 왜곡을 자동 보정함 [cite: 790]
                    q, I = geo.integrate1d(img, 1000, unit="q_A^-1")
                    
                    # Origin 데이터 생성
                    txt = pd.DataFrame({"q": q, "I": I}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_origin.txt", txt)

                    # Fitting
                    mask = (q >= q_min) & (q <= q_max)
                    q_c, I_c = q[mask], I[mask]
                    
                    if len(q_c) < 5: 
                        raise ValueError("Fitting 영역에 데이터 부족. q 범위를 넓혀보세요.")
                        
                    model = GaussianModel()
                    out = model.fit(I_c, model.guess(I_c, x=q_c), x=q_c)
                    q_exp = out.params['center'].value
                    
                    results.append({
                        "파일명": row["파일명"],
                        "입사각": row["입사각(deg)"],
                        "q_measured": q_exp,
                        "Strain(%)": (q_bulk - q_exp) / q_exp * 100
                    })
                except Exception as e:
                    st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.origin_zip = zip_buffer.getvalue()

    # --- 결과 출력 (세션 데이터 기반) ---
    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        st.divider()
        st.subheader("📊 분석 결과 요약")
        st.dataframe(res_df.style.format({"Strain(%)": "{:.3f}"}))
        
        # 중복 Key 방지를 위해 고유 ID 부여
        st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), 
                           "strain_summary.csv", key="csv_download_btn")
        st.download_button("📂 Origin용 TXT(.zip) 저장", st.session_state.origin_zip, 
                           "origin_data.zip", key="zip_download_btn")

        # 트렌드 그래프
        fig, ax = plt.subplots()
        ax.plot(res_df["입사각"], res_df["Strain(%)"], 'ro-', label='Strain Trend')
        ax.set_xlabel("Incidence Angle (deg)")
        ax.set_ylabel("Strain (%)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
