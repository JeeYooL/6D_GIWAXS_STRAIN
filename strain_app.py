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

# --- 헬퍼 함수: 파일명에서 입사각 추출 ---
def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

# --- 페이지 설정 ---
st.set_page_config(page_title="UNIST 6D GIWAXS Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (이미지 반전 & 계산식 복구)")

# --- 사이드바: 기하학 설정 (UNIST 매뉴얼 기준) ---
st.sidebar.header("1. Beamline Setup (Igor DB값)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=11.564, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=3057.064, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=78.125)
# 매뉴얼: MATLAB Center - 1
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=1440.36)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=1053.49)

# 물리량 계산
wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

st.sidebar.divider()
st.sidebar.header("2. Analysis Parameters")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, format="%.4f")
q_min = st.sidebar.number_input("Fit Start q", value=1.4)
q_max = st.sidebar.number_input("Fit End q", value=1.8)

# --- 파일 업로드 ---
uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    
    # 세션 상태 초기화 (결과 보존용)
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'zip_data' not in st.session_state: st.session_state.zip_data = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor")

    if st.button("🚀 전수 분석 실행", type="primary"):
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 파일 선저장
        saved_paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(saved_paths[uf.name], "wb") as f:
                f.write(uf.getbuffer())

        # pyFAI 엔진 설정
        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results = []
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(saved_paths[row["파일명"]]).data
                    # [이미지 반전] 배열 자체를 상하 반전시킴
                    flipped_img = np.flipud(img_data)
                    
                    # 1D 적분 (Vertical 방향 섹터 적분 권장)
                    q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1")
                    
                    # Origin용 텍스트 생성
                    txt = pd.DataFrame({"q": q, "I": I}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_origin.txt", txt)

                    # Fitting
                    mask = (q >= q_min) & (q <= q_max)
                    q_c, I_c = q[mask], I[mask]
                    
                    if len(q_c) < 5: continue
                        
                    model = GaussianModel()
                    out = model.fit(I_c, model.guess(I_c, x=q_c), x=q_c)
                    q_exp = out.params['center'].value
                    
                    # [계산식 복구] 이전의 표준 Strain 공식 사용
                    strain = (q_bulk - q_exp) / q_exp * 100
                    
                    results.append({
                        "파일명": row["파일명"],
                        "입사각": row["입사각(deg)"],
                        "q_measured": q_exp,
                        "Strain(%)": strain
                    })
                    
                    with st.expander(f"📊 {row['파일명']} 상세 (이미지 반전 적용)"):
                        c1, c2 = st.columns(2)
                        with c1:
                            # 반전된 이미지 출력
                            fig2d, ax2d = plt.subplots()
                            im = ax2d.imshow(np.log1p(np.clip(flipped_img, 0, None)), cmap='jet')
                            ax2d.set_title("2D GIWAXS (Vertical Flipped)")
                            plt.colorbar(im, ax=ax2d)
                            st.pyplot(fig2d)
                        with c2:
                            # 1D 피팅 결과
                            fig1d, ax1d = plt.subplots()
                            ax1d.plot(q, I, 'k-', alpha=0.1)
                            ax1d.plot(q_c, I_c, 'bo', markersize=2)
                            ax1d.plot(q_cut, out.best_fit, 'r-')
                            ax1d.set_title(f"Peak: {q_exp:.4f} / Strain: {strain:.3f}%")
                            st.pyplot(fig1d)
                            
                except Exception as e:
                    st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()

    # --- 결과 출력 영역 (Duplicate Key 방지 처리) ---
    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        st.divider()
        st.subheader("📊 분석 결과 및 트렌드")
        
        col_res, col_plt = st.columns([1, 1.5])
        with col_res:
            st.dataframe(res_df.style.format({"Strain(%)": "{:.3f}"}))
            # 고유 Key 부여로 Duplicate Element 에러 방지
            st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), 
                               "strain_summary.csv", key="btn_download_csv_final")
            st.download_button("📂 Origin용 TXT(.zip) 저장", st.session_state.zip_data, 
                               "origin_data.zip", key="btn_download_zip_final")

        with col_plt:
            fig_tr, ax_tr = plt.subplots()
            ax_tr.plot(res_df["입사각"], res_df["Strain(%)"], 'ro-', linewidth=2)
            ax_tr.set_xlabel("Incidence Angle (deg)")
            ax_tr.set_ylabel("Strain (%)")
            ax_tr.grid(True, alpha=0.3)
            st.pyplot(fig_tr)