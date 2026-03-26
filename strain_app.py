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

st.set_page_config(page_title="UNIST 6D GIWAXS Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (이미지 반전 & 피팅 오류 수정)")

# --- 사이드바: 실험 셋업 ---
st.sidebar.header("1. 실험 셋업 (6D UNIST-PAL)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=11.564, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=200.0, format="%.3f") # SDD 200으로 기본값 수정
pixel_um = st.sidebar.number_input("Pixel size (um)", value=78.13)
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

st.sidebar.subheader("🎯 1D 적분 각도(Azimuth) 설정")
st.sidebar.info("💡 실제 데이터가 있는 하반원을 사용하려면 -180 ~ 0 범위를 권장합니다.")
azi_min = st.sidebar.number_input("최소 Azimuth (°)", value=-180) # 데이터가 있는 하반원 기준
azi_max = st.sidebar.number_input("최대 Azimuth (°)", value=0)

# --- 파일 업로드 ---
uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'zip_data' not in st.session_state: st.session_state.zip_data = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor_v2")

    if st.button("🚀 전수 분석 시작", type="primary"):
        temp_dir = "temp_giwaxs"
        os.makedirs(temp_dir, exist_ok=True)
        
        paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(paths[uf.name], "wb") as f: f.write(uf.getbuffer())

        # pyFAI 엔진 (물리적 계산용 - 원본 좌표계 유지)
        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results, zip_buffer = [], io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(paths[row["파일명"]]).data
                    
                    # [계산] 설정된 각도 범위로 1D 적분
                    q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_min, azi_max))
                    
                    # Origin 데이터 저장
                    txt = pd.DataFrame({"q": q, "I": I}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_1D.txt", txt)

                    # [피팅] Gaussian + Linear (배경 제거)
                    mask = (q >= q_min) & (q <= q_max)
                    q_c, I_c = q[mask], I[mask]
                    
                    if len(q_c) < 5: raise ValueError("데이터 부족")
                        
                    model = GaussianModel() + LinearModel()
                    # 초기값 설정 강화
                    params = model.make_params(amplitude=I_c.max()-I_c.min(), center=(q_min+q_max)/2, sigma=0.05, slope=0, intercept=I_c.min())
                    params['center'].set(min=q_min, max=q_max) # 범위 제한
                    
                    out = model.fit(I_c, params, x=q_c)
                    q_exp = out.params['center'].value
                    strain = (q_bulk - q_exp) / q_exp * 100
                    
                    results.append({
                        "파일명": row["파일명"],
                        "입사각": row["입사각(deg)"],
                        "q_measured": q_exp,
                        "Strain(%)": strain
                    })
                    
                    with st.expander(f"📊 {row['파일명']} 상세 분석 (이미지 반전 완료)"):
                        c1, c2 = st.columns(2)
                        with c1:
                            # [요청 반영] 시각적으로 하반원을 위로 Flip
                            img_flipped = np.flipud(img_data)
                            fig2d, ax2d = plt.subplots()
                            im = ax2d.imshow(np.log1p(np.clip(img_flipped, 0, None)), cmap='jet')
                            ax2d.set_title("Visualized GIWAXS (Bottom half at Top)")
                            plt.colorbar(im, ax=ax2d)
                            st.pyplot(fig2d)
                            plt.close(fig2d)
                        with c2:
                            # 1D 피팅 결과 시각화
                            fig1d, ax1d = plt.subplots()
                            ax1d.plot(q_c, I_c, 'bo', markersize=3, label='Data')
                            ax1d.plot(q_c, out.best_fit, 'r-', label='Gauss+Linear Fit')
                            ax1d.axvline(q_exp, color='r', linestyle='--', label=f'Peak: {q_exp:.4f}')
                            ax1d.axvline(q_bulk, color='g', linestyle='--', label=f'Bulk: {q_bulk:.4f}')
                            ax1d.set_title(f"Strain: {strain:.3f}%")
                            ax1d.legend()
                            ax1d.grid(True, alpha=0.3)
                            st.pyplot(fig1d)
                            plt.close(fig1d)
                            
                except Exception as e:
                    st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()

    # --- 최종 결과 트렌드 ---
    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        st.divider()
        st.subheader("📈 입사각별 Strain 트렌드")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.dataframe(res_df.style.format({"q_measured": "{:.4f}", "Strain(%)": "{:.3f}"}))
            st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "strain_results.csv", key="dl_csv")
            st.download_button("📂 Origin용 TXT 저장", st.session_state.zip_data, "origin_data.zip", key="dl_zip")
            
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(res_df["입사각"], res_df["Strain(%)"], 'ro-', label='Strain (%)')
            ax.set_xlabel("Incidence Angle (deg)")
            ax.set_ylabel("Strain (%)")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)