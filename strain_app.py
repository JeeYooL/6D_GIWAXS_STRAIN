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

# --- 굴절 보정 함수 (매뉴얼 p.7 원리) ---
def refraction_correction(q_z_measured, incidence_angle, critical_angle):
    # k0 = 2*pi / wavelength (Å^-1). 대략 1.5 전후 값.
    # 대략적인 보정을 위해 q_z_measured를 사용하여 보정값을 역산
    alpha_i = np.deg2rad(incidence_angle)
    alpha_c = np.deg2rad(critical_angle)
    
    # 굴절 보정 수식 (간소화 버전)
    factor = np.sqrt(alpha_i**2 - alpha_c**2) / alpha_i
    q_z_corrected = q_z_measured * factor
    return q_z_corrected

st.set_page_config(page_title="UNIST GIWAXS Strain Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (이미지 뒤집기 및 굴절 보정 완료)")

# --- 사이드바: 기하학 및 물질 설정 ---
st.sidebar.header("1. 실험 셋업 (6D UNIST-PAL)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=12.4, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=200.0, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=172.0)
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=1023.0)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=511.0)

# 물리량 계산
wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

st.sidebar.divider()
st.sidebar.header("2. 물질 및 분석 파라미터")
# Strain 계산의 기준
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, format="%.4f")
# 피팅 영역 지정 (넉넉히 설정 권장)
q_min = st.sidebar.number_input("Fit 영역 시작 q", value=1.2)
q_max = st.sidebar.number_input("Fit 영역 끝 q", value=2.0)
# 굴절 보정을 위한 임계각 (예: 페로브스카이트 ~0.14)
critical_angle = st.sidebar.number_input("물질 Critical Angle (deg)", value=0.14, format="%.3f")

# --- 파일 업로드 영역 ---
uploaded_files = st.sidebar.file_uploader("📂 TIF 데이터 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    
    # 세션 상태 초기화
    if 'strain_results' not in st.session_state: st.session_state.strain_results = None
    if 'zip_data' not in st.session_state: st.session_state.zip_data = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor_table")

    if st.button("🚀 전수 분석 시작", type="primary"):
        temp_dir = "temp_giwaxs"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. 파일 저장 및 경로 매핑
        paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(paths[uf.name], "wb") as f:
                f.write(uf.getbuffer())

        # 2. 분석 엔진 설정
        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results, zip_buffer = [], io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    f_full_path = paths[row["파일명"]]
                    img = fabio.open(f_full_path).data
                    
                    # [개선] Vertical 방향 섹터 적분 (예: 90도 근처 +-10도)
                    # out-of-plane Strain 분석을 위한 날카로운 피크 확보
                    q, I = geo.integrate1d(img, 1000, unit="q_A^-1", azimuth_range=(80, 100))
                    
                    # Origin 데이터 저장
                    txt_data = pd.DataFrame({"q": q, "I": I}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_Vertical.txt", txt_data)

                    # Fitting 범위 슬라이싱
                    mask = (q >= q_min) & (q <= q_max)
                    q_c, I_c = q[mask], I[mask]
                    
                    if len(q_c) < 5: raise ValueError("Fitting 범위 내 데이터 부족. q 영역을 넓혀보세요.")
                        
                    model = GaussianModel()
                    out = model.fit(I_c, model.guess(I_c, x=q_c), x=q_c)
                    q_raw = out.params['center'].value # 보정 전 측정값
                    
                    # [핵심 개선] 굴절 보정 적용 (Vertical Cut 기준)
                    q_exp = refraction_correction(q_raw, row["입사각(deg)"], critical_angle)
                    
                    results.append({
                        "파일명": row["파일명"],
                        "입사각": row["입사각(deg)"],
                        "q_raw (측정)": q_raw,
                        "q_corrected (보정)": q_exp,
                        "Strain(%)": (q_bulk - q_exp) / q_exp * 100
                    })
                    
                    # 2D 이미지 상세 출력 영역
                    with st.expander(f"📊 {row['파일명']} 상세 분석"):
                        col1, col2 = st.columns(2)
                        with col1:
                            # [요청 반영] 이미지를 수직으로 뒤집어서 출력
                            fig2d, ax2d = plt.subplots()
                            im = ax2d.imshow(np.log1p(np.clip(img, 0, None)), cmap='jet', origin='lower') # origin='lower' 적용
                            ax2d.set_title("2D GIWAXS (Flipped, Jet)")
                            plt.colorbar(im, ax=ax2d)
                            st.pyplot(fig2d)
                        with col2:
                            # 1D 피팅 결과 출력
                            fig1d, ax1d = plt.subplots()
                            ax1d.plot(q_c, I_c, 'bo', markersize=3, label='Data')
                            ax1d.plot(q_c, out.best_fit, 'r-', label='Gaussian Fit')
                            ax1d.axvline(q_bulk, color='g', linestyle='--', label=f'Bulk: {q_bulk:.3f}')
                            ax1d.axvline(q_exp, color='r', linestyle='--', label=f'Corrected: {q_exp:.3f}')
                            ax1d.set_title("Peak Fitting Result")
                            ax1d.legend()
                            ax1d.grid(True, alpha=0.3)
                            st.pyplot(fig1d)
                            
                except Exception as e:
                    st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        # 세션에 결과 저장
        st.session_state.strain_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()

    # --- 최종 결과 트렌드 그래프 영역 ---
    if st.session_state.strain_results is not None:
        res_df = st.session_state.strain_results
        st.divider()
        st.subheader("📈 입사각별 Strain 트렌드 (보정 완료)")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.dataframe(res_df.style.format({
                "q_raw (측정)": "{:.4f}",
                "q_corrected (보정)": "{:.4f}",
                "Strain(%)": "{:.3f}"
            }))
            
            # 다운로드 버튼 (중복 Key 방지)
            st.download_button("💾 Strain 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "strain_summary.csv", key="save_summary_btn")
            st.download_button("📂 Origin용 TXT(.zip) 저장", st.session_state.zip_data, "origin_profiles.zip", key="save_zip_btn")
            
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(res_df["입사각"], res_df["Strain(%)"], 'ro-', label='Strain (%)')
            ax.set_xlabel("Incidence Angle (deg)")
            ax.set_ylabel("Strain (%)")
            ax.set_title("Strain Trend with Refraction Correction")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)