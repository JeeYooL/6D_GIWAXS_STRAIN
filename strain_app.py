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

# --- 파일명에서 입사각 추출 (0.117d 패턴) ---
def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

# --- 페이지 설정 ---
st.set_page_config(page_title="UNIST 6D Strain Analyzer", layout="wide")
st.title("🔬 6D 빔라인 Strain 분석 (Igor 매뉴얼 셋업 적용)")

# --- 사이드바: 빔 정보 설정 ---
st.sidebar.header("1. 빔라인 세팅 (6D UNIST-PAL)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=12.4, help="보통 5~20 keV 사이를 사용합니다.")
dist_mm = st.sidebar.number_input("Sample-to-Detector Distance (mm)", value=200.0)
pixel_size_um = st.sidebar.number_input("Pixel Size (um)", value=172.0, help="검출기 픽셀 크기 (예: Pilatus=172, Rayonix=73.2)")
center_x = st.sidebar.number_input("Beam Center X (pixel)", value=1024)
center_y = st.sidebar.number_input("Beam Center Y (pixel)", value=512)

# 물리량 계산
wavelength = (12.3984 / energy_kev) * 1e-10 # 파장 (m)
dist_m = dist_mm / 1000.0
pixel_m = pixel_size_um * 1e-6
poni1 = center_y * pixel_m # pyFAI 표준 좌표계
poni2 = center_x * pixel_m

st.sidebar.divider()
st.sidebar.header("🎯 Analysis Parameters")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, format="%.4f")
q_min = st.sidebar.number_input("Fitting Start q", value=1.4)
q_max = st.sidebar.number_input("Fitting End q", value=1.7)

# --- 메인 로직 ---
uploaded_files = st.sidebar.file_uploader("📂 TIF 데이터 임시 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    st.subheader("📋 샘플 리스트 (자동 입사각 인식)")
    
    # 파일 이름 정보 추출
    file_list = sorted([f.name for f in uploaded_files])
    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True)

    if st.button("🚀 Strain 분석 실행"):
        # Streamlit 클라우드 환경에서는 fabio로 파일을 직접 열기 위해 임시 폴더에 바이트 정보를 저장합니다.
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        saved_paths = {}
        for uf in uploaded_files:
            tmp_path = os.path.join(temp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths[uf.name] = tmp_path

        # pyFAI 엔진 설정
            geo = AzimuthalIntegrator(dist=dist_m, poni1=poni1, poni2=poni2, 
                                      wavelength=wavelength, pixel1=pixel_m, pixel2=pixel_m)
            
            results = []
            progress = st.progress(0)
            
            for i, row in edited_df.iterrows():
                try:
                    f_path = saved_paths[row["파일명"]]
                    img_data = fabio.open(f_path).data
                    
                    # 1D Integration (Line-cut 대체) 
                    q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1")
                    
                    # Peak Fitting
                    mask = (q >= q_min) & (q <= q_max)
                    q_cut, I_cut = q[mask], I[mask]
                    
                    if len(q_cut) == 0:
                        raise ValueError(f"설정하신 Fitting 범위(q={q_min}~{q_max}) 내에 데이터가 하나도 없습니다! (현재 계산된 전체 이미지의 최대 q값은 {q.max():.4f} 입니다). SDD나 Pixel Size 등 Setup 수치를 다시 확인해 주세요.")
                    
                    model = GaussianModel()
                    params = model.guess(I_cut, x=q_cut)
                    out = model.fit(I_cut, params, x=q_cut)
                    q_exp = out.params['center'].value
                    
                    # Strain 공식 적용
                    strain = (q_bulk - q_exp) / q_exp
                    
                    results.append({
                        "파일명": row["파일명"],
                        "입사각": row["입사각(deg)"],
                        "q_measured": q_exp,
                        "Strain(%)": strain * 100
                    })
                    
                    # 상세 결과 시각화 (Igor jet 스타일 적용) 
                    with st.expander(f"📊 {row['파일명']} 분석 상세"):
                        c1, c2 = st.columns(2)
                        with c1:
                            fig2d, ax2d = plt.subplots()
                            # 매뉴얼의 Rainbow/Reverse 컬러맵 적용 [cite: 695]
                            im = ax2d.imshow(np.log1p(np.clip(img_data, 0, None)), cmap='jet', origin='lower')
                            ax2d.set_title("2D GIWAXS (Jet Colormap)")
                            plt.colorbar(im, ax=ax2d)
                            st.pyplot(fig2d)
                        with c2:
                            fig1d, ax1d = plt.subplots()
                            ax1d.plot(q_cut, I_cut, 'bo', label='Data')
                            ax1d.plot(q_cut, out.best_fit, 'r-', label='Fit')
                            ax1d.axvline(q_exp, color='g', linestyle='--', label=f'q={q_exp:.4f}')
                            ax1d.set_title("Peak Fitting Result")
                            ax1d.set_xlabel("q (Å⁻¹)")
                            ax1d.legend()
                            st.pyplot(fig1d)
                            
                except Exception as e:
                    st.error(f"Error in {row['파일명']}: {e}")
                progress.progress((i + 1) / len(edited_df))

            # 최종 트렌드 보고
            if results:
                res_df = pd.DataFrame(results)
                st.divider()
                st.subheader("📈 입사각별 Strain 트렌드")
                fig_res, ax_res = plt.subplots(figsize=(8, 5))
                ax_res.plot(res_df["입사각"], res_df["Strain(%)"], 'bo-', linewidth=2)
                ax_res.set_xlabel("Incidence Angle (deg)")
                ax_res.set_ylabel("Strain (%)")
                ax_res.grid(True, alpha=0.3)
                st.pyplot(fig_res)
                
                st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "strain_results.csv", key="download_csv_results")