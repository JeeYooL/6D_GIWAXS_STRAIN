import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from lmfit.models import GaussianModel
import tkinter as tk
from tkinter import filedialog
import re

# --- 파일명에서 입사각 추출 (0.117d 패턴) ---
def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(master=root)
    root.destroy()
    return path

# --- 페이지 설정 ---
st.set_config(page_title="UNIST 6D Strain Analyzer", layout="wide")
st.title("🔬 6D 빔라인 Strain 분석 (Igor 매뉴얼 셋업 적용)")

# --- 사이드바: Igor 매뉴얼 기반 설정 [cite: 498, 538] ---
st.sidebar.header("⚙️ Experimental Setup (Igor/MATLAB)")

# 매뉴얼 3p의 용어와 동일하게 구성 
energy_kev = st.sidebar.number_input("X-ray Energy (keV)", value=11.564, format="%.3f")
pixel_size_mm = st.sidebar.number_input("Pixel size (mm)", value=0.078125, format="%.6f")

col_c1, col_c2 = st.sidebar.columns(2)
with col_c1:
    center_x = st.sidebar.number_input("Center X (MATLAB)", value=1441.36)
with col_c2:
    center_y = st.sidebar.number_input("Center Y (MATLAB)", value=1054.49)

# Igor 매뉴얼 핵심: DBx = CenterX - 1 
dbx = center_x - 1
dby = center_y - 1

sdd_mm = st.sidebar.number_input("SDD (mm)", value=3057.064, format="%.3f")

# 물리량 변환 (pyFAI용)
wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = sdd_mm / 1000.0
pixel_m = pixel_size_mm / 1000.0
# pyFAI는 (y, x) 순서이며 0-based 이므로 DB값을 그대로 미터로 환산 
poni1 = dby * pixel_m 
poni2 = dbx * pixel_m

st.sidebar.divider()
st.sidebar.header("🎯 Analysis Parameters")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, format="%.4f")
q_min = st.sidebar.number_input("Fitting Start q", value=1.4)
q_max = st.sidebar.number_input("Fitting End q", value=1.7)

# --- 메인 로직 ---
if st.sidebar.button("📂 데이터 폴더 선택"):
    st.session_state.target_folder = select_folder()

if 'target_folder' in st.session_state:
    files = [f for f in os.listdir(st.session_state.target_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    if files:
        st.subheader("📋 샘플 리스트 (자동 입사각 인식)")
        file_list = sorted(files)
        angles = [extract_incidence_angle(f) for f in file_list]
        input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
        edited_df = st.data_editor(input_df, use_container_width=True)

        if st.button("🚀 Strain 분석 실행"):
            # pyFAI 엔진 설정
            geo = AzimuthalIntegrator(dist=dist_m, poni1=poni1, poni2=poni2, 
                                      wavelength=wavelength, pixel1=pixel_m, pixel2=pixel_m)
            
            results = []
            progress = st.progress(0)
            
            for i, row in edited_df.iterrows():
                try:
                    f_path = os.path.join(st.session_state.target_folder, row["파일명"])
                    img_data = fabio.open(f_path).data
                    
                    # 1D Integration (Line-cut 대체) 
                    q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1")
                    
                    # Peak Fitting
                    mask = (q >= q_min) & (q <= q_max)
                    q_cut, I_cut = q[mask], I[mask]
                    
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
                
                st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "strain_results.csv")