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
import re # 정규표현식 라이브러리 추가

# --- 파일명에서 입사각 추출하는 함수 ---
def extract_incidence_angle(filename):
    # 패턴 설명: 숫자 + . + 숫자 + d (예: 0.117d)를 찾아서 숫자만 추출
    match = re.search(r"(\d+\.\d+)d", filename)
    if match:
        return float(match.group(1))
    return 0.10 # 패턴이 없을 경우 기본값

# --- GUI 폴더 선택 함수 ---
def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(master=root)
    root.destroy()
    return path

# --- 페이지 설정 ---
st.set_page_config(page_title="6D Beamline Strain Analyzer", layout="wide")
st.title("🔬 6D 빔라인 데이터 기반 Strain 분석 (테스트 모드)")

# --- 사이드바: 임의의 빔 정보 설정 ---
st.sidebar.header("1. 가상 빔라인 세팅 (임의 값)")
# 6D 빔라인의 일반적인 범위를 기준으로 설정
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
st.sidebar.header("2. 분석 대상 설정")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, help="변형이 없는 기준 피크 위치")
q_min = st.sidebar.number_input("Fitting 범위 시작 (q)", value=1.4)
q_max = st.sidebar.number_input("Fitting 범위 끝 (q)", value=1.7)

# --- 메인 로직 ---
if st.sidebar.button("📂 데이터 폴더 선택"):
    st.session_state.target_folder = select_folder()

if 'target_folder' in st.session_state:
    st.sidebar.write(f"현재 폴더: `{os.path.basename(st.session_state.target_folder)}`")
    files = [f for f in os.listdir(st.session_state.target_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    if files:
        st.subheader("📋 샘플 정보 (입사각 자동 추출 완료)")
        
        # 입사각을 자동으로 추출하여 데이터프레임 생성
        file_list = sorted(files)
        angles = [extract_incidence_angle(f) for f in file_list]
        
        input_df = pd.DataFrame({
            "파일명": file_list,
            "입사각(deg)": angles
        })
        
        # 사용자가 수동으로 검토 및 수정 가능
        edited_df = st.data_editor(input_df, use_container_width=True)

        if st.button("🚀 전수 분석 시작"):
            # pyFAI 기하학 엔진 설정
            geo = AzimuthalIntegrator(dist=dist_m, poni1=poni1, poni2=poni2, 
                                      wavelength=wavelength, pixel1=pixel_m, pixel2=pixel_m)
            
            results = []
            progress = st.progress(0)
            
            for i, row in edited_df.iterrows():
                try:
                    f_path = os.path.join(st.session_state.target_folder, row["파일명"])
                    img_data = fabio.open(f_path).data
                    
                    # 1D Integration
                    q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1")
                    
                    # Peak Fitting (Gaussian)
                    mask = (q >= q_min) & (q <= q_max)
                    q_cut, I_cut = q[mask], I[mask]
                    
                    model = GaussianModel()
                    params = model.guess(I_cut, x=q_cut)
                    out = model.fit(I_cut, params, x=q_cut)
                    q_exp = out.params['center'].value
                    
                    strain = (q_bulk - q_exp) / q_exp
                    
                    results.append({
                        "파일명": row["파일명"],
                        "입사각": row["입사각(deg)"],
                        "q_measured": q_exp,
                        "Strain(%)": strain * 100
                    })
                    
                    # --- 1D 및 2D 그래프 출력 ---
                    with st.expander(f"📊 {row['파일명']} 상세 결과 (입사각: {row['입사각(deg)']}°)"):
                        c_2d, c_1d = st.columns(2)
                        
                        # 2D GIWAXS 그래프
                        with c_2d:
                            fig2d, ax2d = plt.subplots(figsize=(5, 4))
                            # 2D 패턴은 log 단위로 시각화 (0 미만 값 방지를 위해 clip 처리)
                            im = ax2d.imshow(np.log1p(np.clip(img_data, 0, None)), cmap='jet', origin='lower')
                            ax2d.set_title("2D GIWAXS Pattern")
                            ax2d.axis('off')
                            plt.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
                            st.pyplot(fig2d)
                            plt.close(fig2d)
                            
                        # 1D XRD 프로파일 및 Fitting 결과
                        with c_1d:
                            fig1d, ax1d = plt.subplots(figsize=(5, 4))
                            ax1d.plot(q, I, 'k-', alpha=0.3, label='Whole Profile')
                            ax1d.plot(q_cut, I_cut, 'bo', markersize=3, label='Fit Range Data')
                            ax1d.plot(q_cut, out.best_fit, 'r-', linewidth=2, label='Gaussian Fit')
                            ax1d.axvline(x=q_exp, color='g', linestyle='--', label=f'Peak: {q_exp:.4f}')
                            
                            ax1d.set_xlim(max(0, q_min - 0.3), q_max + 0.3)
                            
                            # Y축 설정 (너무 낮은 배경을 제외하고 Peak 위주로 확대)
                            max_I = np.max(I_cut) if len(I_cut) > 0 else 1
                            ax1d.set_ylim(0, max_I * 1.5)
                            
                            ax1d.set_xlabel("q (Å⁻¹)")
                            ax1d.set_ylabel("Intensity (a.u.)")
                            ax1d.set_title("1D Profile & Peak Fitting")
                            ax1d.legend(fontsize=8)
                            st.pyplot(fig1d)
                            plt.close(fig1d)
                            
                except Exception as e:
                    st.error(f"에러 ({row['파일명']}): {e}")
                progress.progress((i + 1) / len(edited_df))

            # 결과 시각화
            if results:
                res_df = pd.DataFrame(results)
                st.divider()
                st.subheader("📊 분석 결과 요약")
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(res_df.style.format({"q_measured": "{:.4f}", "Strain(%)": "{:.3f}"}))
                with c2:
                    fig, ax = plt.subplots()
                    ax.plot(res_df["입사각"], res_df["Strain(%)"], 'bo-', label='Strain Trend')
                    ax.set_xlabel("Incidence Angle (deg)")
                    ax.set_ylabel("Strain (%)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                csv = res_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 결과 저장(CSV)", csv, "strain_test_result.csv")
    else:
        st.warning("폴더 안에 TIF 파일이 없습니다.")
else:
    st.info("사이드바에서 데이터 폴더를 선택해 주세요.")