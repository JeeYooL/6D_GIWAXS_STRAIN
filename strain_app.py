import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from lmfit.models import GaussianModel, LinearModel # 배경 제거를 위해 Linear 모델 추가
import re
import zipfile
import io

def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

st.set_page_config(page_title="UNIST 6D GIWAXS Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (그래프 정상화 버전)")

# --- 사이드바 설정 ---
st.sidebar.header("1. 실험 셋업")
energy_kev = st.sidebar.number_input("Energy (keV)", value=11.564, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=3057.064, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=78.125)
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=1440.36)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=1053.49)

wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

st.sidebar.divider()
st.sidebar.header("2. 분석 파라미터")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.542, format="%.4f")
q_min = st.sidebar.number_input("Fit 영역 시작 q", value=1.4)
q_max = st.sidebar.number_input("Fit 영역 끝 q", value=1.8)

st.sidebar.subheader("🎯 1D 적분 각도(Azimuth) 설정")
st.sidebar.info("💡 1D 그래프에 피크가 안 보인다면 이 각도를 90도씩 돌려보세요.")
azi_min = st.sidebar.number_input("최소 Azimuth (°)", value=-180)
azi_max = st.sidebar.number_input("최대 Azimuth (°)", value=0) # 아래반원 강조를 위해 변경 테스트 제안

uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    if 'strain_results' not in st.session_state: st.session_state.strain_results = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor")

    if st.button("🚀 전수 분석 시작", type="primary"):
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(paths[uf.name], "wb") as f: f.write(uf.getbuffer())

        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results = []
        pbar = st.progress(0)
        for i, row in edited_df.iterrows():
            try:
                img_data = fabio.open(paths[row["파일명"]]).data
                
                # [개선] 적분 각도 범위를 적용하여 배경 노이즈 최소화
                q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_min, azi_max))
                
                # 피팅 범위 데이터 슬라이싱
                mask = (q >= q_min) & (q <= q_max)
                q_c, I_c = q[mask], I[mask]
                
                if len(q_c) < 10: raise ValueError("데이터 부족")

                # [개선] Gaussian + Linear 배경 모델 적용
                # 우상향하는 배경을 무시하고 피크만 찾기 위함
                model = GaussianModel() + LinearModel()
                params = model.make_params(amplitude=I_c.max()-I_c.min(), center=(q_min+q_max)/2, sigma=0.05, slope=0, intercept=I_c.min())
                # center 값이 범위를 벗어나지 않도록 제한
                params['center'].set(min=q_min, max=q_max)
                
                out = model.fit(I_c, params, x=q_c)
                q_exp = out.params['center'].value
                
                strain = (q_bulk - q_exp) / q_exp * 100
                results.append({"파일명": row["파일명"], "입사각": row["입사각(deg)"], "q_measured": q_exp, "Strain(%)": strain})
                
                with st.expander(f"📊 {row['파일명']} 상세 분석"):
                    c1, c2 = st.columns(2)
                    with c1:
                        # [요청 반영] 2D 이미지를 상하 반전하여 시각화
                        fig2d, ax2d = plt.subplots()
                        ax2d.imshow(np.log1p(np.clip(np.flipud(img_data), 0, None)), cmap='jet')
                        ax2d.set_title("2D GIWAXS (Vertical Flipped)")
                        st.pyplot(fig2d)
                        plt.close(fig2d)
                    with c2:
                        # [개선] 피팅 결과 그래프
                        fig1d, ax1d = plt.subplots()
                        ax1d.plot(q_c, I_c, 'bo', markersize=2, label='Data')
                        ax1d.plot(q_c, out.best_fit, 'r-', label='Fit (Gauss+Linear)')
                        ax1d.axvline(q_exp, color='r', linestyle='--', label=f'Peak: {q_exp:.4f}')
                        ax1d.axvline(q_bulk, color='g', linestyle='--', label=f'Bulk: {q_bulk:.4f}')
                        ax1d.set_title(f"Strain: {strain:.3f}%")
                        ax1d.legend()
                        st.pyplot(fig1d)
                        plt.close(fig1d)
            except Exception as e:
                st.error(f"❌ {row['파일명']} 실패: {e}")
            pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.strain_results = pd.DataFrame(results)

    if st.session_state.strain_results is not None:
        st.divider()
        st.subheader("📊 최종 Strain 분석 결과")
        st.dataframe(st.session_state.strain_results)
        # 트렌드 그래프 추가
        fig_tr, ax_tr = plt.subplots()
        ax_tr.plot(st.session_state.strain_results["입사각"], st.session_state.strain_results["Strain(%)"], 'ro-')
        ax_tr.set_xlabel("Incidence Angle (deg)")
        ax_tr.set_ylabel("Strain (%)")
        st.pyplot(fig_tr)