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

# [추가] 지평선 및 빔스탑 기반 2단계 자동 정렬 알고리즘
def auto_calibrate_center(img_data):
    # Step 1: DBy 찾기 (수직 그라디언트 최대 지점 = Horizon)
    # axis=1로 합산하여 각 행의 전체 강도 변화를 관찰
    v_profile = np.sum(img_data, axis=1)
    v_gradient = np.abs(np.diff(v_profile))
    dby_auto = np.argmax(v_gradient) # 지평선(Horizon) 행 번호
    
    # Step 2: DBx 찾기 (검출된 지평선상에서 가장 어두운 지점 = Beamstop)
    h_profile = img_data[dby_auto, :]
    dbx_auto = np.argmin(h_profile) # 빔스탑 중심 열 번호
    
    return float(dbx_auto), float(dby_auto)

st.set_page_config(page_title="UNIST 6D GIWAXS Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (2단계 자동 정렬 적용)")

# --- 세션 상태 초기화 (자동 정렬 값 유지) ---
if 'dbx' not in st.session_state: st.session_state.dbx = 1440.36
if 'dby' not in st.session_state: st.session_state.dby = 1053.49

# --- 사이드바: 실험 셋업 ---
st.sidebar.header("1. 실험 셋업 (6D UNIST-PAL)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=11.564, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=100.0, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=78.13)

st.sidebar.divider()
st.sidebar.subheader("🎯 빔 센터(Beam Center) 정렬")

# [기능 추가] 자동 정렬 버튼
if st.sidebar.button("🪄 빔 센터 자동 찾기 (지평선 기반)"):
    # 현재 업로드된 파일이 있는지 확인
    if 'current_img' in st.session_state:
        dbx_a, dby_a = auto_calibrate_center(st.session_state.current_img)
        st.session_state.dbx = dbx_a
        st.session_state.dby = dby_a
        st.sidebar.success(f"자동 정렬 완료! (X:{dbx_a:.2f}, Y:{dby_a:.2f})")
    else:
        st.sidebar.warning("⚠️ 먼저 TIF 파일을 업로드해주세요.")

# 수동 조정 입력창 (세션 상태와 연동)
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=st.session_state.dbx, step=0.01)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=st.session_state.dby, step=0.01)

wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

st.sidebar.divider()
st.sidebar.header("2. 분석 파라미터")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.5420, format="%.4f")
q_min = st.sidebar.number_input("Fit 영역 시작 q", value=1.40)
q_max = st.sidebar.number_input("Fit 영역 끝 q", value=1.80)

st.sidebar.subheader("🎯 1D 적분 각도 설정")
azi_min = st.sidebar.number_input("최소 Azimuth (°)", value=-180)
azi_max = st.sidebar.number_input("최대 Azimuth (°)", value=0)

# --- 파일 업로드 ---
uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    
    # 자동 정렬을 위해 첫 번째 이미지 미리 로드
    if 'current_img' not in st.session_state or st.session_state.first_file != file_list[0]:
        st.session_state.current_img = fabio.open(uploaded_files[0]).data
        st.session_state.first_file = file_list[0]

    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'zip_data' not in st.session_state: st.session_state.zip_data = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor_auto")

    if st.button("🚀 전수 분석 시작", type="primary"):
        temp_dir = "temp_giwaxs"
        os.makedirs(temp_dir, exist_ok=True)
        paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
        for uf in uploaded_files:
            with open(paths[uf.name], "wb") as f: f.write(uf.getbuffer())

        # pyFAI 엔진 설정
        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results, zip_buffer = [], io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(paths[row["파일명"]]).data
                    q, I = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_min, azi_max))
                    
                    # Origin 저장
                    txt = pd.DataFrame({"q": q, "I": I}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_1D.txt", txt)

                    # 피팅
                    mask = (q >= q_min) & (q <= q_max)
                    qc, Ic = q[mask], I[mask]
                    if len(qc) < 5: continue
                        
                    model = GaussianModel() + LinearModel()
                    params = model.make_params(amplitude=Ic.max()-Ic.min(), center=(q_min+q_max)/2, sigma=0.05, slope=0, intercept=Ic.min())
                    params['center'].set(min=q_min, max=q_max)
                    out = model.fit(Ic, params, x=qc)
                    q_exp = out.params['center'].value
                    strain = (q_bulk - q_exp) / q_exp * 100
                    
                    results.append({"파일명": row["파일명"], "입사각": row["입사각(deg)"], "q_measured": q_exp, "Strain(%)": strain})
                    
                    with st.expander(f"📊 {row['파일명']} 상세 분석"):
                        c1, c2 = st.columns(2)
                        with c1:
                            # [논문 표기법 적용] q_xy, q_z 축 변환 및 시각화
                            h, w = img_data.shape
                            dq = (2*np.pi/(wavelength*1e10)) * (px_m/dist_m)
                            ext = [-dbx*dq, (w-dbx)*dq, -dby*dq, (h-dby)*dq]
                            fig2d, ax2d = plt.subplots()
                            ax2d.imshow(np.log1p(np.clip(np.flipud(img_data), 0, None)), cmap='jet', extent=ext)
                            ax2d.set_title("2D GIWAXS (Automated Alignment)"); ax2d.set_xlabel(r"$q_{xy} (\AA^{-1})$"); ax2d.set_ylabel(r"$q_z (\AA^{-1})$")
                            st.pyplot(fig2d); plt.close(fig2d)
                        with c2:
                            fig1d, ax1d = plt.subplots()
                            ax1d.plot(qc, Ic, 'bo', markersize=3, label='Data')
                            ax1d.plot(qc, out.best_fit, 'r-', label='Fit')
                            ax1d.set_title(f"Strain: {strain:.3f}%"); ax1d.legend(); st.pyplot(fig1d); plt.close(fig1d)
                            
                except Exception as e: st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()

    if st.session_state.analysis_results is not None:
        st.divider(); st.subheader("📈 입사각별 Strain 트렌드")
        res_df = st.session_state.analysis_results
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.dataframe(res_df.style.format({"q_measured": "{:.4f}", "Strain(%)": "{:.3f}"}))
            st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "strain_results.csv", key="dl_csv_auto")
        with c2:
            fig_tr, ax_tr = plt.subplots()
            ax_tr.plot(res_df["입사각"], res_df["Strain(%)"], 'ro-'); ax_tr.set_xlabel("Incidence Angle (deg)"); ax_tr.set_ylabel("Strain (%)"); st.pyplot(fig_tr)