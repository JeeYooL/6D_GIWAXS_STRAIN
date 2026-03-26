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

# [강화된 2단계 자동 정렬 알고리즘] Difference of Gaussians (DoG) 기반 빔스탑 탐지
def auto_calibrate_center(img_data):
    try:
        import scipy.ndimage as ndi
        
        # 1. 핫픽셀 및 우측 하단 에러 픽셀(빨간 네모) 등 극단적인 노이즈 자르기
        p99 = np.percentile(img_data, 99.5)
        clipped = np.clip(img_data, 0, p99)
        
        # 2. DoG 필터 (Mexican Hat) 적용
        # 빔스탑(어두운 원)과 할로(밝은 회절 링)의 극적인 대비를 이용
        # sigma=50은 주변 밝기를 넓게 평균내고, sigma=10은 빔스탑의 날카로운 윤곽을 잡음
        blur_large = ndi.gaussian_filter(clipped, sigma=50)
        blur_small = ndi.gaussian_filter(clipped, sigma=10)
        
        dog = blur_large - blur_small
        
        # 3. 가장 완벽한 둥근 얼룩(Blob)의 중심 좌표 찾기
        # 빔스탑에 가려져 중심이 어두우면 dog가 크게 양수(+)
        # 만약 마스킹 없이 빔이 직접 때려 포화된 경우 dog가 크게 음수(-)
        if abs(np.min(dog)) > abs(np.max(dog)):
            dby_auto, dbx_auto = np.unravel_index(np.argmin(dog), dog.shape)
        else:
            dby_auto, dbx_auto = np.unravel_index(np.argmax(dog), dog.shape)
            
        return float(dbx_auto), float(dby_auto)
    except ImportError:
        # scipy가 없을 경우를 대비한 단순 Fallback
        dby_auto, dbx_auto = np.unravel_index(np.argmin(img_data), img_data.shape)
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

st.sidebar.subheader("🎯 1D 적분 각도 (Out / In-plane)")
c_out1, c_out2 = st.sidebar.columns(2)
azi_out_min = c_out1.number_input("Out 최소(°)", value=-110)
azi_out_max = c_out2.number_input("Out 최대(°)", value=-70)

c_in1, c_in2 = st.sidebar.columns(2)
azi_in_min = c_in1.number_input("In 최소(°)", value=-20)
azi_in_max = c_in2.number_input("In 최대(°)", value=0)

# --- 파일 업로드 방식 결정 ---
st.sidebar.subheader("📂 데이터 업로드 방식")
use_sample_data = st.sidebar.checkbox("✅ 서버의 샘플 데이터로 테스트하기", help="미리 올려둔 'sample_data' 폴더의 파일 사용")

uploaded_files = []
if use_sample_data:
    sample_dir = "sample_data"
    if os.path.exists(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.lower().endswith(('.tif', '.tiff')):
                fpath = os.path.join(sample_dir, fname)
                with open(fpath, "rb") as f:
                    file_obj = io.BytesIO(f.read())
                    file_obj.name = fname
                    uploaded_files.append(file_obj)
    if not uploaded_files:
        st.sidebar.warning(f"❌ `{sample_dir}` 폴더가 비어 있거나 TIF 파일이 없습니다. 파일을 넣어주세요!")
else:
    uploaded_files = st.sidebar.file_uploader("📂 TIF 파일 업로드", type=['tif', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    file_list = sorted([f.name for f in uploaded_files])
    
    # [수정] fabio.open() 에러 방지를 위해 우선 모든 파일을 물리적 저장소에 기록
    temp_dir = "temp_giwaxs"
    os.makedirs(temp_dir, exist_ok=True)
    paths = {uf.name: os.path.join(temp_dir, uf.name) for uf in uploaded_files}
    for uf in uploaded_files:
        with open(paths[uf.name], "wb") as f: f.write(uf.getbuffer())
    
    # 물리적으로 저장된 첫 번째 이미지를 읽어서 캐싱
    if 'current_img' not in st.session_state or st.session_state.first_file != file_list[0]:
        st.session_state.current_img = fabio.open(paths[file_list[0]]).data
        st.session_state.first_file = file_list[0]

    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'zip_data' not in st.session_state: st.session_state.zip_data = None

    angles = [extract_incidence_angle(f) for f in file_list]
    input_df = pd.DataFrame({"파일명": file_list, "입사각(deg)": angles})
    edited_df = st.data_editor(input_df, use_container_width=True, key="data_editor_auto")

    # --- [검증용] 실시간 2D 프리뷰 (센터 표시) ---
    st.subheader("🖼️ 현재 빔 센터 정렬 확인 (Preview) - 가장 위의 이미지 기준")
    img_preview = st.session_state.current_img
    flipped_img = np.flipud(img_preview)
    
    fig_pre, ax_pre = plt.subplots(figsize=(6, 4))
    im_pre = ax_pre.imshow(np.log1p(np.clip(flipped_img, 0, None)), cmap='jet')
    
    # 십자선 표시 (Flip 고려)
    h_pre, w_pre = img_preview.shape
    if dbx is not None and dby is not None:
        ax_pre.axvline(x=dbx, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_pre.axhline(y=h_pre-dby, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_pre.scatter(dbx, h_pre-dby, color='red', s=100, marker='+', label='Current Center')
    
    ax_pre.set_title(f"Center Preview (X={dbx:.1f}, Y={dby:.1f})")
    plt.colorbar(im_pre, ax=ax_pre)
    st.pyplot(fig_pre)
    plt.close(fig_pre)
    st.info("💡 위 이미지의 **빨간 십자선(+)**이 파란색 빔스탑의 정중앙에 위치하는지 확인하시고 아래 버튼을 누르세요.")

    if st.button("🚀 위 설정으로 전수 분석 시작", type="primary"):
        # pyFAI 엔진 설정 (파일은 이미 로딩 과정에서 temp_dir에 저장 완료됨)
        geo = AzimuthalIntegrator(dist=dist_m, poni1=dby*px_m, poni2=dbx*px_m, 
                                  wavelength=wavelength, pixel1=px_m, pixel2=px_m)
        
        results, zip_buffer = [], io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(paths[row["파일명"]]).data
                    q_out, I_out = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_out_min, azi_out_max))
                    q_in, I_in = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_in_min, azi_in_max))
                    
                    # Origin 저장 (두 방향 통합)
                    txt = pd.DataFrame({"q_out": q_out, "I_out": I_out, "q_in": q_in, "I_in": I_in}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_1D.txt", txt)

                    # --- [강화된 다중 피크(Multi-Peak) 피팅 함수] ---
                    def fit_peak(q, I):
                        from scipy.signal import find_peaks
                        mask = (q >= q_min) & (q <= q_max)
                        qc, Ic = q[mask], I[mask]
                        if len(qc) < 5: return None, None, None, None
                        
                        # 1. 주어진 q 범위 내에서 눈에 띄는(prominence) 모든 피크 위치를 탐색
                        peaks, _ = find_peaks(Ic, prominence=0.03 * (Ic.max() - Ic.min()), distance=5)
                        
                        # 피크가 하나도 검색되지 않으면 단순히 가장 큰 값을 피크 배열로 간주
                        if len(peaks) == 0:
                            peaks = [np.argmax(Ic)]
                            
                        # 2. 다중 피크 피팅을 위한 동적 모델(Composite Model) 생성
                        # 기본 배경(Background)을 위한 Linear 모델 추가
                        model = LinearModel(prefix='bkg_')
                        params = model.make_params(slope=0, intercept=Ic.min())
                        
                        # 찾은 각각의 피크마다 별도의 Gaussian 모델을 생성하여 전체 모델에 덧셈
                        for i, p_idx in enumerate(peaks):
                            center_guess = qc[p_idx]
                            amp_guess = (Ic[p_idx] - Ic.min()) * 0.05
                            
                            p_model = GaussianModel(prefix=f'p{i}_')
                            p_params = p_model.make_params(amplitude=amp_guess, center=center_guess, sigma=0.02)
                            
                            # 해당 피크 중심이 초기 추측값 근처(±0.05)를 벗어나 엉뚱하게 발산하는 것을 방지
                            p_params[f'p{i}_center'].set(min=center_guess - 0.05, max=center_guess + 0.05)
                            
                            model += p_model
                            params.update(p_params)
                            
                        # 3. 모델 피팅 수행
                        out = model.fit(Ic, params, x=qc)
                        
                        # 4. 피팅된 여러 개의 피크들 중에서, 우리가 관심 있는 '기준 q_bulk'와 가장 가까운 메인 피크를 선택
                        centers = [out.params[f'p{i}_center'].value for i in range(len(peaks))]
                        best_center = min(centers, key=lambda c: abs(c - q_bulk))
                        
                        # 5. 메인 피크 기준으로 변형률(Strain) 계산
                        strain = (q_bulk - best_center) / best_center * 100
                        return qc, Ic, out, strain

                    # 두 방향 각각 피팅
                    qc_out, Ic_out, fit_out, strain_out = fit_peak(q_out, I_out)
                    qc_in, Ic_in, fit_in, strain_in = fit_peak(q_in, I_in)
                    
                    if fit_out is None or fit_in is None:
                        st.warning(f"{row['파일명']}: 피팅할 데이터가 부족합니다.")
                        continue
                        
                    results.append({"파일명": row["파일명"], "입사각": row["입사각(deg)"], 
                                    "Strain_Out(%)": strain_out, "Strain_In(%)": strain_in})
                    
                    with st.expander(f"📊 {row['파일명']} 상세 분석"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            # 2D GIWAXS 패턴
                            h, w = img_data.shape
                            dq = (2*np.pi/(wavelength*1e10)) * (px_m/dist_m)
                            ext = [-dbx*dq, (w-dbx)*dq, -dby*dq, (h-dby)*dq]
                            fig2d, ax2d = plt.subplots()
                            ax2d.imshow(np.log1p(np.clip(np.flipud(img_data), 0, None)), cmap='jet', extent=ext)
                            ax2d.set_title("2D GIWAXS"); ax2d.set_xlabel(r"$q_{xy} (\AA^{-1})$"); ax2d.set_ylabel(r"$q_z (\AA^{-1})$")
                            st.pyplot(fig2d); plt.close(fig2d)
                        with c2:
                            # 1D 피팅 결과 (Out-of-plane)
                            fig_out, ax_out = plt.subplots()
                            ax_out.plot(qc_out, Ic_out, 'bo', markersize=3, label='Data')
                            ax_out.plot(qc_out, fit_out.best_fit, 'r-', label='Fit')
                            ax_out.set_title(f"Out-of-plane Strain: {strain_out:.3f}%")
                            ax_out.set_xlabel(r"$q_z (\AA^{-1})$"); ax_out.legend()
                            st.pyplot(fig_out); plt.close(fig_out)
                        with c3:
                            # 1D 피팅 결과 (In-plane)
                            fig_in, ax_in = plt.subplots()
                            ax_in.plot(qc_in, Ic_in, 'bo', markersize=3, label='Data')
                            ax_in.plot(qc_in, fit_in.best_fit, 'r-', label='Fit')
                            ax_in.set_title(f"In-plane Strain: {strain_in:.3f}%")
                            ax_in.set_xlabel(r"$q_{xy} (\AA^{-1})$"); ax_in.legend()
                            st.pyplot(fig_in); plt.close(fig_in)
                            
                except Exception as e: st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()

    if st.session_state.analysis_results is not None:
        st.divider(); st.subheader("📈 입사각별 Strain 트렌드")
        res_df = st.session_state.analysis_results
        if res_df.empty:
            st.warning("⚠️ 성공적으로 분석된 데이터가 없습니다. 피크가 잡히지 않았거나 데이터가 부족합니다. 적분 각도(Azimuth)와 Fit(q) 영역을 다시 조절해 보세요.")
        else:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.dataframe(res_df.style.format({"Strain_Out(%)": "{:.3f}", "Strain_In(%)": "{:.3f}"}))
                st.download_button("💾 결과 CSV 저장", res_df.to_csv(index=False).encode('utf-8-sig'), "strain_results.csv", key="dl_csv_auto")
            with c2:
                fig_tr, ax_tr = plt.subplots()
                ax_tr.plot(res_df["입사각"], res_df["Strain_Out(%)"], 'bo-', label="Out-of-plane")
                ax_tr.plot(res_df["입사각"], res_df["Strain_In(%)"], 'ro-', label="In-plane")
                ax_tr.set_xlabel("Incidence Angle (deg)")
                ax_tr.set_ylabel("Strain (%)")
                ax_tr.legend()
                ax_tr.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig_tr)
                plt.close(fig_tr)