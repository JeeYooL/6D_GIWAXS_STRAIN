import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from lmfit.models import GaussianModel, PseudoVoigtModel, LinearModel, PolynomialModel
import re
import zipfile
import io
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- 헬퍼 함수 ---
def extract_incidence_angle(filename):
    match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

# [NEW] 회절 링(Ring) 기반 Azimuthal Variance Minimization 원점 탐색
def auto_calibrate_center(img_data, base_x=None, base_y=None, window=100):
    """
    회절 링의 azimuthal intensity variance를 최소화하는 (cx, cy)를 탐색.
    정확한 중심에서는 링 위의 밝기가 균일(variance↓), 빗나가면 불균일(variance↑).
    """
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.ndimage import gaussian_filter
        
        h, w = img_data.shape
        p99 = np.percentile(img_data, 99.5)
        img_smooth = gaussian_filter(np.clip(img_data.astype(float), 0, p99), sigma=3)
        
        if base_x is None or base_y is None:
            # 전역 초기 탐색: 가우시안 블러 최소값
            margin_x, margin_y = int(w * 0.20), int(h * 0.10)
            safe = gaussian_filter(np.clip(img_data.astype(float), 0, p99), sigma=25)
            safe_region = safe[margin_y:h-margin_y, margin_x:w-margin_x]
            dy, dx = np.unravel_index(np.argmin(safe_region), safe_region.shape)
            base_x, base_y = float(margin_x + dx), float(margin_y + dy)
        
        # --- 회절 링 기반 원점 최적화 ---
        # 하반원(빔 아래쪽)에서만 샘플링: 각도 범위 200°~340° (약 -160°~ -20°, 즉 아래쪽 반원)
        # GIWAXS 상반원은 데이터가 없으므로 제외
        n_angles = 72  # 5° 간격
        angles = np.linspace(np.radians(200), np.radians(340), n_angles)
        
        # 사용할 반지름: 빔스탑을 넘어서 링이 존재하는 영역
        radii = np.arange(150, 650, 50)  # 150~600px, 50px 간격
        
        def azimuthal_cost(center):
            cx, cy = center
            total_var = 0.0
            n_valid = 0
            for r in radii:
                intensities = []
                for theta in angles:
                    px = int(cx + r * np.cos(theta))
                    py = int(cy + r * np.sin(theta))
                    if 0 <= px < w and 0 <= py < h:
                        val = img_smooth[py, px]
                        if val > 0:  # 마스크/빔스탑 영역(0 또는 매우 작은 값) 제외
                            intensities.append(val)
                if len(intensities) > n_angles // 3:  # 최소 1/3 이상의 유효 데이터가 있어야 분산 계산
                    arr = np.array(intensities)
                    # 정규화된 분산 (스케일 독립적)
                    mean_val = arr.mean()
                    if mean_val > 0:
                        total_var += arr.std() / mean_val
                        n_valid += 1
            return total_var / max(n_valid, 1)
        
        # Nelder-Mead 최적화 (초기값 ±30px 범위 내에서 탐색)
        x0 = [base_x, base_y]
        result = minimize(azimuthal_cost, x0, method='Nelder-Mead',
                          options={'xatol': 0.5, 'fatol': 1e-6, 'maxiter': 200})
        
        opt_x, opt_y = result.x
        
        # 결과가 초기값에서 너무 벗어나면 (>30px) 초기값을 유지 (안전장치)
        if abs(opt_x - base_x) > 30 or abs(opt_y - base_y) > 30:
            return float(base_x), float(base_y)
        
        return float(opt_x), float(opt_y)
        
    except Exception:
        if base_x is not None and base_y is not None: return float(base_x), float(base_y)
        return float(img_data.shape[1]/2.0), float(img_data.shape[0]/2.0)

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

# 수동 조정 입력창 (세션 상태와 연동)
dbx = st.sidebar.number_input("DBx (Center X - 1)", value=st.session_state.dbx, step=0.01)
dby = st.sidebar.number_input("DBy (Center Y - 1)", value=st.session_state.dby, step=0.01)

# [기능 개선] 동적 자동 정렬 버튼 (사용자가 수동으로 입력해둔 부근에서 빔 센터 미세조정)
if st.sidebar.button("🪄 빔 센터 미세조정 (±100px 자동 탐색)"):
    if 'current_img' in st.session_state:
        # 화면의 Number_input에 바로 입력된 최신값(dbx, dby) 주변 ±100px 영역으로 국한하여 탐색
        dbx_a, dby_a = auto_calibrate_center(
            st.session_state.current_img, 
            base_x=dbx, 
            base_y=dby, 
            window=100
        )
        st.session_state.dbx = dbx_a
        st.session_state.dby = dby_a
        st.sidebar.success(f"미세조정 완료! (X:{dbx_a:.2f}, Y:{dby_a:.2f})")
    else:
        st.sidebar.warning("⚠️ 먼저 TIF 파일을 업로드해주세요.")

wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

# --- 사이드바: 2D 시각화 설정 ---
st.sidebar.divider()
st.sidebar.subheader("🎨 2D 시각화 옵션")
mask_bg = st.sidebar.checkbox("상반원 배경 지우기 (Intensity ≤ 5)", value=True, help="배경 노이즈를 투명하게 처리하여 실제 경계선과 회절 링을 명확하게 봅니다.")

st.sidebar.divider()
st.sidebar.header("2. 분석 파라미터")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.5420, format="%.4f")
q_min = st.sidebar.number_input("피크 탐색 시작 q", value=1.20, help="피크를 자동 탐색할 q 범위의 시작점 (Fitting 범위 아님)")
q_max = st.sidebar.number_input("피크 탐색 끝 q", value=1.70, help="피크를 자동 탐색할 q 범위의 끝점 (Fitting 범위 아님)")
fwhm_mult = st.sidebar.number_input("Fitting window (×FWHM)", value=3.0, min_value=1.5, max_value=6.0, step=0.5, help="Target peak의 FWHM × 이 값 = 실제 fitting 범위. 클수록 넓게 피팅.")

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
    
    # 명암(Log) 이미지 생성 및 배경 제거
    log_preview = np.log1p(np.clip(flipped_img, 0, None))
    if mask_bg:
        log_preview = np.where(log_preview <= 5.0, np.nan, log_preview)
        
    fig_pre, ax_pre = plt.subplots(figsize=(6, 4))
    
    cmap_pre = plt.cm.jet.copy()
    cmap_pre.set_bad('white', 1.)
    
    im_pre = ax_pre.imshow(log_preview, cmap=cmap_pre)
    
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
        results, zip_buffer = [], io.BytesIO()
        report_figures = []  # Word 보고서용 그래프 저장
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            
            # [동적 빔 센터 흐름 추적]
            # 첫 샘플은 사용자가 설정한 dbx, dby를 기준으로, 다음 샘플부터는 이전 샘플의 중심을 기준으로 ±20 픽셀씩만 한정 추적
            track_x, track_y = dbx, dby
            
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(paths[row["파일명"]]).data
                    
                    # 현재 샘플의 물리적 원점(Beam Center) 미세조정 탐색 및 업데이트
                    track_x, track_y = auto_calibrate_center(img_data, base_x=track_x, base_y=track_y, window=20)
                    
                    # 각 이미지만의 고유하게 틀어진 빔 센터를 바탕으로 pyFAI 물리적 엔진 초기화
                    geo = AzimuthalIntegrator(dist=dist_m, poni1=track_y*px_m, poni2=track_x*px_m, 
                                              wavelength=wavelength, pixel1=px_m, pixel2=px_m)
                                              
                    q_out, I_out = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_out_min, azi_out_max))
                    q_in, I_in = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_in_min, azi_in_max))
                    
                    # Origin 저장 (두 방향 통합)
                    txt = pd.DataFrame({"q_out": q_out, "I_out": I_out, "q_in": q_in, "I_in": I_in}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_1D.txt", txt)

                    # --- [Peak-Centric Adaptive Pseudo-Voigt Fitting Engine] ---
                    def fit_peak(q, I):
                        from scipy.signal import find_peaks, peak_widths
                        
                        # ===== STEP 1: 탐색 범위에서 모든 피크 자동 감지 =====
                        search_mask = (q >= q_min) & (q <= q_max)
                        q_search, I_search = q[search_mask], I[search_mask]
                        if len(q_search) < 10: return None, None, None, None, {}
                        
                        peaks, props = find_peaks(I_search, 
                                                   prominence=0.05 * (I_search.max() - I_search.min()), 
                                                   distance=5, width=2)
                        
                        if len(peaks) == 0:
                            peaks = np.array([np.argmax(I_search)])
                        
                        # ===== STEP 2: q_bulk에 가장 가까운 target peak 선정 =====
                        peak_q_vals = q_search[peaks]
                        target_idx = np.argmin(np.abs(peak_q_vals - q_bulk))
                        target_q = peak_q_vals[target_idx]
                        target_peak_idx = peaks[target_idx]
                        
                        # ===== STEP 3: FWHM 기반 adaptive fitting window =====
                        # 피크의 half-max 폭 추정
                        try:
                            widths_result = peak_widths(I_search, [target_peak_idx], rel_height=0.5)
                            fwhm_pts = widths_result[0][0]  # FWHM in data points
                            dq = np.mean(np.diff(q_search))  # q spacing
                            fwhm_q = fwhm_pts * dq
                        except:
                            fwhm_q = 0.03  # 기본값
                        
                        # Adaptive window: target peak 중심 ± (FWHM × multiplier)
                        fit_half_width = max(fwhm_q * fwhm_mult, 0.04)  # 최소 ±0.04
                        fit_q_min = target_q - fit_half_width
                        fit_q_max = target_q + fit_half_width
                        
                        fit_mask = (q >= fit_q_min) & (q <= fit_q_max)
                        qc, Ic = q[fit_mask], I[fit_mask]
                        if len(qc) < 8: return None, None, None, None, {}
                        
                        # ===== STEP 4: Polynomial baseline 추정 및 제거 =====
                        # fitting window 양 끝 10%의 점들로 baseline 추정
                        n_edge = max(3, len(qc) // 10)
                        edge_q = np.concatenate([qc[:n_edge], qc[-n_edge:]])
                        edge_I = np.concatenate([Ic[:n_edge], Ic[-n_edge:]])
                        baseline_coeffs = np.polyfit(edge_q, edge_I, 2)
                        baseline = np.polyval(baseline_coeffs, qc)
                        
                        # ===== STEP 5: Window 내 피크 재감지 + Multi-peak Pseudo-Voigt =====
                        Ic_sub = Ic - baseline  # baseline 제거된 데이터
                        Ic_sub = np.maximum(Ic_sub, 0)  # 음수 방지
                        
                        local_peaks, _ = find_peaks(Ic_sub, 
                                                     prominence=0.03 * (Ic_sub.max() - Ic_sub.min() + 1),
                                                     distance=3)
                        if len(local_peaks) == 0:
                            local_peaks = np.array([np.argmax(Ic_sub)])
                        
                        # Pseudo-Voigt multi-peak model
                        model = PolynomialModel(degree=2, prefix='bkg_')  # 2차 다항 배경
                        params = model.make_params(c0=baseline_coeffs[2], c1=baseline_coeffs[1], c2=baseline_coeffs[0])
                        
                        for i, p_idx in enumerate(local_peaks):
                            center_guess = qc[p_idx]
                            amp_guess = max(Ic_sub[p_idx] * 0.05, 1.0)
                            
                            p_model = PseudoVoigtModel(prefix=f'p{i}_')
                            p_params = p_model.make_params(
                                amplitude=amp_guess, 
                                center=center_guess, 
                                sigma=fwhm_q / 2.355,  # FWHM → sigma 변환
                                fraction=0.5  # Lorentzian 비율 초기값 50%
                            )
                            p_params[f'p{i}_center'].set(min=center_guess - 0.03, max=center_guess + 0.03)
                            p_params[f'p{i}_sigma'].set(min=0.002, max=0.1)
                            p_params[f'p{i}_fraction'].set(min=0, max=1)
                            
                            model += p_model
                            params.update(p_params)
                        
                        # ===== STEP 6: 피팅 수행 =====
                        out = model.fit(Ic, params, x=qc)
                        
                        # ===== STEP 7: Target peak 선택 (q_bulk에 가장 가까운 amplitude 최대) =====
                        candidates = []
                        for j in range(len(local_peaks)):
                            c_val = out.params[f'p{j}_center'].value
                            a_val = out.params[f'p{j}_amplitude'].value
                            if abs(c_val - q_bulk) < fit_half_width:
                                candidates.append((c_val, a_val, j))
                        
                        if candidates:
                            best = max(candidates, key=lambda x: x[1])
                            best_center = best[0]
                            best_idx = best[2]
                        else:
                            best_center = target_q
                            best_idx = 0
                        
                        # ===== STEP 8: Strain 계산 + 진단 지표 =====
                        strain = (q_bulk - best_center) / q_bulk * 100  # q_bulk 기준 정규화
                        
                        # 진단 지표
                        sigma_fit = out.params[f'p{best_idx}_sigma'].value
                        fwhm_fit = sigma_fit * 2.355  # Gaussian FWHM 근사
                        ss_res = np.sum((Ic - out.best_fit) ** 2)
                        ss_tot = np.sum((Ic - np.mean(Ic)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        diagnostics = {
                            'center': best_center,
                            'fwhm': fwhm_fit,
                            'r_squared': r_squared,
                            'fit_range': (fit_q_min, fit_q_max),
                            'n_peaks': len(local_peaks)
                        }
                        
                        return qc, Ic, out, strain, diagnostics

                    # 두 방향 각각 피팅
                    result_out = fit_peak(q_out, I_out)
                    result_in = fit_peak(q_in, I_in)
                    
                    if result_out[2] is None or result_in[2] is None:
                        st.warning(f"{row['파일명']}: 피팅할 데이터가 부족합니다.")
                        continue
                    
                    qc_out, Ic_out, fit_out, strain_out, diag_out = result_out
                    qc_in, Ic_in, fit_in, strain_in, diag_in = result_in
                        
                    results.append({"파일명": row["파일명"], "입사각": row["입사각(deg)"], 
                                    "Strain_Out(%)": strain_out, "Strain_In(%)": strain_in})
                    
                    with st.expander(f"📊 {row['파일명']} 상세 분석"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            # 2D GIWAXS 패턴 (동적으로 추적된 track_x, track_y 기준)
                            h, w = img_data.shape
                            dq = (2*np.pi/(wavelength*1e10)) * (px_m/dist_m)
                            ext = [-track_x*dq, (w-track_x)*dq, -track_y*dq, (h-track_y)*dq]
                            
                            log_final = np.log1p(np.clip(np.flipud(img_data), 0, None))
                            if mask_bg:
                                log_final = np.where(log_final <= 5.0, np.nan, log_final)
                                
                            fig2d, ax2d = plt.subplots()
                            cmap_final = plt.cm.jet.copy()
                            cmap_final.set_bad('white', 1.)
                            
                            ax2d.imshow(log_final, cmap=cmap_final, extent=ext)
                            ax2d.set_title("2D GIWAXS"); ax2d.set_xlabel(r"$q_{xy} (\AA^{-1})$"); ax2d.set_ylabel(r"$q_z (\AA^{-1})$")
                            buf_2d = io.BytesIO(); fig2d.savefig(buf_2d, format='png', dpi=150, bbox_inches='tight'); buf_2d.seek(0)
                            st.pyplot(fig2d); plt.close(fig2d)
                        with c2:
                            # 1D 피팅 결과 (Out-of-plane) + 진단 정보
                            fig_out, ax_out = plt.subplots()
                            ax_out.plot(qc_out, Ic_out, 'bo', markersize=3, label='Data')
                            ax_out.plot(qc_out, fit_out.best_fit, 'r-', label='Fit')
                            ax_out.set_title(f"Out Strain: {strain_out:.3f}%\nFWHM={diag_out['fwhm']:.4f}, R²={diag_out['r_squared']:.3f}", fontsize=9)
                            ax_out.set_xlabel(r"$q_z (\AA^{-1})$"); ax_out.legend(fontsize=7)
                            buf_out = io.BytesIO(); fig_out.savefig(buf_out, format='png', dpi=150, bbox_inches='tight'); buf_out.seek(0)
                            st.pyplot(fig_out); plt.close(fig_out)
                        with c3:
                            # 1D 피팅 결과 (In-plane) + 진단 정보
                            fig_in, ax_in = plt.subplots()
                            ax_in.plot(qc_in, Ic_in, 'bo', markersize=3, label='Data')
                            ax_in.plot(qc_in, fit_in.best_fit, 'r-', label='Fit')
                            ax_in.set_title(f"In Strain: {strain_in:.3f}%\nFWHM={diag_in['fwhm']:.4f}, R²={diag_in['r_squared']:.3f}", fontsize=9)
                            ax_in.set_xlabel(r"$q_{xy} (\AA^{-1})$"); ax_in.legend(fontsize=7)
                            buf_in = io.BytesIO(); fig_in.savefig(buf_in, format='png', dpi=150, bbox_inches='tight'); buf_in.seek(0)
                            st.pyplot(fig_in); plt.close(fig_in)
                        
                        # Word 보고서용 그래프 저장
                        report_figures.append({
                            'name': row['파일명'], 'angle': row['입사각(deg)'],
                            'strain_out': strain_out, 'strain_in': strain_in,
                            'fig_2d': buf_2d, 'fig_out': buf_out, 'fig_in': buf_in
                        })
                            
                except Exception as e: st.error(f"❌ {row['파일명']} 실패: {e}")
                pbar.progress((i + 1) / len(edited_df))
        
        st.session_state.analysis_results = pd.DataFrame(results)
        st.session_state.zip_data = zip_buffer.getvalue()
        st.session_state.report_figures = report_figures

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
                buf_trend = io.BytesIO(); fig_tr.savefig(buf_trend, format='png', dpi=150, bbox_inches='tight'); buf_trend.seek(0)
                plt.close(fig_tr)
            
            # --- Word 보고서 생성 및 다운로드 ---
            st.divider()
            st.subheader("📝 Word 보고서 다운로드")
            
            if st.button("📄 Word 보고서 생성", key="gen_docx"):
                doc = Document()
                doc.add_heading('GIWAXS Strain Analysis Report', level=0)
                doc.add_paragraph(f'Bulk q-value: {q_bulk:.4f} Å⁻¹  |  Fit range: [{q_min:.2f}, {q_max:.2f}] Å⁻¹')
                doc.add_paragraph(f'Energy: {energy_kev} keV  |  Distance: {dist_mm} mm  |  Pixel: {pixel_um} μm')
                
                # 1. 결과 테이블
                doc.add_heading('1. Strain Results Table', level=1)
                table = doc.add_table(rows=1, cols=4, style='Light Shading Accent 1')
                hdr = table.rows[0].cells
                hdr[0].text = '파일명'; hdr[1].text = '입사각(deg)'
                hdr[2].text = 'Strain_Out(%)'; hdr[3].text = 'Strain_In(%)'
                for _, r in res_df.iterrows():
                    row_cells = table.add_row().cells
                    row_cells[0].text = str(r['파일명'])
                    row_cells[1].text = f"{r['입사각']:.2f}"
                    row_cells[2].text = f"{r['Strain_Out(%)']:.3f}"
                    row_cells[3].text = f"{r['Strain_In(%)']:.3f}"
                
                # 2. 트렌드 그래프
                doc.add_heading('2. Strain Trend', level=1)
                doc.add_picture(buf_trend, width=Inches(5.5))
                last_paragraph = doc.paragraphs[-1]
                last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # 3. 각 샘플별 상세 분석 (2D + Out + In)
                doc.add_heading('3. Per-Sample Analysis', level=1)
                figs = st.session_state.get('report_figures', [])
                for item in figs:
                    doc.add_heading(f"{item['name']} (Angle: {item['angle']:.2f}°)", level=2)
                    p_info = doc.add_paragraph()
                    p_info.add_run(f"Out-of-plane Strain: {item['strain_out']:.3f}%  |  In-plane Strain: {item['strain_in']:.3f}%")
                    
                    # 2D 패턴
                    item['fig_2d'].seek(0)
                    doc.add_picture(item['fig_2d'], width=Inches(4.0))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Out-of-plane & In-plane (나란히 배치는 docx 한계로 순차 배치)
                    item['fig_out'].seek(0)
                    doc.add_picture(item['fig_out'], width=Inches(4.0))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    item['fig_in'].seek(0)
                    doc.add_picture(item['fig_in'], width=Inches(4.0))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    doc.add_page_break()
                
                # 문서 저장
                docx_buffer = io.BytesIO()
                doc.save(docx_buffer)
                docx_buffer.seek(0)
                st.session_state.docx_data = docx_buffer.getvalue()
                st.success("✅ Word 보고서가 생성되었습니다!")
            
            if st.session_state.get('docx_data'):
                st.download_button(
                    "💾 Word 보고서 다운로드 (.docx)",
                    st.session_state.docx_data,
                    "GIWAXS_Strain_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_docx"
                )