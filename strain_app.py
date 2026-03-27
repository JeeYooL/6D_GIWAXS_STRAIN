import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from lmfit.models import GaussianModel, PseudoVoigtModel, LinearModel, PolynomialModel
from scipy.signal import find_peaks
import re
import zipfile
import io
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- 헬퍼 함수 ---
def extract_incidence_angle(filename):
    match = re.search(r"_(\d+\.\d+)d", filename)
    if not match:
        match = re.search(r"(\d+\.\d+)d", filename)
    return float(match.group(1)) if match else 0.10

# [Step-1] 회절 링(Ring) 기반 Azimuthal Variance Minimization 원점 탐색 (X+Y 모두 보정)
def auto_calibrate_center(img_data, base_x=None, base_y=None, window=100):
    """
    회절 링의 azimuthal intensity variance를 최소화하는 (cx, cy)를 탐색.
    1단계 (첫 이미지 X+Y 보정)에서만 사용.
    """
    try:
        from scipy.optimize import minimize
        from scipy.ndimage import gaussian_filter
        
        h, w = img_data.shape
        p99 = np.percentile(img_data, 99.5)
        img_smooth = gaussian_filter(np.clip(img_data.astype(float), 0, p99), sigma=3)
        
        if base_x is None or base_y is None:
            margin_x, margin_y = int(w * 0.20), int(h * 0.10)
            safe = gaussian_filter(np.clip(img_data.astype(float), 0, p99), sigma=25)
            safe_region = safe[margin_y:h-margin_y, margin_x:w-margin_x]
            dy, dx = np.unravel_index(np.argmin(safe_region), safe_region.shape)
            base_x, base_y = float(margin_x + dx), float(margin_y + dy)
        
        n_angles = 72
        angles = np.linspace(np.radians(200), np.radians(340), n_angles)
        radii = np.arange(150, 650, 50)
        
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
                        if val > 0:
                            intensities.append(val)
                if len(intensities) > n_angles // 3:
                    arr = np.array(intensities)
                    mean_val = arr.mean()
                    if mean_val > 0:
                        total_var += arr.std() / mean_val
                        n_valid += 1
            return total_var / max(n_valid, 1)
        
        x0 = [base_x, base_y]
        result = minimize(azimuthal_cost, x0, method='Nelder-Mead',
                          options={'xatol': 0.5, 'fatol': 1e-6, 'maxiter': 200})
        opt_x, opt_y = result.x
        
        if abs(opt_x - base_x) > window or abs(opt_y - base_y) > window:
            return float(base_x), float(base_y)
        
        return float(opt_x), float(opt_y)
        
    except Exception:
        if base_x is not None and base_y is not None: return float(base_x), float(base_y)
        return float(img_data.shape[1]/2.0), float(img_data.shape[0]/2.0)


# [Step-2] 데이터 경계 기반 center_y 탐색
def find_center_y_by_data_edge(img_data, base_x, base_y, dist_m, px_m, window=80,
                               qxy_range=3.0, qxy_beamstop=0.2, threshold_ratio=0.05):
    """
    q_xy = [-qxy_range, -qxy_beamstop] ∪ [qxy_beamstop, qxy_range] 범위에 해당하는
    픽셀 열들의 강도를 각 후보 행(row)에서 측정하여,
    신호가 처음으로 전부 나타나는 경계 행을 center_y로 반환한다.
    (raw 이미지에서 center_y 위쪽 = 데이터 없음, center_y 아래쪽 = GIWAXS 데이터)
    탐색 방향: base_y 기준으로 ±window 픽셀씩 스캔.
    """
    try:
        h, w = img_data.shape
        # q_xy → 픽셀 오프셋 변환: Δpx = q_xy * dist_m / px_m  (단위: px)
        # (pyFAI의 q = 2π/λ * sin(2θ)/... 여기서는 단순 기하 근사 사용)
        # 실제 q_xy 픽셀 매핑: px_col = base_x + q_xy * dist_m / px_m * ...
        # 단순화: 픽셀 수 = q_xy [Å⁻¹] * dist_m [m] / px_m [m] (소각 근사)
        # 하지만 q = 4π/λ*sin(θ) ≒ 2π/λ * 2θ ≒ 2π*tan(2θ/2)/λ ≒ 2π*r/(λ*dist)
        # r [px] = q [Å⁻¹] * dist_m / (2π * px_m) * λ [m] * 1e10
        # 여기선 dist_m, px_m을 직접 받지 않으므로, px/q 비율을 아래처럼 계산
        # → 실제 호출 시 dist_m, px_m, wavelength를 넘기도록 변경
        # 단, 이 함수는 wavelength 없이 px_per_q = dist_m / px_m 근사 사용
        px_per_q = dist_m / px_m  # [px / rad] ≈ [px / q Å⁻¹] at small q

        # 측정할 q_xy 샘플 포인트
        q_left  = np.linspace(-qxy_range, -qxy_beamstop, 20)
        q_right = np.linspace( qxy_beamstop,  qxy_range, 20)
        q_samples = np.concatenate([q_left, q_right])
        
        # 각 q_xy에 해당하는 픽셀 열 인덱스
        col_indices = (base_x + q_samples * px_per_q).astype(int)
        valid_cols = col_indices[(col_indices >= 0) & (col_indices < w)]
        
        if len(valid_cols) == 0:
            return float(base_y)
        
        # 배경 임계값: 이미지 전체 하위 퍼센타일
        bg_level = np.percentile(img_data, 20)
        signal_thresh = bg_level + threshold_ratio * (np.percentile(img_data, 99) - bg_level)
        
        # 후보 행 스캔 범위: base_y ± window (정수)
        y_lo = max(0, int(base_y) - window)
        y_hi = min(h - 1, int(base_y) + window)
        
        # 각 행에서 valid_cols 픽셀들의 신호 유효 비율 계산
        row_scores = []
        for row in range(y_lo, y_hi + 1):
            vals = img_data[row, valid_cols]
            frac = np.mean(vals > signal_thresh)  # 신호가 있는 열 비율
            row_scores.append((row, frac))
        
        row_scores = np.array(row_scores)  # shape (N, 2): [row, frac]
        
        # 신호 비율의 변화가 가장 급격한 경계 행 찾기
        # 스무딩 후 gradient 최대 지점 = 데이터가 갑자기 나타나기 시작하는 경계
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(row_scores[:, 1], size=5)
        gradient = np.gradient(smoothed)
        
        # gradient > 0인 구간에서 최대 기울기 위치 (아래로 내려갈수록 신호 증가)
        # 단, 경계는 "신호가 처음 충분히 넘는" 행 → frac >= 0.6 인 첫 번째 행 (위에서 아래로)
        threshold_frac = 0.6
        for idx in range(len(row_scores)):
            if row_scores[idx, 1] >= threshold_frac:
                best_row = int(row_scores[idx, 0])
                return float(best_row)
        
        # fallback: 신호가 없으면 gradient 최대 지점 반환
        peak_idx = np.argmax(gradient)
        return float(row_scores[peak_idx, 0])
    
    except Exception:
        return float(base_y)

st.set_page_config(page_title="UNIST 6D GIWAXS Analyzer", layout="wide")
st.title("🔬 6D GIWAXS Strain 분석 (2단계 빔 센터 보정)")

# --- 세션 상태 초기화 ---
if 'dbx' not in st.session_state: st.session_state.dbx = 1435.83
if 'dby' not in st.session_state: st.session_state.dby = 1439.11
if 'per_sample_centers' not in st.session_state: st.session_state.per_sample_centers = None
if 'step1_done' not in st.session_state: st.session_state.step1_done = False
if 'step2_done' not in st.session_state: st.session_state.step2_done = False

# --- 사이드바: 실험 셋업 ---
st.sidebar.header("1. 실험 셋업 (6D UNIST-PAL)")
energy_kev = st.sidebar.number_input("Energy (keV)", value=11.564, format="%.3f")
dist_mm = st.sidebar.number_input("SDD (mm)", value=200.0, format="%.3f")
pixel_um = st.sidebar.number_input("Pixel size (um)", value=78.13)

st.sidebar.divider()
st.sidebar.subheader("🎯 빔 센터(Beam Center) 수동 설정")

# 수동 조정 입력창 (세션 상태와 연동)
dbx = st.sidebar.number_input("DBx (Center X)", value=st.session_state.dbx, step=0.01)
dby = st.sidebar.number_input("DBy (Center Y)", value=st.session_state.dby, step=0.01)

wavelength = (12.3984 / energy_kev) * 1e-10 
dist_m = dist_mm / 1000.0
px_m = pixel_um * 1e-6

# --- 사이드바: 2D 시각화 설정 ---
st.sidebar.divider()
st.sidebar.subheader("🎨 2D 시각화 옵션")
mask_bg = st.sidebar.checkbox("상반원 배경 지우기 (Intensity ≤ 5)", value=True, help="배경 노이즈를 투명하게 처리하여 실제 경계선과 회절 링을 명확하게 봅니다.")

st.sidebar.divider()
st.sidebar.header("2. 분석 파라미터")
q_bulk = st.sidebar.number_input("Bulk q-value (Å⁻¹)", value=1.05, format="%.4f")
target_q = st.sidebar.number_input("Target Peak q (Å⁻¹)", value=q_bulk, format="%.4f", help="추적할 특정 피크의 q값. q_bulk와 같거나 근처로 설정.")
peak_window = st.sidebar.number_input("피크 선택 창(±Å⁻¹)", value=0.05, format="%.3f", help="target_q ± 이 범위 안의 피크만 선택. 작을수록 정확, 클수록 유연.")
q_min = st.sidebar.number_input("Fit 영역 시작 q", value=0.95)
q_max = st.sidebar.number_input("Fit 영역 끝 q", value=1.15)

st.sidebar.subheader("🎯 1D 적분 각도 (Out / In-plane)")
c_out1, c_out2 = st.sidebar.columns(2)
azi_out_min = c_out1.number_input("Out 최소(°)", value=-100)
azi_out_max = c_out2.number_input("Out 최대(°)", value=-80)

c_in1, c_in2 = st.sidebar.columns(2)
azi_in_min = c_in1.number_input("In 최소(°)", value=-15)
azi_in_max = c_in2.number_input("In 최대(°)", value=-5)

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

    # ========================================
    # 1단계: 첫 번째 이미지 (저각) 빔 센터 미세조정
    # ========================================
    st.subheader("📌 1단계: 첫 번째 이미지 센터 미세조정")
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
    
    st.info("💡 위 이미지의 **빨간 십자선(+)**이 빔스탑 정중앙에 위치하는지 확인하세요.")
    
    if st.button("🪄 1단계: 첫 이미지 센터 자동 미세조정 (X+Y, ±100px)", key="step1_btn"):
        dbx_a, dby_a = auto_calibrate_center(
            st.session_state.current_img, 
            base_x=dbx, base_y=dby, window=100
        )
        st.session_state.dbx = dbx_a
        st.session_state.dby = dby_a
        st.session_state.step1_done = True
        st.session_state.per_sample_centers = None  # 2단계 초기화
        st.session_state.step2_done = False
        st.success(f"✅ 1단계 완료! 첫 이미지 센터: X={dbx_a:.2f}, Y={dby_a:.2f}")
        st.rerun()
    
    if st.session_state.step1_done:
        st.success(f"✅ 1단계 완료 — 기준 센터: X={st.session_state.dbx:.2f}, Y={st.session_state.dby:.2f}")
    
    # ========================================
    # 2단계: 각 샘플별 q_z (Y) 원점 보정
    # ========================================
    st.divider()
    st.subheader("📌 2단계: 각 샘플별 q_z 원점 보정")
    st.caption("각이 증가할수록 반전된 이미지 기준으로 원점이 위로 올라가는 경향을 보정합니다. q_xy(X)는 첫 이미지 기준으로 고정하고 q_z(Y)만 각 샘플별로 미세조정합니다.")
    
    if st.button("🔍 2단계: 각 샘플별 q_z 보정 실행", key="step2_btn", type="secondary"):
        fixed_x = st.session_state.dbx
        track_y = st.session_state.dby
        per_centers = []
        pbar2 = st.progress(0)
        
        for i, row in edited_df.iterrows():
            img_data = fabio.open(paths[row["파일명"]]).data
            # 데이터 경계 스캔으로 center_y 결정
            # 첫 샘플은 ±80px, 나머지는 이전 샘플 기준 ±40px 탐색
            search_window = 80 if i == 0 else 40
            cal_y = find_center_y_by_data_edge(
                img_data, fixed_x, track_y,
                dist_m=dist_m, px_m=px_m,
                window=search_window
            )
            track_y = cal_y  # 다음 샘플의 초기값으로 전달
            per_centers.append({'파일명': row['파일명'], '입사각': row['입사각(deg)'],
                                'center_x': fixed_x, 'center_y': cal_y})
            pbar2.progress((i + 1) / len(edited_df))
        
        st.session_state.per_sample_centers = per_centers
        st.session_state.step2_done = True
        st.rerun()
    
    # 2단계 결과 표시: 센터 테이블 + 각 샘플 2D/1D 프리뷰
    if st.session_state.per_sample_centers is not None:
        centers_df = pd.DataFrame(st.session_state.per_sample_centers)
        st.success("✅ 2단계 완료 — 각 샘플별 보정된 빔 센터:")
        st.dataframe(centers_df.style.format({'center_x': '{:.2f}', 'center_y': '{:.2f}', '입사각': '{:.2f}'}), use_container_width=True)
        
        # 각 샘플에 대한 2D GIWAXS + 1D 적분 프리뷰
        st.divider()
        st.subheader("🖼️ 2단계 보정 결과 프리뷰 (2D GIWAXS)")
        qxy_preview = 3.5  # q_xy 표시 범위 [Å⁻¹]
        
        for ci, cinfo in enumerate(st.session_state.per_sample_centers):
            fname = cinfo['파일명']
            cx, cy = cinfo['center_x'], cinfo['center_y']
            angle_deg = cinfo['입사각']
            
            with st.expander(f"📊 {fname} (입사각 {angle_deg:.2f}°, Y={cy:.2f})", expanded=(ci < 3)):
                img_data = fabio.open(paths[fname]).data
                h, w = img_data.shape
                dq = (2*np.pi/(wavelength*1e10)) * (px_m/dist_m)
                
                # 2D GIWAXS 프리뷰만 표시 (1D 제거)
                data_half = img_data[:int(cy), :]
                log_half = np.log1p(np.clip(data_half, 0, None))
                if mask_bg:
                    log_half = np.where(log_half <= 5.0, np.nan, log_half)
                h_half = log_half.shape[0]
                ext = [-cx*dq, (w-cx)*dq, 0, h_half*dq]
                
                fig2d, ax2d = plt.subplots(figsize=(5, 4))
                cmap2 = plt.cm.jet.copy(); cmap2.set_bad('white', 1.)
                ax2d.imshow(log_half, cmap=cmap2, extent=ext, aspect='auto')
                ax2d.set_title(f"2D GIWAXS — {fname}\n(center_y={cy:.1f}, 입사각={angle_deg:.2f}°)", fontsize=9)
                ax2d.set_xlabel(r"$q_{xy}$ (Å⁻¹)"); ax2d.set_ylabel(r"$q_z$ (Å⁻¹)")
                ax2d.set_xlim(-qxy_preview, qxy_preview)
                st.pyplot(fig2d); plt.close(fig2d)
    
    # ========================================
    # 3단계: 전수 Strain 분석 (보정된 센터 사용)
    # ========================================
    st.divider()
    st.subheader("🚀 3단계: Strain 전수 분석")
    if not st.session_state.step2_done:
        st.warning("⚠️ 먼저 2단계 (각 샘플별 q_z 보정)를 완료해주세요.")
    
    if st.button("🚀 보정된 센터로 전수 분석 시작", type="primary", disabled=(not st.session_state.step2_done)):
        results, zip_buffer = [], io.BytesIO()
        report_figures = []  # Word 보고서용 그래프 저장
        
        # 2단계에서 보정된 각 샘플별 센터를 딕셔너리로 변환
        center_map = {c['파일명']: (c['center_x'], c['center_y']) for c in st.session_state.per_sample_centers}
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            pbar = st.progress(0)
            
            for i, row in edited_df.iterrows():
                try:
                    img_data = fabio.open(paths[row["파일명"]]).data
                    
                    # 2단계에서 보정된 센터 사용
                    track_x, track_y = center_map.get(row["파일명"], (st.session_state.dbx, st.session_state.dby))
                    
                    # 입사각 보정: pyFAI rot1 파라미터로 GIWAXS 입사각 반영
                    incidence_rad = np.radians(row["입사각(deg)"])
                    
                    # 각 이미지만의 고유하게 틀어진 빔 센터를 바탕으로 pyFAI 물리적 엔진 초기화
                    geo = AzimuthalIntegrator(dist=dist_m, poni1=track_y*px_m, poni2=track_x*px_m, 
                                              wavelength=wavelength, pixel1=px_m, pixel2=px_m,
                                              rot1=incidence_rad)
                                              
                    q_out, I_out = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_out_min, azi_out_max))
                    q_in, I_in = geo.integrate1d(img_data, 1000, unit="q_A^-1", azimuth_range=(azi_in_min, azi_in_max))
                    
                    # Origin 저장 (두 방향 통합)
                    txt = pd.DataFrame({"q_out": q_out, "I_out": I_out, "q_in": q_in, "I_in": I_in}).to_csv(sep='\t', index=False)
                    zip_file.writestr(f"{row['파일명']}_1D.txt", txt)

                    # --- [Multi-Peak Pseudo-Voigt Fitting] ---
                    def fit_peak(q, I):
                        # 사용자가 지정한 q 범위에서 직접 피팅 (안정적)
                        mask = (q >= q_min) & (q <= q_max)
                        qc, Ic = q[mask], I[mask]
                        if len(qc) < 5: return None, None, None, None, {}
                        
                        # 1. 피크 자동 감지
                        peaks, _ = find_peaks(Ic, prominence=0.03 * (Ic.max() - Ic.min()), distance=5)
                        if len(peaks) == 0:
                            peaks = [np.argmax(Ic)]
                        
                        # 2. Linear 배경 + Multi-peak Pseudo-Voigt 모델
                        model = LinearModel(prefix='bkg_')
                        params = model.make_params(slope=0, intercept=Ic.min())
                        
                        for i, p_idx in enumerate(peaks):
                            center_guess = qc[p_idx]
                            amp_guess = (Ic[p_idx] - Ic.min()) * 0.05
                            
                            p_model = PseudoVoigtModel(prefix=f'p{i}_')
                            p_params = p_model.make_params(
                                amplitude=max(amp_guess, 1.0),
                                center=center_guess,
                                sigma=0.02,
                                fraction=0.5
                            )
                            p_params[f'p{i}_center'].set(min=center_guess - 0.05, max=center_guess + 0.05)
                            p_params[f'p{i}_sigma'].set(min=0.002, max=0.1)
                            p_params[f'p{i}_fraction'].set(min=0, max=1)
                            
                            model += p_model
                            params.update(p_params)
                        
                        # 3. 피팅 수행
                        out = model.fit(Ic, params, x=qc)
                        
                        # 4. target_q ± peak_window 안의 피크만 후보로 선택
                        candidates = []
                        for j in range(len(peaks)):
                            c_val = out.params[f'p{j}_center'].value
                            a_val = out.params[f'p{j}_amplitude'].value
                            if abs(c_val - target_q) < peak_window:
                                candidates.append((c_val, a_val, j))
                        
                        if candidates:
                            best = max(candidates, key=lambda x: x[1])
                            best_center, best_idx = best[0], best[2]
                        else:
                            centers = [out.params[f'p{j}_center'].value for j in range(len(peaks))]
                            best_center = min(centers, key=lambda c: abs(c - target_q))
                            best_idx = 0
                        
                        # 5. Strain 계산 (d-spacing 기반 정확 공식: ε = q₀/q - 1)
                        strain = (q_bulk / best_center - 1) * 100
                        
                        sigma_fit = out.params[f'p{best_idx}_sigma'].value
                        fwhm_fit = sigma_fit * 2.355
                        ss_res = np.sum((Ic - out.best_fit) ** 2)
                        ss_tot = np.sum((Ic - np.mean(Ic)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        diagnostics = {
                            'center': best_center, 'fwhm': fwhm_fit,
                            'r_squared': r_squared, 'n_peaks': len(peaks),
                            'all_peaks': [{
                                'q': out.params[f'p{j}_center'].value,
                                'fwhm_q': out.params[f'p{j}_sigma'].value * 2.355
                            } for j in range(len(peaks))]
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
                                    "Strain_Out(%)": strain_out, "Strain_In(%)": strain_in,
                                    "diag_out": diag_out, "diag_in": diag_in})
                    
                    with st.expander(f"📊 {row['파일명']} 상세 분석"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            # 2D GIWAXS 패턴 — 실제 데이터(하반원)를 상반원으로 표시
                            h, w = img_data.shape
                            dq = (2*np.pi/(wavelength*1e10)) * (px_m/dist_m)
                            
                            # 실제 회절 데이터가 있는 반쪽 추출 (raw 상단 = 화면 하반원)
                            data_half = img_data[:int(track_y), :]
                            log_half = np.log1p(np.clip(data_half, 0, None))
                            if mask_bg:
                                log_half = np.where(log_half <= 5.0, np.nan, log_half)
                            
                            h_half = log_half.shape[0]
                            ext = [-track_x*dq, (w-track_x)*dq, 0, h_half*dq]
                            
                            fig2d, ax2d = plt.subplots()
                            cmap_final = plt.cm.jet.copy()
                            cmap_final.set_bad('white', 1.)
                            
                            ax2d.imshow(log_half, cmap=cmap_final, extent=ext, aspect='auto')
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
        
        # DataFrame에는 표시용 컬럼만, diagnostics는 별도 저장
        display_results = [{"파일명": r["파일명"], "입사각": r["입사각"], 
                            "Strain_Out(%)": r["Strain_Out(%)"], "Strain_In(%)": r["Strain_In(%)"]} 
                           for r in results]
        st.session_state.analysis_results = pd.DataFrame(display_results)
        st.session_state.wh_results = results  # diagnostics 포함 원본
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