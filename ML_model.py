"""
====================================================================
 도시 공원 열섬 완화 효과 예측 - 머신러닝 모델 (Random Forest)
====================================================================
 Data   : suncheon_parks_lst_analysis.csv (GEE 추출, 순천시 161개 공원)
 Target : cooling_300m (300m 반경 냉각 강도)
          = buffer_300_lst - park_lst (양수일수록 공원이 더 시원)
 Goal   : 소규모 공원의 버퍼별 온도차 확인(Graph.py)에 이어,
          "어떤 공원 특성이 냉각 효과를 결정하는가?"를 정량화
====================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("  [1] 데이터 로드 및 전처리")
print("=" * 55)

df = pd.read_csv('suncheon_parks_lst_analysis.csv')
print(f"전체 데이터: {df.shape[0]}개 공원, {df.shape[1]}개 컬럼")

# ── 파생변수 생성 ────────────────────────────────────────────────
df['log_area'] = np.log10(df['area'].clip(lower=1))

df['size_class'] = pd.cut(df['log_area'],
                           bins=[0, 4.0, 5.0, 99],
                           labels=[0, 1, 2]).astype(float)

le = LabelEncoder()
df['park_type_enc'] = le.fit_transform(df['park_type'].fillna('기타'))

facility_cols = ['amusement_facility', 'sports_facility',
                 'cultural_facility',  'convenience_facility']
for col in facility_cols:
    df[col] = df[col].apply(
        lambda x: 1 if (pd.notna(x) and str(x).strip() not in ['', 'nan', '0']) else 0
    )
df['facility_count'] = df[facility_cols].sum(axis=1)

# ── 타깃 변수 (300m 단일) ────────────────────────────────────────
df['cooling_300m'] = df['buffer_300_lst'] - df['park_lst']

# ── 피처 정의 ────────────────────────────────────────────────────
FEATURES = ['log_area', 'park_ndvi', 'park_lst', 'park_type_enc',
            'facility_count', 'size_class']

FEATURE_LABELS = {
    'log_area'      : '공원 면적 (log)',
    'park_ndvi'     : 'NDVI (식생 지수)',
    'park_lst'      : '공원 내부 온도',
    'park_type_enc' : '공원 유형',
    'facility_count': '시설물 보유 수',
    'size_class'    : '규모 클래스',
}

df_ml = df[FEATURES + ['cooling_300m', 'park_type']].dropna(
    subset=['log_area', 'park_lst', 'cooling_300m']
)
print(f"전처리 후 분석 대상: {len(df_ml)}개")
print(f"  냉각 강도 mean={df_ml['cooling_300m'].mean():.2f}℃  "
      f"std={df_ml['cooling_300m'].std():.2f}℃")

# ══════════════════════════════════════════════════════════════════
# 2. 모델 학습
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  [2] 모델 학습 (Random Forest, 타깃: 300m 냉각 강도)")
print("=" * 55)

X = df_ml[FEATURES]
y = df_ml['cooling_300m']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"Train: {len(X_train)}개  |  Test: {len(X_test)}개")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
mae      = mean_absolute_error(y_test, y_pred)
rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
r2       = r2_score(y_test, y_pred)
cv_r2    = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"\n  R2   (Test)     : {r2:.4f}")
print(f"  MAE             : {mae:.4f} degC")
print(f"  RMSE            : {rmse:.4f} degC")
print(f"  CV R2 (5-fold)  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

feat_imp = dict(zip(FEATURES, model.feature_importances_))
print("\n  [변수 중요도 순위]")
for rank, (feat, imp) in enumerate(
        sorted(feat_imp.items(), key=lambda x: x[1], reverse=True), 1):
    bar = '#' * int(imp * 40)
    print(f"    {rank}. {FEATURE_LABELS[feat]:<16} {bar} {imp:.4f}")

# ══════════════════════════════════════════════════════════════════
# 3. 시각화 (3-패널)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  [3] 시각화 생성")
print("=" * 55)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    '순천시 도시공원 열섬 완화 효과 - 머신러닝 분석 (300m 반경)\n'
    'Random Forest | 공원 161개',
    fontsize=13, fontweight='bold'
)

BLUE = '#2196F3'

# ── 패널 ①: 실제 vs 예측 산점도 ─────────────────────────────────
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.65, color=BLUE,
           edgecolors='white', s=65, linewidth=0.5)
lims = [min(y_test.min(), y_pred.min()) - 0.5,
        max(y_test.max(), y_pred.max()) + 0.5]
ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.6, label='완벽한 예측 (y=x)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title('실제 vs 예측 냉각 강도 (300m)', fontsize=11, fontweight='bold')
ax.set_xlabel('실제 냉각 강도 (℃)', fontsize=10)
ax.set_ylabel('예측 냉각 강도 (℃)', fontsize=10)
textstr = (f'R² = {r2:.3f}\n'
           f'MAE = {mae:.3f}℃\n'
           f'RMSE = {rmse:.3f}℃\n'
           f'CV R² = {cv_r2.mean():.3f}±{cv_r2.std():.3f}')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)

# ── 패널 ②: Feature Importance 수평 막대 ────────────────────────
ax = axes[1]
sorted_items = sorted(feat_imp.items(), key=lambda x: x[1])
labels  = [FEATURE_LABELS[f] for f, _ in sorted_items]
values  = [v for _, v in sorted_items]
colors_bar = ['#EF9A9A' if v < 0.1 else '#42A5F5' for v in values]
bars = ax.barh(labels, values, color=colors_bar, edgecolor='white', height=0.6)
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_title('변수 중요도 (Feature Importance)\n300m 냉각 강도 예측 기준',
             fontsize=11, fontweight='bold')
ax.set_xlabel('중요도', fontsize=10)
ax.axvline(x=0.1, color='gray', linestyle=':', lw=1, alpha=0.7)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# ── 패널 ③: 공원 유형별 평균 냉각 강도 ─────────────────────────
ax = axes[2]
type_mean = (df_ml.groupby('park_type')['cooling_300m']
               .agg(['mean', 'count'])
               .sort_values('mean', ascending=True))
bar_colors = ['#EF5350' if m < 0 else '#66BB6A' for m in type_mean['mean']]
bars3 = ax.barh(type_mean.index, type_mean['mean'],
                color=bar_colors, edgecolor='white', height=0.6, alpha=0.88)

# x축 범위: 최솟값에서 왼쪽으로 충분한 여백 확보
x_min = type_mean['mean'].min()
x_max = type_mean['mean'].max()
ax.set_xlim(x_min - abs(x_min) * 0.6, x_max + abs(x_max) * 0.5 + 0.5)

for bar, (_, row) in zip(bars3, type_mean.iterrows()):
    x_pos = bar.get_width()
    if x_pos < 0:
        # 음수 막대: 텍스트를 막대 오른쪽 끝(0 방향)에 배치
        ax.text(x_pos + 0.05, bar.get_y() + bar.get_height()/2,
                f'{x_pos:.2f}℃ (n={int(row["count"])})',
                va='center', ha='left', fontsize=8.5, color='white', fontweight='bold')
    else:
        # 양수 막대: 텍스트를 막대 오른쪽 바깥에 배치
        ax.text(x_pos + 0.05, bar.get_y() + bar.get_height()/2,
                f'+{x_pos:.2f}℃ (n={int(row["count"])})',
                va='center', ha='left', fontsize=8.5)

ax.axvline(x=0, color='black', linestyle='--', lw=1.5, alpha=0.7)
ax.set_title('공원 유형별 평균 냉각 강도 (300m)\n빨강=열섬 완화 실패 / 초록=완화 성공',
             fontsize=11, fontweight='bold')
ax.set_xlabel('평균 냉각 강도 (℃)\n양수: 공원이 주변보다 시원', fontsize=10)
ax.tick_params(axis='y', labelsize=9)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('ml_analysis_result.png', dpi=300, bbox_inches='tight')
print("  [OK] ml_analysis_result.png 저장 완료")

print("\n" + "=" * 55)
print("  분석 완료")
print("=" * 55)
