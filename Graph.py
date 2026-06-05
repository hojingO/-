import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 깨짐 방지 폰트 설정 (Windows 기준)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. GEE에서 받아온 데이터 로드
df = pd.read_csv('suncheon_parks_lst_analysis.csv')

# ==========================================================
# [그래프 1] 공원 면적 vs 완화 효과 (100m 및 300m 각각 생성)
# ==========================================================
for target_buffer in [100, 300]:
    # 결측치 제거
    df_clean = df.dropna(subset=['park_lst', f'buffer_{target_buffer}_lst', 'area'])
    
    # 열섬 완화 효과 계산 (주변 온도 - 공원 온도)
    df_clean['cooling_intensity'] = df_clean[f'buffer_{target_buffer}_lst'] - df_clean['park_lst']
    
    plt.figure(figsize=(10, 6))
    # 면적을 로그 스케일로 시각화
    sns.regplot(x=np.log10(df_clean['area']), y=df_clean['cooling_intensity'], 
                scatter_kws={'alpha':0.6, 'color':'teal'}, 
                line_kws={'color':'red', 'lw':2})
    
    plt.title(f'순천시 도시공원 면적에 따른 열 완화 효과 ({target_buffer}m 반경)', fontsize=14, pad=15)
    plt.xlabel('공원 면적 (로그 스케일, $log_{10}(m^2)$)', fontsize=12)
    plt.ylabel('열 완화 효과 (주변 온도 - 공원 온도, ℃)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = f'park_area_vs_cooling_{target_buffer}m.png'
    plt.savefig(filename, dpi=300)
    print(f"그래프 저장 완료: {filename}")
    plt.show()

# ==========================================================
# [그래프 2] 거리별 온도 분포 변화 (전체 거리 연속 비교)
# ==========================================================
df_line = df.dropna(subset=['park_lst', 'buffer_50_lst', 'buffer_100_lst', 'buffer_200_lst', 'buffer_300_lst', 'buffer_500_lst', 'area'])
df_line['size_class'] = df_line['area'].apply(lambda x: '소형 (<10,000㎡)' if x < 10000 else '중대형 (≥10,000㎡)')

distance_cols = ['park_lst', 'buffer_50_lst', 'buffer_100_lst', 'buffer_200_lst', 'buffer_300_lst', 'buffer_500_lst']
melted_df = df_line.melt(id_vars=['size_class'], value_vars=distance_cols, 
                        var_name='Distance', value_name='Temperature')

# 거리 라벨 매핑
dist_map = {
    'park_lst': '공원 내부',
    'buffer_50_lst': '50m',
    'buffer_100_lst': '100m',
    'buffer_200_lst': '200m',
    'buffer_300_lst': '300m',
    'buffer_500_lst': '500m'
}
melted_df['Distance'] = melted_df['Distance'].map(dist_map)

plt.figure(figsize=(12, 6))
sns.lineplot(data=melted_df, x='Distance', y='Temperature', hue='size_class', 
             marker='o', err_style='band', palette={'소형 (<10,000㎡)': 'orange', '중대형 (≥10,000㎡)': 'green'})

plt.title('공원 경계로부터의 거리에 따른 온도 변화 분포 (소형 vs 중대형)', fontsize=14, pad=15)
plt.xlabel('거리', fontsize=12)
plt.ylabel('평균 온도 (℃)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

filename_dist = 'distance_vs_temp.png'
plt.savefig(filename_dist, dpi=300)
print(f"그래프 저장 완료: {filename_dist}")
plt.show()
