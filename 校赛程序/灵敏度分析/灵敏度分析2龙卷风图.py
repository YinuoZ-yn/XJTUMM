import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取数据
df = pd.read_excel('第二问指标处理版.xlsx')

# 2. 设置变量和权重
variables = {
    '标准化1（研究生毕业生占比）': {'name': '研究生毕业生占比', 'weight': 0.256619},
    '标准化2（高等学校专任教师数占比）': {'name': '高等学校专任教师数占比', 'weight': 0.452332},
    '标准化3（师生比）': {'name': '师生比', 'weight': 0.018068},
    '标准化4（生均教育经费）': {'name': '生均教育经费', 'weight': 0.272982}
}

# 3. 计算非对称影响值
results = []
for var, meta in variables.items():
    impact = df[var] * meta['weight'] * 40
    pos_impact = impact[impact > 0].mean()
    neg_impact = impact[impact < 0].mean()
    results.append({
        'name': meta['name'],
        'pos': pos_impact if not np.isnan(pos_impact) else 0,
        'neg': neg_impact if not np.isnan(neg_impact) else 0,
        'weight': meta['weight']
    })

# 按最大绝对影响排序
results.sort(key=lambda x: max(abs(x['pos']), abs(x['neg'])), reverse=True)

# 4. 创建不对称坐标轴
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(results))

# 计算非对称轴范围
max_pos = max(r['pos'] for r in results) * 1.3
max_neg = min(r['neg'] for r in results) * 1.3
ax.set_xlim(max_neg, max_pos)  # 关键差异点：左右界限独立设置

# 5. 绘制非对称条形图
for i, r in enumerate(results):
    # 正向影响（右）
    if r['pos'] != 0:
        ax.barh(i, r['pos'], left=0, height=0.6,
                color='#E4B1F3', alpha=0.8, edgecolor='darkgreen')
        ax.text(r['pos'] * 1.02, i, f'+{r["pos"]:.2f}',
                va='center', color='#D888EF', fontweight='bold')

    # 负向影响（左）
    if r['neg'] != 0:
        ax.barh(i, r['neg'], left=0, height=0.6,
                color='#e74c3c', alpha=0.8, edgecolor='darkred')
        ax.text(r['neg'] * 1.02, i, f'{r["neg"]:.2f}',
                va='center', color='darkred', fontweight='bold', ha='right')

# 6. 添加特殊标记
for i, r in enumerate(results):
    ax.text(max_pos * 0.95, i, f"权重:{r['weight'] * 100:.1f}%",
            va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))
    ax.plot([0, 0], [i - 0.3, i + 0.3], color='black', linestyle=':', linewidth=1)

# 7. 美化图形
ax.set_yticks(y_pos)
ax.set_yticklabels([r['name'] for r in results], fontsize=12)
ax.invert_yaxis()
ax.set_title('各标准化变量对得分的敏感度分析', fontsize=14, pad=20)
ax.set_xlabel('对得分的差异化影响', fontsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()