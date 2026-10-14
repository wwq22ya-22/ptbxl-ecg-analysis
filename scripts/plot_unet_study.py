"""Paired-seed validation plot; no significance or test claims."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path('D:/MyProject/college/ecg1')
OUT = ROOT/'outputs/unet_teacher_20261010'
summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
audit = json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
assert audit['passed'] and not audit['partial'] and summary['fold10_evaluation'] is None
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
fig, axes=plt.subplots(1,2,figsize=(10.2,4.5),sharey=True)
colors=['#2469a0','#cb7a24','#36815b']
for ax, pop, title in zip(axes,['filtered','unfiltered'],['筛选后：17084训练 / 2146验证','未筛选：17418训练 / 2183验证']):
    for seed,color in zip([42,43,44],colors):
        values=[next(r for r in summary['runs'] if r['training_cohort']==pop and r['model']==model and r['seed']==seed)['validation'][pop]['macro_f1']*100
                for model in ['unet_encoder1d','unet1d']]
        ax.plot([0,1],values,color=color,marker='o',lw=1.5,label=f'seed{seed}')
        for x,v in enumerate(values):ax.annotate(f'{v:.2f}',(x,v),xytext=(4,5),textcoords='offset points',fontsize=8,color=color)
    means=[summary['three_seed_statistics'][pop][m][pop]['macro_f1']['mean']*100 for m in ['unet_encoder1d','unet1d']]
    ax.scatter([0,1],means,marker='D',s=45,color='#222222',zorder=5,label='三种子均值')
    ax.set_xticks([0,1],['相同编码器对照','1D U-Net']);ax.set_xlim(-.3,1.3)
    ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('折9宏F1（%）')
fig.suptitle('1D U-Net分类：原生人群的配对种子验证结果',fontsize=14)
fig.text(.5,.02,'分别按各自折9细阈值计分；用于选择模型和阈值，未评估折10。对照未匹配参数量。',ha='center',fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,fontsize=9,loc='lower center',bbox_to_anchor=(.5,.07))
fig.tight_layout(rect=[0,.18,1,.94]);fig.savefig(OUT/'paired_validation_f1.png',dpi=180);plt.close(fig)
print('Saved paired_validation_f1.png')
