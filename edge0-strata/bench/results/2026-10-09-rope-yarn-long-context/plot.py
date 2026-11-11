"""Regenerate the figure from results.json: python plot.py (requires matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'results.json').read_text())
rows=data['runs']
BG='#f5f7fb';INK='#14243a';MUTED='#516379';GRID='#dbe2ec'
colors=['#426b9f' if r['rope']=='none' else '#008879' for r in rows]
labels=[('Ordinary RoPE' if r['rope']=='none' else 'YaRN 4x')+'  /  '+('512K' if r['input_tokens']==524288 else '1M') for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':INK})
fig=plt.figure(figsize=(14,9),facecolor=BG)
fig.text(.055,.945,'Ordinary RoPE vs YaRN at long context',fontsize=25,weight='bold')
fig.text(.055,.900,'Existing Strata on untouched main  |  ISTA IQ3_XXS  |  FP16 KV  |  RTX PRO 6000 Blackwell',fontsize=12,color=MUTED)
left=fig.add_axes([.24,.52,.24,.29],facecolor=BG)
right=fig.add_axes([.64,.52,.30,.29],facecolor=BG)
for ax in (left,right):
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_ylim(2.6,-.6);ax.tick_params(axis='both',length=0)
left.set_title('Correct code associations',loc='left',pad=24,fontsize=14,weight='bold')
left.set_yticks(range(3),labels);left.tick_params(axis='y',pad=15);left.set_xticks([]);left.set_xlim(0,3.65)
for i,r in enumerate(rows):
    correct=r['correct_code_pairs']
    for j in range(3):
        left.add_patch(FancyBboxPatch((j+.04,i-.24),.87,.48,boxstyle='round,pad=0.015,rounding_size=.05',
                                     linewidth=0,facecolor=colors[i] if j<correct else GRID))
    left.text(3.12,i,f'{correct}/3',va='center',weight='bold',fontsize=15)
right.set_title('Fresh prefill time',loc='left',pad=24,fontsize=14,weight='bold')
right.set_yticks([]);right.set_xlim(0,max(r['prefill_s'] for r in rows)*1.23)
right.set_xlabel('Seconds · lower is faster',labelpad=12)
right.xaxis.grid(True,color=GRID);right.set_axisbelow(True)
for i,r in enumerate(rows):
    right.barh(i,r['prefill_s'],height=.48,color=colors[i])
    right.text(r['prefill_s']+5,i,f'{r["prefill_s"]:.1f}s',va='center',weight='bold')
fig.text(.055,.425,'The exact answers',fontsize=16,weight='bold')
fig.text(.055,.382,'Expected',color=MUTED,fontsize=11)
fig.text(.25,.382,data['expected'].replace('|','  |  '),family='DejaVu Sans Mono',fontsize=12,weight='bold')
for i,r in enumerate(rows):
    y=.329-i*.047
    fig.text(.055,y,labels[i],fontsize=11,color=colors[i])
    fig.text(.25,y,r['answer'].replace('|','  |  '),family='DejaVu Sans Mono',fontsize=12)
fig.text(.055,.125,'One prompt per condition; three facts amid repeated filler. Zero cached tokens reused.',fontsize=11,color=MUTED)
fig.text(.055,.090,'512K success does not establish a safe cutoff. Fresh YaRN at 1M does not validate cache migration.',fontsize=11,color=MUTED)
fig.text(.055,.055,'Actual inputs: 524,288 / 1,048,576 tokens. Allocation: 1,048,832 (includes output headroom).',fontsize=10,color=MUTED)
fig.savefig(HERE/'overview.png',dpi=180,facecolor=BG)
fig.savefig(HERE/'overview.svg',facecolor=BG)
plt.close(fig)
