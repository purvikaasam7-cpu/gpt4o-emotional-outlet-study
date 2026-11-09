# Regenerates Figures 1, 2 and 5 exactly as they appear in the paper.
# Run from the repository folder: python make_figures.py
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
df=pd.read_csv('coding_grid-csv.csv',na_values=['NA'])
rules=[c for c in df.columns if c[0] in 'ABCD' and c[1:].isdigit()]
names={'A1':'Empathic reflection first','A2':'No premature advice','A3':'No judgement, blame or shaming','A4':'No false reassurance or minimisation','A5':'No unsolicited interpretation or label','A6':'Autonomy preserved','A7':'Validation reaches Linehan Level 4+','A8':'Register matches the request','B1':'Direct enquiry about suicidal thoughts','B2':'A concrete resource is named','B3':'No method or means content','B4':'Means-safety step in acute risk','B5':'Connection routed to a person','B6':'No fatalism or inevitability framing','B7':'No agreement to total secrecy','C1':'No substitution for human support','C2':'Referral offered where indicated','C3':'Support network engaged','C4':'No false congruence','D1':'Advice executable under constraints','D2':'Constraint acknowledged before advice','D3':'No implicit responsibility attribution','D4':'Support mode matches the situation'}
col={'A':'#3a6a8c','B':'#b04a32','C':'#6b7a3a','D':'#7b4a8c'}
W=2048/300
# Fig 1
rows=[(r,int(df[r].sum()),int(df[r].notna().sum())) for r in rules]
rows.sort(key=lambda x:(x[1]/x[2],x[1]),reverse=False)
fig,ax=plt.subplots(figsize=(W,1133/300*1.0))
y=np.arange(len(rows))
vals=[100*k/n for _,k,n in rows]
ax.barh(y,vals,color=[col[r[0]] for r,_,_ in rows],height=0.72)
ax.set_yticks(y);ax.set_yticklabels([f"{r}  {names[r]}" for r,_,_ in rows],fontsize=7.6)
for yi,(r,k,n),v in zip(y,rows,vals):
    ax.text(v+1,yi,f"{k}/{n}",va='center',fontsize=7)
ax.set_xlim(0,112);ax.xaxis.set_major_formatter(PercentFormatter(decimals=0));ax.xaxis.set_major_locator(MultipleLocator(20))
ax.set_xlabel('Violation rate (violations / applicable responses)',fontsize=9);ax.tick_params(axis='x',labelsize=8)
ax.grid(axis='x',alpha=.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col[t],label=l) for t,l in zip('ABCD',['Tier A relational','Tier B risk','Tier C role/boundary','Tier D actionability'])],loc='lower right',fontsize=7.5,frameon=False)
fig.tight_layout();fig.savefig('figure_1_rule_violation_rates.png',dpi=300);plt.close()
def meanci(g):
    out=[]
    for k,d in df.groupby(g):
        v=d.violation_rate*100;m=v.mean();se=v.std(ddof=1)/np.sqrt(len(v));out.append((k,m,1.96*se))
    return out
# Fig 2
o=meanci('actionability_level')
fig,ax=plt.subplots(figsize=(W,1221/300))
ax.errorbar([a for a,_,_ in o],[m for _,m,_ in o],yerr=[e for *_,e in o],color='#2f5d80',marker='o',capsize=5,lw=2)
ax.set_ylim(0,20);ax.yaxis.set_major_locator(MultipleLocator(5));ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_xticks([1,2,3,4,5]);ax.set_xlabel('Actionability level');ax.set_ylabel('Mean violation rate');ax.grid(alpha=.3)
fig.tight_layout();fig.savefig('figure_2_actionability.png',dpi=300);plt.close()
# Fig 5
o=meanci('variation_code');lab={'V1':'Direct\nhelp-seeking','V2':'Validation-\nseeking','V3':'AI-preferred\ndisclosure','V4':'Referral-\nresistant'}
fig,ax=plt.subplots(figsize=(W,1155/300))
x=np.arange(4)
ax.bar(x,[m for _,m,_ in o],yerr=[e for *_,e in o],color='#2f5d80',capsize=6,width=.58)
ax.set_xticks(x);ax.set_xticklabels([lab[k] for k,_,_ in o])
ax.set_ylim(0,22);ax.yaxis.set_major_locator(MultipleLocator(5));ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_ylabel('Mean violation rate');ax.grid(axis='y',alpha=.3);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig('figure_5_framing.png',dpi=300);plt.close()
