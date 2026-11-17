import re, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
TEAL='#0E9F8A'; CORAL='#D93F4F'; INK='#2B2B2B'; GREY='#8A8F94'; AMB='#F59E0B'
rows=[]
for l in open('../../results/rq2_repair_fifteen_seeds.md'):
    m=re.match(r'\| (\d{4}) \| ([\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \|',l)
    if m: rows.append([float(x) for x in m.groups()[2:]])
A=np.array(rows); assert A.shape==(15,4),A.shape
M=None
for l in open('../../results/rq2_repair_fifteen_seeds.md'):
    m=re.match(r'\| mean \| [\d.]+ \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \|',l)
    if m: M=np.array([float(x) for x in m.groups()])
assert M is not None
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1.15,1]})
names=[f'{n}\n({M[j]:+.4f})' for j,n in enumerate(['VGR','NB','GNB','GNB, VGR'])]
rng=np.random.RandomState(0)
for j in range(4):
    ax[0].bar(j,M[j],color=TEAL,alpha=0.35,width=0.6)
    ax[0].scatter(j+rng.uniform(-0.18,0.18,15),A[:,j],s=9,color=INK,zorder=3)
ax[0].axhline(0,color=GREY,lw=0.8)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names)
ax[0].set_ylabel('Recall@20 gain over unrepaired'); ax[0].set_title('(a) 15 seeds, transductive (dots: seeds)',fontsize=8,loc='left')
names2=['VGR','NB','GNB','Oracle\nneighbour blend']
inn=[0.0026,0.0059,0.0078,0.0057]; out=[-0.0003,-0.0006,0.0004,0.0029]
x=np.arange(4); w=0.36
b1=ax[1].bar(x-w/2,inn,w,color=TEAL,label='cold items in training losses (released code)')
b2=ax[1].bar(x+w/2,out,w,color=CORAL,label='cold items left out (warm-only)')
for xs,v in list(zip(x-w/2,inn))+list(zip(x+w/2,out)):
    ax[1].text(xs,v+(0.0004 if v>=0 else -0.0011),f'{v:+.4f}',ha='center',fontsize=6)
ax[1].axhline(0,color=GREY,lw=0.8)
ax[1].set_xticks(x); ax[1].set_xticklabels(names2)
ax[1].set_title('(b) gain depends on the setting (15 seeds)',fontsize=8,loc='left')
ax[1].legend(fontsize=6,frameon=False,loc='upper left')
ax[1].set_ylim(-0.0025,0.0125)
fig.tight_layout()
fig.savefig('fig/repair_gains.pdf'); fig.savefig('fig/repair_gains.png',dpi=200)
print('ok',A.mean(0))
