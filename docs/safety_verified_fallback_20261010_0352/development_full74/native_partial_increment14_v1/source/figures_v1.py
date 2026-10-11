"""Plot already-closed 2x32 CSV observations; no native array reconstruction."""
def figures(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    keyed={(int(x['micro']),int(x['lane'])):x for x in rows}
    def values(m,k):return np.array([float(keyed[m,i][k]) for i in range(32)])
    xs=np.arange(32);colors=['#287b8e','#a54a30']
    f,ax=plt.subplots(3,1,figsize=(11,8.4),sharex=True,layout='constrained')
    for m in (0,1):
        ax[0].plot(xs,1000*values(m,'raw9021_min_m'),'o-',ms=3,lw=1,color=colors[m],label='Initial (micro0)' if m==0 else 'First recorded step (micro1)')
        ax[1].plot(xs,values(m,'qd_over_vmax_ratio'),'o-',ms=3,lw=1,color=colors[m])
    ax[0].axhline(0,color='#333',lw=1,ls='--');ax[0].set_ylabel('Minimum sphere-proxy gap (mm)');ax[0].legend(loc='lower left',ncol=2)
    ax[1].axhline(1,color='#333',lw=1,ls='--');ax[1].set_ylabel('Maximum |qd| / native vmax')
    mask=np.array([[int(keyed[1,i][k] in ('True','true','1')) for i in range(32)] for k in ['hard74','velocity74','soft26','raw9021','point_scalar82']])
    from matplotlib.colors import ListedColormap
    ax[2].imshow(mask,aspect='auto',interpolation='nearest',extent=[-.5,31.5,4.5,-.5],cmap=ListedColormap(['#edf0ef','#a54a30']),vmin=0,vmax=1)
    ax[2].set_yticks(range(5),['Hard74 (18)','Velocity74 (22)','Soft26 (3)','Proxy gap (2)','Recorded scalar82 (30)'])
    ax[2].set_xlabel('Related lane index (32 fixed candidates; not new IID trials)')
    ax[2].set_xticks(range(32));ax[2].tick_params(axis='x',labelsize=8)
    for a in ax[:2]:
        a.grid(alpha=.2)
        for i in (17,27):a.axvspan(i-.35,i+.35,color='#ddbb55',alpha=.18)
    f.suptitle('Initial clearance does not establish first-step compliance',fontsize=14)
    f.text(.01,-.025,'Source: closed independent 2x32 CSV. Recorded scalar predicate is not a force-peak certificate; SDK refresh epoch UNKNOWN.',fontsize=9)
    for ext in ('png','svg'):f.savefig(out/('all32_initial_to_micro1.'+ext),dpi=170,bbox_inches='tight')
    plt.close(f)
    f,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,label in [(17,'Lane17 / seed2 / proposal2532'),(27,'Lane27 / seed3 / proposal2913')]:
        c=colors[0] if i==17 else colors[1]
        for a,key,scale in [(ax[0],'raw9021_min_m',1000),(ax[1],'qd_over_vmax_ratio',1)]:
            y=[float(keyed[m,i][key])*scale for m in (0,1)];a.plot([0,1],y,'o-',color=c,label=label)
            for x,v in enumerate(y):a.annotate(f'{v:.3f}',(x,v),xytext=(7,12 if i==17 else -18),textcoords='offset points',fontsize=9)
    ax[0].axhline(0,color='#333',ls='--',lw=1);ax[1].axhline(1,color='#333',ls='--',lw=1)
    ax[0].set_ylabel('Minimum sphere-proxy gap (mm)');ax[1].set_ylabel('Maximum |qd| / native vmax')
    for a in ax:a.set_xticks([0,1],['Initial micro0','Recorded micro1']);a.grid(alpha=.2)
    ax[0].legend(fontsize=8,loc='lower left');f.suptitle('Two recorded first-step negative-gap witnesses',fontsize=13)
    for ext in ('png','svg'):f.savefig(out/('two_lanes_first_step.'+ext),dpi=170,bbox_inches='tight')
    plt.close(f)
