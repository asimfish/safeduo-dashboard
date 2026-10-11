"""Complete closed55/60 traces and native60 weak contact, original gates fixed."""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
P=H/'loaded_rate_delivery_v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt=P/'FIGURE_CUSTODY_V1.json';assert not receipt.exists()
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True,layout='constrained')
    sources=[Path(__file__)];summaries=[];last=None
    for version,color in [(55,'#28658b'),(60,'#b34531')]:
        bundle=(H/'hand_target_delivery_v1' if version==55 else P)/'bundle_v1'/f'native_v{version}'
        file=bundle/f'SELECTED_EVERY_STEP_NATIVE_FIELDS_V{2 if version==55 else 4}.npz'
        paramsfile=bundle/'native_parameters.npz';regfile=bundle/'REGISTRATION_V1.json';auditfile=bundle/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json'
        audit=read(auditfile);assets=read(read(regfile)['coupled_assets'])['assets']
        with np.load(file) as z,np.load(paramsfile) as params:
            times=z['state_time_s'];qe=np.zeros(len(times));ve=np.zeros(len(times));jump=np.zeros(len(times))
            for arm,asset in assets.items():
                names=params[arm+'_joint_names'].tolist();ids=[names.index(n) for n in asset['hand_joint_names']]
                target=z[arm+'_actual_full_position_target'][:,0,ids];jump[1:]=np.maximum(jump[1:],abs(np.diff(target,axis=0)).max(1))
                q,v=z[arm+'_native_q'][:,0],z[arm+'_native_qd'][:,0]
                urdf=Path(asset['original_URDF']);sources.append(urdf)
                for j in ET.parse(urdf).findall('joint'):
                    m=j.find('mimic')
                    if m is None or j.get('name') not in asset['hand_joint_names']:continue
                    si,mi=names.index(j.get('name')),names.index(m.get('joint'));slope=float(m.get('multiplier','1'));bias=float(m.get('offset','0'))
                    qe=np.maximum(qe,abs(q[:,si]-slope*q[:,mi]-bias));ve=np.maximum(ve,abs(v[:,si]-slope*v[:,mi]))
            assert float(qe.max())==audit['maximum_q_residual_rad'] and float(ve.max())==audit['maximum_qd_residual_rad_s']
            force=z['four_hand_normal_N'].max(1);active=z['phase']=='CLOSE';label=f'Native V{version}'
            for ax,data in zip(axes,[jump,force,qe,ve]):ax.plot(times,data,color=color,label=label,linewidth=1.)
            summaries.append(dict(version=version,physics_events=len(times),duration_s=float(times[-1]),maximum_force_N=float(force.max()),q_error_rad=float(qe.max()),qd_error_rad_s=float(ve.max()),model_pass=audit['model_window_pass']))
            if version==60:last=dict(event=z['physics_event'][-16:].copy(),force=force[-16:].copy(),qe=qe[-16:].copy(),ve=ve[-16:].copy())
        sources.extend([file,paramsfile,regfile,auditfile])
    labels=['Max hand target increment / step (rad)','Max hand normal force (N)','Original position residual (rad)','Original velocity residual (rad/s)']
    for i,ax in enumerate(axes):
        ax.set_ylabel(labels[i]);ax.grid(alpha=.25);ax.set_yscale('symlog',linthresh=1e-6)
        if i:ax.axhline(30 if i==1 else 1e-4,color='#a83432',ls='--',label='Original limit')
        ax.legend(loc='upper left',fontsize=9)
    axes[-1].set_xlabel('Actual native simulation time (s); different stop times')
    fig.suptitle('One known development input: submitted target rate 0.6 vs 0.02 rad/s\nV60 CLOSE-only early model abort; actor OFF / System 0 OFF; neither is a successful task')
    overview=P/'native_loaded_rate_complete.png';fig.savefig(overview,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    for ax,key,title in zip(axes,['force','qe','ve'],['Weak native contact (N)','Original position residual (rad)','Original velocity residual (rad/s)']):
        ax.plot(last['event'],last[key],'o-',color='#b34531');ax.set_ylabel(title);ax.grid(alpha=.25)
        if key!='force':ax.axhline(1e-4,color='#a83432',ls='--',label='Original 1e-4 limit');ax.legend()
    axes[-1].set_xlabel('Native V60 physics event');fig.suptitle('V60 preserved failure: only weak contact, original velocity gate still fails\nFour-hand contact >=0.1 N not attained; no stable grasp or lift')
    zoom=P/'native_loaded_rate_first_contact.png';fig.savefig(zoom,dpi=170);plt.close(fig)
    receipt.write_text(json.dumps(dict(status='PASS_FIGURES_BOUND_TO_COMPLETE_NATIVE_RECORDS_AND_ORIGINAL_URDF_PEAKS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summaries=summaries,
        images=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in [overview,zoom]],
        source_sha256={str(p):sha(p) for p in sources},no_new_random_trials=True,formal_holdout_count=0,fullSystem0_accepted=False),indent=2)+'\n')
    print('LOADED_RATE_COMPLETE_AND_WEAK_CONTACT_FIGURES',summaries,flush=True)


if __name__=='__main__':main()
