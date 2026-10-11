"""Plot closed native54/55 traces, preserving all original failed gates."""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
P = H/'contact_predictor_delivery_v1'
ARMS = ('F_L','F_R','U_L','U_R')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = P/'native_contact_predictor_counterfactual.png'
    receipt = P/'FIGURE_CUSTODY_V1.json'
    assert not out.exists() and not receipt.exists()
    fig, axes = plt.subplots(4,1,figsize=(11.6,12),sharex=True,layout='constrained')
    sources = [Path(__file__)]
    summaries = []
    for version,color in [(55,'#26628b'),(57,'#27795e')]:
        bundle = (H/'hand_target_delivery_v1' if version == 55 else P)/'bundle_v1'/f'native_v{version}'
        file = bundle/('SELECTED_EVERY_STEP_NATIVE_FIELDS_V2.npz' if version == 55 else 'SELECTED_EVERY_STEP_NATIVE_FIELDS_V3.npz')
        native = bundle/'native_parameters.npz'
        audit = read(bundle/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
        reg = read(bundle/'REGISTRATION_V1.json')
        assets = read(reg['coupled_assets'])['assets']
        with np.load(file,allow_pickle=False) as d,np.load(native,allow_pickle=False) as params:
            times = d['state_time_s']
            qerr = np.zeros(len(times));verr=np.zeros(len(times));jump=np.zeros(len(times))
            for arm,asset in assets.items():
                names = params[arm+'_joint_names'].tolist()
                handids = [names.index(name) for name in asset['hand_joint_names']]
                targets = d[arm+'_actual_full_position_target'][:,0,handids]
                jump[1:] = np.maximum(jump[1:],abs(np.diff(targets,axis=0)).max(1))
                q,v = d[arm+'_native_q'][:,0],d[arm+'_native_qd'][:,0]
                urdf = Path(asset['original_URDF']);sources.append(urdf)
                for joint in ET.parse(urdf).findall('joint'):
                    mimic = joint.find('mimic')
                    if mimic is None or joint.get('name') not in asset['hand_joint_names']:
                        continue
                    slave,master=names.index(joint.get('name')),names.index(mimic.get('joint'))
                    scale,bias=float(mimic.get('multiplier','1')),float(mimic.get('offset','0'))
                    qerr=np.maximum(qerr,abs(q[:,slave]-scale*q[:,master]-bias))
                    verr=np.maximum(verr,abs(v[:,slave]-scale*v[:,master]))
            assert float(qerr.max()) == audit['maximum_q_residual_rad']
            assert float(verr.max()) == audit['maximum_qd_residual_rad_s']
            active = d['phase']=='CLOSE'
            label=f'Native V{version}'
            axes[0].plot(times[active],jump[active],color=color,label=label,linewidth=1.)
            axes[1].plot(times[active],d['four_hand_normal_N'][active].max(1),color=color,label=label,linewidth=1.)
            axes[2].plot(times,qerr,color=color,label=label,linewidth=1.)
            axes[3].plot(times,verr,color=color,label=label,linewidth=1.)
            summaries.append(dict(version=version,physics_events=len(times),close_microsteps=int(active.sum()),
                maximum_close_target_jump_rad=float(jump[active].max()),
                maximum_close_hand_normal_N=float(d['four_hand_normal_N'][active].max()),
                original_model_position_error_rad=float(qerr.max()),original_model_velocity_error_rad_s=float(verr.max()),
                original_model_window_pass=audit['model_window_pass']))
        sources.extend([file,native,bundle/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json',bundle/'REGISTRATION_V1.json'])
    axes[0].set_ylabel('Max hand-target increment\nper physical step (rad)')
    axes[1].set_ylabel('Max of four hand normal\nforce sums (N)')
    axes[1].axhline(30.,color='#a83432',linestyle='--',label='Original 30 N limit')
    for i,label in [(2,'Max original-URDF\nposition residual (rad)'),(3,'Max original-URDF\nvelocity residual (rad/s)')]:
        axes[i].set_ylabel(label);axes[i].axhline(1e-4,color='#a83432',linestyle='--',label='Original 1e-4 tolerance')
        axes[i].set_yscale('symlog',linthresh=1e-6)
    for ax in axes:
        ax.grid(alpha=.25);ax.legend(loc='upper left',fontsize=9)
    axes[-1].set_xlabel('Actual native simulation time (s)')
    fig.suptitle('One known development input: contact-disturbance predictor counterfactual\nOriginal hardware and gates; actor OFF, System 0 OFF; no formal acceptance',fontsize=14)
    fig.savefig(out,dpi=180);plt.close(fig)
    receipt.write_text(json.dumps(dict(status='PASS_CLOSED_NATIVE_TRACE_FIGURE_WITH_EXACT_ORIGINAL_URDF_PEAKS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summaries=summaries,
        image_sha256=sha(out),image_bytes=out.stat().st_size,native_trial_count_added=0,
        independent_random_cases=0,formal_holdout_count=0,fullSystem0_accepted=False,
        source_sha256={str(file):sha(file) for file in sources}),indent=2)+'\n')
    print('CONTACT_PREDICTOR_COUNTERFACTUAL_FIGURE_ORIGINAL_GATE_PEAKS_BOUND',summaries,flush=True)


if __name__ == '__main__':
    main()
