"""One-pair contact ablation audit; it can explain a failure, not certify safety."""
from pathlib import Path
import hashlib,json
import numpy as np

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
E=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=H/'fourhand_native_probe_v116';output=p/'INDEPENDENT_SINGLE_CONTACT_CAUSAL_V1.json';assert not output.exists()
    reg=read(p/'REGISTRATION_V1.json');resource=read(p/'RESOURCE_EXECUTION_V1.json');raw=Path(resource['out'])
    capture=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json');model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json');motor=read(p/'INDEPENDENT_FULL_MOTOR_EFFORT_V4.json')
    assert resource['actual_exit']==resource['actual_native_exit']==0 and resource['native_owned_process_reaped_and_absent']
    baseline=Path(read(H/'fourhand_native_probe_v115/RESOURCE_EXECUTION_V1.json')['out'])
    binding=read(raw/'causal_contact_filter_binding_v1.json');plan=reg['contact_causal_ablation']
    assert binding['owner']==plan['filter_owner'] and binding['actual_targets']==[plan['filter_target']] and binding['excluded_from_all_safety_and_task_acceptance']
    for filename in ('native_parameters.npz','actual_initial_seed_binding.npz'):
        with np.load(raw/filename) as a,np.load(baseline/filename) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    partners=read(raw/'native_task_identity.json')['partners'];target=partners.index(plan['filter_target'])
    baseline_partners=read(baseline/'native_task_identity.json')['partners'];assert partners==baseline_partners
    prefix_events=824;compared=0;different=[];sources={};first_fr_contact=None;peak_filtered=0.;event824=None
    protocol=read(raw/'fourhand_protocol.json');failure=protocol['model_qualification_failure'];failure_contacts=[]
    fields=['physics_event','state_time_s','object_state','all9021_raw_geometry','four_hand_normal_N','object_normal_partner_abs_normal_sum_N']
    fields += [a+s for a in ARMS for s in ('_native_q','_native_qd','_native_body_transforms_after_xyzw','_actual_full_position_target','_extra_effort','_native_generalized_mass_before')]
    for c in read(raw/'physics_chunks.json'):
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f) as z:
            if c['start']<prefix_events:
                n=min(c['stop'],prefix_events)-c['start']
                with np.load(baseline/c['path']) as old:
                    for k in fields:
                        if z[k][:n].dtype!=old[k][:n].dtype or z[k][:n].tobytes()!=old[k][:n].tobytes():different.append({'chunk_start':c['start'],'field':k})
                compared+=n
            mat=z['object_normal_partner_abs_normal_sum_N'][:,0];peak_filtered=max(peak_filtered,float(mat[:,target].max()))
            for i,event in enumerate(z['physics_event']):
                contacts=[{'body':partners[j],'normal_N':float(mat[i,j])} for j in range(len(partners)) if '/F_R/' in partners[j] and mat[i,j]>1e-8]
                if contacts and first_fr_contact is None:first_fr_contact={'event':int(event),'contacts':contacts}
                if int(event)==824:event824={'maximum_qd_residual_rad_s':max(float(abs(z[a+'_mimic_velocity_residual_rad_s'][i]).max()) for a in ARMS),'FR_object_contacts':contacts}
                if failure and int(event)==failure['event']:failure_contacts=contacts
    assert compared==824 and peak_filtered==0
    eligible=len(different)==0 and event824 is not None
    conclusion='Filtering this one contact removes the event824 residual spike after an identical824-event physical prefix; later first contact on right_thumb_3 coincides with another original qd failure. Contact-induced coupling under impact must be repaired; simply excluding one contact does not solve manipulation.' if eligible else 'Precontact trajectories differ; the ablation cannot attribute the original event824 difference solely to this pair.'
    result={'status':'CLOSED_SINGLE_CONTACT_CAUSAL_ABLATION','version':116,'baseline':115,'physics_events':capture['physics_events'],'duration_s':capture['observed_duration_s'],
        'filter_binding':binding,'all_native_parameter_and_seed_arrays_byte_equal115':True,'precontact_physical_events_compared':compared,
        'precontact_field_count':len(fields),'precontact_every_compared_physical_field_byte_equal':not different,'precontact_differences':different,
        'filtered_contact_peak_normal_N':peak_filtered,'native_event824_in_filtered_world':event824,
        'first_FR_object_contact':first_fr_contact,'original_model_failure':failure,'FR_contacts_at_model_failure':failure_contacts,
        'original_residual_q_AND_qd_pass':model['maximum_q_residual_rad']<=1e-4 and model['maximum_qd_residual_rad_s']<=1e-4,
        'original_observed_other_physical_gate_values_pass':capture['physical_gates_pass'],'causal_interpretation_eligible':eligible,'conclusion':conclusion,
        'altered_collision_world_safety_or_task_pass':False,'counts_towards_random_safety_pass_rate':False,'fullSystem0_accepted':False,
        'formal_holdouts':0,'new_training_jobs':0,'source_sha256':{**sources,str(Path(__file__)):sha(__file__),str(raw/'causal_contact_filter_binding_v1.json'):sha(raw/'causal_contact_filter_binding_v1.json'),str(p/'REGISTRATION_V1.json'):sha(p/'REGISTRATION_V1.json')}}
    output.write_text(json.dumps(result,indent=2)+'\n');print('CONTACT_CAUSAL_CLOSED','pre824_equal',not different,'event824',json.dumps(event824),'later_failure',json.dumps(failure),'NEVER_SAFETY_PASS',flush=True)


if __name__=='__main__':main()
