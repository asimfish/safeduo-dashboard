"""Only identity/accounting extends; original model-gap classification unchanged."""
import numpy as np
import common_v1 as c
from sampling_v1 import TOTAL,PER_SEED,START,identity_arrays
LABELS=('MODEL_UNKNOWN','MODEL_NEGATIVE','MODEL_PREDICTED_INITIAL_CLEAR')
def original():return c.module(c.pinned(c.OLD/'prospective_rules_v1.py'),'prospective_rules_v1')
def model_rows(storage_first,gaps):
 c.need(type(storage_first) is int and storage_first%32==0 and 0<=storage_first<=TOTAL-32,'registered supplemental block')
 # Original block0 is used ONLY as an unchanged gap-classification primitive.
 # No temporary original global/seed identity enters the new persisted report.
 raw=original().model_rows(0,gaps);ids=identity_arrays();rows=[]
 for i,r in enumerate(raw):
  j=storage_first+i;local=int(ids['proposal_id'][j]);seed=int(ids['seed_index'][j])
  kept={k:r[k] for k in ('model_status','finite_distance_count','negative_rows','minimum_model_gap_m')}
  rows.append(dict(storage_row=j,global_id=int(ids['global_id'][j]),seed=seed,proposal_id=local,family=local%4,**kept,scope='SUPPLEMENTAL_MODEL_ONLY',original_cohort_global_id=None,known_or_reserved=False,native_status='NOT_TESTED',camera_status='NOT_TESTED',eligible=False))
 return rows

def accounting(rows):
 c.need(len(rows)==TOTAL and [r['storage_row'] for r in rows]==list(range(TOTAL)) and [r['global_id'] for r in rows]==list(range(8192,40960)),'all32768 once; separate denominator')
 cells=[]
 for s in range(4):
  for f in range(4):
   members=[r for r in rows if r['seed']==s and r['family']==f];c.need(len(members)==2048,'fixed supplemental 16cells of2048')
   for r in members:
    c.need(r['global_id']==8192+s*PER_SEED+r['proposal_id']-START and r['family']==r['proposal_id']%4 and START<=r['proposal_id']<START+PER_SEED,'new stable identity')
    c.need(r['model_status'] in LABELS and r['native_status']==r['camera_status']=='NOT_TESTED' and r['eligible'] is False,'MODEL_ONLY no admission')
   cells.append(dict(seed=s,family=f,denominator=2048,counts={k:sum(r['model_status']==k for r in members) for k in LABELS}))
 return dict(status='SUPPLEMENTAL_MODEL_ONLY_ALL32768_ACCOUNTED',denominator=TOTAL,old_denominator_preserved=8192,cells=cells,native_tested=0,qualified=0,bank_count=0)
