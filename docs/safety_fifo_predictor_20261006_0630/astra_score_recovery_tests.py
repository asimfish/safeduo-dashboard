"""Recovery scheduling tests; synthetic data only, no raw outcome imports."""
import copy,io,json,unittest
from concurrent.futures import Future
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
import astra_final_score as s
import astra_score_recovery as r
import astra_score_tests as fixture

def future(value):
    f=Future();f.set_result(value);return f

class RecoveryContract(unittest.TestCase):
    def args(self):return ({'output_root':'/synthetic'}, {'id':'one'}, {'id':'one','status':'complete'})
    def route(self,ledger=None,value=None):
        ledger=s.Ledger() if ledger is None else ledger;args=self.args();value=({'cell':True},{'/synthetic/a':'digest'},None) if value is None else value
        return ledger,r.routed_cell(s,ledger,{('/synthetic','one'):args},{('/synthetic','one'):future(value)}),args,value
    def test_return_identity_and_hash_merge(self):
        ledger,route,args,value=self.route();self.assertIs(route(ledger,*args),value[0]);self.assertEqual(ledger.hashes,value[1])
    def test_wrong_ledger_or_registered_arguments_rejected(self):
        ledger,route,args,_=self.route()
        for which in ['ledger','path','job','record']:
            a=copy.deepcopy(args);actual=ledger
            if which=='ledger':actual=s.Ledger()
            elif which=='path':a[0]['output_root']='/wrong'
            elif which=='job':a[1]['unknown']=True
            else:a[2]['status']='failed'
            with self.subTest(which=which),self.assertRaises(ValueError):route(actual,*a)
    def test_shared_hash_conflict_rejected(self):
        ledger=s.Ledger();ledger.hashes['/synthetic/a']='changed'
        ledger,route,args,_=self.route(ledger)
        with self.assertRaisesRegex(ValueError,'input changed'):route(ledger,*args)
    def test_worker_exception_preserves_partial_input_hashes(self):
        e=ValueError('invalid raw condition');ledger,route,args,_=self.route(value=(None,{'/synthetic/a':'digest'},e))
        with self.assertRaisesRegex(ValueError,'invalid raw'):route(ledger,*args)
        self.assertEqual(ledger.hashes,{'/synthetic/a':'digest'})
    def test_future_exception_propagates(self):
        ledger=s.Ledger();args=self.args();f=Future();f.set_exception(RuntimeError('worker interrupted'))
        route=r.routed_cell(s,ledger,{('/synthetic','one'):args},{('/synthetic','one'):f})
        with self.assertRaisesRegex(RuntimeError,'worker interrupted'):route(ledger,*args)
    def test_unregistered_source_rejected_before_lock_or_outputs(self):
        reg={'status':'REGISTERED_RECOVERY_BEFORE_ANY_RECOVERY_OUTCOMES','workers':4,'sources':{'/fake/source':'expected'}}
        with mock.patch.object(Path,'read_text',return_value=json.dumps(reg)),mock.patch.object(r,'sha',return_value='changed'),mock.patch.object(Path,'open') as op:
            with self.assertRaises(AssertionError):r.main()
            op.assert_not_called()
    def test_original_192case_768window_aggregation_identical_after_routing(self):
        plans=[];design={'rows':[]};entries={};cells={};campaigns={}
        for b in range(3):
            p={'output_root':f'/synthetic/{b}','cwd':'/synthetic','source_sha256':{},'research_source_sha256':{},'checkpoint_path':'/synthetic/actor','checkpoint_sha256':'actor','jobs':[]}
            design['rows'].append({'initial_seed':b,'command_seed':b+10})
            records=[]
            for i,m in enumerate(s.MODES):
                j={'id':f'{m}_{b+10}','env':{'SAFEDUO_JOINT_MODE':m}};p['jobs'].append(j);records.append({'id':j['id'],'status':'complete'})
                w=[copy.deepcopy(fixture.NativeScoreTests.window((e+i+b)%4==0)) for e in range(64)]
                for e,v in enumerate(w):v['env']=e
                cells[j['id']]=dict(id=j['id'],mode=m,status='VERIFIED',windows=w,counts=s.aggregate(w))
            plans.append(p);campaigns[str(Path(p['output_root'])/'campaign.json')]={'plan':p,'status':'complete','jobs':records}
            for j,record in zip(p['jobs'],records):entries[p['output_root'],j['id']]=(p,j,record)
        class L(s.Ledger):
            def bind(self,path,expected=None):self._remember(Path(path),expected or 'digest',expected);return expected or 'digest'
            def json(self,path,expected=None):return campaigns.get(str(path),{})
            def recheck(self):return []
        def serial(ledger,plan,job,record):
            ledger._remember(Path('/synthetic')/job['id'],'digest',None)
            return copy.deepcopy(cells[job['id']])
        outputs=[]
        with mock.patch.object(s,'readiness',return_value={'all_terminal':True}),mock.patch.object(s,'now',return_value='fixed'),mock.patch.object(s,'write_owned',side_effect=lambda n,v:outputs.append((n,copy.deepcopy(v)))),redirect_stdout(io.StringIO()):
            with mock.patch.object(s,'score_cell',side_effect=serial):s.run_score(design,plans,L())
            reference=next(v for n,v in outputs if n=='ASTRA_FINAL_SCORE.json');outputs.clear()
            ledger=L();futures={key:future((copy.deepcopy(cells[arg[1]['id']]),{str(Path('/synthetic')/arg[1]['id']):'digest'},None)) for key,arg in reversed(list(entries.items()))}
            with mock.patch.object(s,'score_cell',r.routed_cell(s,ledger,entries,futures)):s.run_score(design,plans,ledger)
            actual=next(v for n,v in outputs if n=='ASTRA_FINAL_SCORE.json')
        self.assertEqual(actual,reference);self.assertEqual(actual['status'],'PASS_COMPLETE_INDEPENDENT_NUMERIC');self.assertEqual(len(actual['cases']),192)
if __name__=='__main__':unittest.main(verbosity=2)
