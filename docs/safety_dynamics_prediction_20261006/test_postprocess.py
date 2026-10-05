"""Audit failures cannot pass into final publication; preserve execution order."""
from pathlib import Path
import json,tempfile,unittest,subprocess
from unittest.mock import patch
import postprocess
class Tests(unittest.TestCase):
    def test_failed_audit_stops_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'retry_completion.json').write_text('{"status":"complete"}')
            with patch.object(postprocess,'HERE',root),patch('sys.argv',['postprocess','--driver-pid','999']),patch.object(postprocess.os,'pidfd_open',side_effect=ProcessLookupError),patch.object(postprocess.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['score'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):postprocess.main()
                self.assertEqual(run.call_count,1);self.assertTrue(run.call_args[0][0][1].endswith('score.py'))
                self.assertEqual(json.loads((root/'postprocess_status.json').read_text())['status'],'auditing_full_stream')
    def test_incomplete_physics_never_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'retry_completion.json').write_text('{"status":"contains_invalid_or_pending"}')
            with patch.object(postprocess,'HERE',root),patch('sys.argv',['postprocess','--driver-pid','999']),patch.object(postprocess.os,'pidfd_open',side_effect=ProcessLookupError),patch.object(postprocess.subprocess,'run') as run:
                with self.assertRaises(AssertionError):postprocess.main()
                run.assert_not_called()
    def test_successful_audit_then_latest_remote_then_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'retry_completion.json').write_text('{"status":"complete"}')
            (root/'delivery_complete.json').write_text('{"status":"PASS_EXACT_COMMIT_CI_PUBLIC_READBACK_BROWSER"}')
            with patch.object(postprocess,'HERE',root),patch('sys.argv',['postprocess','--driver-pid','999']),patch.object(postprocess.os,'pidfd_open',side_effect=ProcessLookupError),patch.object(postprocess.subprocess,'run') as run:
                postprocess.main();calls=[c[0][0] for c in run.call_args_list]
                self.assertEqual(len(calls),4);self.assertTrue(calls[0][1].endswith('score.py'))
                self.assertEqual(calls[1],['git','fetch','origin','main']);self.assertEqual(calls[2],['git','merge','--ff-only','origin/main'])
                self.assertTrue(calls[3][1].endswith('publish.py'))
                self.assertEqual(json.loads((root/'postprocess_status.json').read_text())['status'],'complete')
if __name__=='__main__':unittest.main()
