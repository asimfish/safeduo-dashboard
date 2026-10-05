"""Original initial qualification, new seeds; never read candidate policy outcomes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'safety_risk_strata_20261004'))
import risk_bank
risk_bank.SEEDS=[331047829,389116237,451902773]
if __name__=='__main__':risk_bank.main()
