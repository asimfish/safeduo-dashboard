"""Same preregistered sampler, new seeds; no random-policy outcome access."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'safety_risk_strata_20261004'))
import risk_bank
risk_bank.SEEDS=[152684921,198470327,237901613]
if __name__=='__main__':risk_bank.main()
