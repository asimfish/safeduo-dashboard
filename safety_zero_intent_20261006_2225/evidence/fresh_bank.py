from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"safety_risk_strata_20261004"))
import risk_bank
risk_bank.SEEDS=[r["initial_seed"] for r in json.loads((HERE/"RANDOM_EXPERIMENT_DESIGN.json").read_text())["rows"]]
if __name__=="__main__":risk_bank.main()
