import argparse
import glob
from pathlib import Path


from aggregate_outputs import BoundOutput, ScanParameter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filter", type=str)

    args = parser.parse_args()

    files = glob.glob(args.filter)

    for file in files:
        try:
            bound = BoundOutput(file)
        except:
            bound = BoundOutput(file, scan_param=ScanParameter.PotScaling)

        p = Path(file)
        p = p.with_suffix('.dat')

        bound.save(p)
