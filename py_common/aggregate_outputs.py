from dataclasses import dataclass
from enum import Enum, auto
import re
import sys
from typing import Iterable
import argparse

class MolscatOutput:
    def __init__(self, filepath):
        with open(filepath, "r") as f:
            text = f.read()
            pattern = re.compile(
                r"EFV SET\s+\d+:\s+POTL SCALING FACTOR\s*=\s*([\d\.E+-]+)\s*;"
                r"\s*MAGNETIC Z FIELD\s*=\s*([\d\.E+-]+)(?:\s|\S)*?"
                r"K-DEPENDENT SCATTERING LENGTHS(?:.|\n)*?"
                r"^\s*\d+\s+\d+\s+\d+\s+[\d\.E+-]+\s+([\d\.E+-]+)\s+([\d\.E+-]+)", 
                re.MULTILINE
            )

            matches = pattern.findall(text)
            self.matches: list[tuple[str, str, str, str]] = matches

        if len(self.matches) == 0:
            raise ValueError("couldn't parse output, make sure IPRINT = 11")

    def print_all(self):
        for match in self.matches:
            scaling_factor, magnetic_z_field, re_a, im_a = match
            print(f"Scaling Factor: {scaling_factor}")
            print(f"Magnetic Z Field: {magnetic_z_field}")
            print(f"RE(A): {re_a}")
            print(f"IM(A): {im_a}")
            print("-" * 40)
    
    def save(self, savefile):
        with open(savefile, "w+") as f:
            f.write("Scaling_factor\tmagnetic_field\tscattering_re\tscattering_im\n")
            f.writelines(map(lambda l: f"{l[0]}\t{l[1]}\t{l[2]}\t{l[3]}\n", self.matches))

    def combine(self, other: Iterable['MolscatOutput'] | 'MolscatOutput'):
        if type(other) is Iterable['MolscatOutput']:
            for f in other:
                self.matches.extend(f.matches)
        elif type(other) is MolscatOutput:
            self.matches.extend(other.matches)
        else:
            raise TypeError(type(other))

class ScanParameter(Enum):
    MagneticField = auto()
    PotScaling = auto()

@dataclass
class ScanResult:
    scan_variable: float

    states: list[tuple[int, float]]

class BoundOutput:
    def __init__(self, filepath: str, scan_param: ScanParameter = ScanParameter.MagneticField):
        name = "POTL SCALING FACTOR" if scan_param == ScanParameter.PotScaling else "MAGNETIC Z FIELD"

        with open(filepath, "r") as f:
            text = f.read()

            efv_pattern = re.compile(rf"EFV SET\s+\d+:\s+{name}\s*=\s*([\d\.E+-]+)", re.MULTILINE)
            state_pattern = re.compile(r"CONVERGED ON STATE NUMBER\s+(\d+)\s+AT\s+ENERGY\s*=\s*([-]?\d+\.\d+(?:E[+-]?\d+)?)\s*(\w+)", re.MULTILINE)

            # Find all EFV sets
            efv_matches = list(efv_pattern.finditer(text))

            parsed_data = []
            unit = None

            for i, efv_match in enumerate(efv_matches):
                scan_variable = float(efv_match.group(1))
                start_index = efv_match.end()  # Where this EFV SET ends

                # Find the next EFV SET to define the boundary
                if i + 1 < len(efv_matches):
                    end_index = efv_matches[i + 1].start()
                else:
                    end_index = len(text)

                # Extract state numbers and energies between start_index and end_index
                efv_text = text[start_index:end_index]
                state_matches = state_pattern.findall(efv_text)

                states = []
                for state_match in state_matches:
                    state_number = int(state_match[0])
                    energy = float(state_match[1])
                    unit = state_match[2]  # Extract unit (should be the same for all)

                    states.append((state_number, energy))

                parsed_data.append(ScanResult(scan_variable, states))

            self.unit = unit
            self.scan_param = scan_param
            self.matches: list[ScanResult] = parsed_data

        if len(self.matches) == 0:
            raise ValueError("couldn't parse output")

    def print_all(self):
        print(f"Scanning {self.scan_param}")
        for match in self.matches:
            print(f"For {match.scan_variable}:")
            for state in match.states:
                print(f"state: {state[0]}, energy: {state[1]} {self.unit}")
    
    def save(self, savefile):
        with open(savefile, "w+") as f:
            f.write(f"{self.scan_param}\tState_number\tEnergy_[{self.unit}]\n")
            for p in self.matches:
                f.writelines(map(lambda l: f"{p.scan_variable}\t{l[0]}\t{l[1]}\n", p.states))

    def combine(self, other: Iterable['BoundOutput'] | 'BoundOutput'):
        if type(other) is Iterable['BoundOutput']:
            for f in other:
                assert f.scan_param == self.scan_param
                assert f.unit == self.unit

                self.matches.extend(f.matches)
        elif type(other) is BoundOutput:
            assert other.scan_param == self.scan_param
            assert other.unit == self.unit

            self.matches.extend(other.matches)
        else:
            raise TypeError(type(other))
        
        self.matches.sort(key=lambda x: x.scan_variable)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bound", action="store_true")
    parser.add_argument("-s", "--scaling", action="store_true")
    parser.add_argument("output", type=str)
    parser.add_argument("save", type=str)

    args = parser.parse_args()
    if args.scaling:
        scan_param = ScanParameter.PotScaling
    else:
        scan_param = ScanParameter.MagneticField

    if args.bound:
        bound = BoundOutput(args.output, scan_param)

        bound.print_all()
        bound.save(args.save)
    else:
        molscat = MolscatOutput(args.output)

        molscat.print_all()
        molscat.save(args.save)
