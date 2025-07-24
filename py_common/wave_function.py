from dataclasses import dataclass
import json
import re
import numpy as np
import numpy.typing as npt
from array import array
from .units import GHZ

def parse_wavefunction_file(path, basis_size, max_coeff=None):
    header_re = re.compile(r"WAVEFUNCTION FOR STATE\s+(\d+)\s+AT ENERGY\s+([-+]?\d*\.\d+(?:[eE][+-]?\d+)?)")

    data = {}
    current_state = None
    energy = None

    r_arr = array('d')
    coeffs_arr = array('d')
    buf = array('d')
    row_length = 1 + basis_size

    coeff_count = (max_coeff + 1) if max_coeff is not None else basis_size

    with open(path, 'r') as f:
        for line in f:
            m = header_re.search(line)
            if m:
                if current_state is not None:
                    npts = len(r_arr)
                    data[current_state] = {
                        'energy': energy,
                        'r': np.frombuffer(r_arr, dtype=float).copy(),
                        'coeffs': np.frombuffer(coeffs_arr, dtype=float).reshape(npts, coeff_count).copy()
                    }

                current_state = int(m.group(1))
                energy = float(m.group(2))
                r_arr = array('d')
                coeffs_arr = array('d')
                buf = array('d')
                continue

            if not line or line[0] in ('#', '\n', '\r'):
                continue

            nums = np.fromstring(line, dtype=float, sep=' ')
            if nums.size == 0:
                continue
            buf.extend(nums.tolist())

            while len(buf) >= row_length:
                r_arr.append(buf[0])
                start = 1
                end = 1 + coeff_count
                coeffs_arr.extend(buf[start:end])
                del buf[:row_length]

    if current_state is not None:
        npts = len(r_arr)
        data[current_state] = {
            'energy': energy,
            'r': np.frombuffer(r_arr, dtype=float).copy(),
            'coeffs': np.frombuffer(coeffs_arr, dtype=float).reshape(npts, coeff_count).copy()
        }

    return data

@dataclass
class WaveFunction:
    energy: float
    distances: npt.NDArray
    values: npt.NDArray

def wavefunction_json(path: str, take: int) -> dict[int, WaveFunction]:
    with open(path, "r") as file:
        data = json.load(file)

    waves = {}
    for (n, e, w) in zip(data["bounds"]["nodes"], data["bounds"]["energies"], data["waves"]):
        wave = WaveFunction(e / GHZ, np.array(w["distances"]), np.array(w["values"])[:, :take])
        waves[n] = wave

    return waves

if __name__ == '__main__':
    parsed = parse_wavefunction_file('data/wave_function_singlet_175.output', 176, max_coeff = 5)

    for st, info in parsed.items():
        print(f"State {st}: energy={info['energy']}, points={len(info['r'])}, coeffs_shape={info['coeffs'].shape}")
        print(info['r'][0:10])
        print(info['coeffs'][0:10, :])
