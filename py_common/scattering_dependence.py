from dataclasses import dataclass
from enum import Enum, auto
import json
import numpy as np
import numpy.typing as npt
from .units import ANGS

@dataclass
class ScatteringObserwables:
    entrance: int
    scattering_length: complex
    elastic_cross_section: float
    inelastic_cross_sections: list[float]

@dataclass
class ScatteringDependence:
    parameters: list[float]
    cross_sections: list[ScatteringObserwables]

    @staticmethod
    def parse_json(filename: str) -> 'ScatteringDependence':
        with open(filename, "r") as file:
            data = json.load(file)

        cross_sections = [
            ScatteringObserwables(
                entrance = item['entrance'],
                scattering_length = complex(*item['scattering_length']),
                elastic_cross_section = item['elastic_cross_section'],
                inelastic_cross_sections = item['inelastic_cross_sections']
            )
            for item in data['observables']
        ]

        return ScatteringDependence(
            parameters=data['parameters'],
            cross_sections=cross_sections
        )
    
    def s_lengths(self) -> npt.NDArray[np.complex64]:
        return np.array(list(map(lambda x: x.scattering_length, self.cross_sections)))
    
    def elastic_cross_sections(self) -> npt.NDArray[np.float64]:
        return np.array(list(map(lambda x: x.elastic_cross_section, self.cross_sections)))
    
    def inelastic_cross_sections(self, channel: int) -> npt.NDArray[np.float64]:
        return np.array(list(map(lambda x: x.inelastic_cross_sections[channel], self.cross_sections)))
    
@dataclass
class ScatteringDependence2D:
    parameters: npt.NDArray
    cross_sections: list[ScatteringObserwables]

    @staticmethod
    def parse_json(filename: str) -> 'ScatteringDependence2D':
        with open(filename, "r") as file:
            data = json.load(file)

        parameters = np.array(data['parameters'])

        cross_sections = [
            ScatteringObserwables(
                entrance = item['entrance'],
                scattering_length = complex(*item['scattering_length']),
                elastic_cross_section = item['elastic_cross_section'],
                inelastic_cross_sections = item['inelastic_cross_sections']
            )
            for item in data['observables']
        ]

        return ScatteringDependence2D(parameters, cross_sections)
    
    def slice(self, index: int, axis: int = 1) -> tuple[ScatteringDependence, float]:
        assert axis == 0 or axis == 1

        grid = np.unique(self.parameters[:, axis])
        assert index < len(grid)

        slice = grid[index]
        filtering = self.parameters[:, axis] == slice

        cross_sections = [self.cross_sections[i] for i in range(len(self.cross_sections)) if filtering[i]]

        return ScatteringDependence(list(self.parameters[filtering, (axis + 1) % 2]), cross_sections), slice

class ParameterType(Enum):
    Energy = 4
    Scaling = 5
    Field = 5
    FieldWithScaling = 6

def read_molscat_field_dependence(filename: str, parameter: ParameterType = ParameterType.FieldWithScaling) -> npt.NDArray[np.float64]:
    with open(filename, "r") as file:
        data = file.readlines()

    start = next(filter(lambda x: x[1].find("SCATTERING LENGTH") != -1, enumerate(data)))
    data = data[start[0] + 2:-2]
    
    data = list(map(lambda x: [float(x[parameter.value]), float(x[-2]) * ANGS, float(x[-1]) * ANGS], \
                    map(lambda x: x.split(), data)))
    data = np.array(data)
    
    return data

if __name__ == "__main__":
    read_molscat_field_dependence("data/srf_rb_n_0_molscat.3")