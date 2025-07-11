from dataclasses import dataclass
import json
from typing import Iterable
import numpy as np
import numpy.typing as npt

class BoundsDependence:
    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter="\t", skiprows=1)

    @staticmethod
    def parse_json(filename: str) -> 'BoundsDependence':
        with open(filename, "r") as file:
            data = json.load(file)

        parameters = data['parameters']
        bound_states = [
            BoundStates(
                item['energies'],
                item['nodes']
            )
            for item in data['bound_states']
        ]

        data = np.zeros((0, 3))
        for parameter, bounds in zip(parameters, bound_states):
            for node, energy in zip(bounds.nodes, bounds.energies):
                data = np.append(data, np.array([parameter, node, energy]).reshape((1, 3)), axis=0)

        instance = BoundsDependence.__new__(BoundsDependence)
        instance.data = data

        return instance

    def dependence(self) -> npt.NDArray[np.float64]:
        return self.data[:, [0,2]]
    
    def states(self) -> Iterable[npt.NDArray[np.float64]]:
        states = np.unique(self.data[:, 1])
        for s in states:
            mask = self.data[:, 1] == s
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = np.delete(filtered, 1, axis=1)
            if len(filtered.shape) == 1:
                yield filtered
                continue

            sorted_indices = np.argsort(filtered[:, 0])

            yield filtered[sorted_indices]

    def fields(self) -> Iterable[npt.NDArray[np.float64]]:
        fields = np.unique(self.data[:, 0])
        for f in fields:
            mask = self.data[:, 0] == f
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = np.delete(filtered, 1, axis=1)
            if len(filtered.shape) == 1:
                yield filtered
                continue

            sorted_indices = np.argsort(filtered[:, 0])[::-1]

            yield filtered[sorted_indices]

class BoundsDependence2D:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def parse_json(filename: str) -> 'BoundsDependence2D':
        with open(filename, "r") as file:
            data = json.load(file)

        parameters = data['parameters']
        bound_states = [
            BoundStates(
                item['energies'],
                item['nodes']
            )
            for item in data['bound_states']
        ]

        data = np.zeros((0, 4))
        for parameter, bounds in zip(parameters, bound_states):
            for node, energy in zip(bounds.nodes, bounds.energies):
                data = np.append(data, np.array([parameter[0], parameter[1], node, energy]).reshape((1, 4)), axis=0)

        return BoundsDependence2D(data)

    def dependence(self) -> npt.NDArray[np.float64]:
        return self.data[:, [0, 1, 3]]
    
    def states(self) -> Iterable[npt.NDArray[np.float64]]:
        states = np.unique(self.data[:, 2])
        for s in states:
            mask = self.data[:, 2] == s
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = filtered[:, [0, 1, 3]]
            if len(filtered.shape) == 1:
                yield filtered
                continue

            yield filtered

    def slice_len(self, axis: int = 1) -> int:
        grid = np.unique(self.data[:, axis])
        return len(grid)

    def slice(self, index: int, axis: int = 1) -> tuple[BoundsDependence, float]:
        assert axis == 0 or axis == 1

        grid = np.unique(self.data[:, axis])
        assert index < len(grid)

        slice = grid[index]
        filtering = self.data[:, axis] == slice

        instance = BoundsDependence.__new__(BoundsDependence)
        instance.data = (self.data[filtering, :])[:, [(axis + 1) % 2, 2, 3]]

        return instance, slice # type: ignore
@dataclass
class BoundStates:
    energies: list[float]
    nodes: list[int]

if __name__ == "__main__":
    for s in BoundsDependence(f"data/srf_rb_bounds_2.dat").states():
        print(s)