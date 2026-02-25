import gzip
import pickle
from pathlib import Path
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Cell(BaseModel):
    idx: int
    day: str
    tags: set[int]
    expr: np.ndarray
    state: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Trajectory(BaseModel):
    cells: list[Cell] = Field(
        ...,
        description="List of Cell objects representing the trajectory. Must be exactly 3 cells long.",
        min_items=3,
        max_items=3,
    )
    trajectory_group_id: str = Field(
        ...,
        description="Unique identifier for trajectories from the same CellGroup sequence (d0->d7->d14).",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_expr(self) -> np.ndarray:
        """
        Returns the expression matrix of the trajectory.
        """
        return np.array([cell.expr for cell in self.cells])

    def to_indices(self) -> np.ndarray:
        """
        Returns the indices of the cells in the trajectory.
        """
        return [cell.idx for cell in self.cells]

    def to_days(self) -> list[str]:
        """
        Returns the days of the cells in the trajectory.
        """
        return [cell.day for cell in self.cells]

    def to_tags(self) -> list[set[int]]:
        """
        Returns the tags of the cells in the trajectory.
        """
        return [cell.tags for cell in self.cells]

    def to_states(self) -> list[str]:
        """
        Returns the states of the cells in the trajectory.
        """
        return [cell.state for cell in self.cells]


class TrajectoryList(BaseModel):
    trajectories: list[Trajectory] = Field(
        ...,
        description="List of Trajectory objects.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save(self, path: str = "trajectories.pkl.gz"):
        outfile = Path(path).expanduser()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(outfile, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str = "trajectories.pkl.gz") -> "TrajectoryList":
        infile = Path(path)
        with gzip.open(infile, "rb") as f:
            trajectories: TrajectoryList = pickle.load(f)
        return trajectories

    def get_unique_cells_by_day(self) -> dict[str, list[Cell]]:
        def all_cells():
            for traj in self.trajectories:
                for cell in traj.cells:
                    yield cell

        idx2cells = {cell.idx: cell for cell in all_cells()}
        day0_cell_indices = set(traj.cells[0].idx for traj in self.trajectories)
        day7_cell_indices = set(traj.cells[1].idx for traj in self.trajectories)
        day14_cell_indices = set(traj.cells[2].idx for traj in self.trajectories)

        return {
            "day0": [idx2cells[idx] for idx in day0_cell_indices],
            "day7": [idx2cells[idx] for idx in day7_cell_indices],
            "day14": [idx2cells[idx] for idx in day14_cell_indices],
        }


class AEConfig(BaseModel):
    latent_dim: int = Field(512, description="Dimension of AE latent embedding")
    batch_size: int = Field(2048)
    epochs: int = Field(500)
    learning_rate: float = Field(5e-4)
    patience: int = Field(30)
    l2: float = Field(1e-5)
    input_dropout: float = Field(
        0.0, description="Dropout rate applied to encoder input"
    )
    results_dir: Path = Field(Path("local_bucket/ae_results"))


class LSTMConfig(BaseModel):
    cell_type: Literal["LSTM", "GRU"] = Field("LSTM")
    num_layers: int = Field(3)
    hidden_dim: int = Field(256)
    dropout: float = Field(0.3)
    batch_size: int = Field(1024)
    epochs: int = Field(300)
    learning_rate: float = Field(5e-4)
    patience: int = Field(75)
    l2: float = Field(1e-5, description="L2 regularization strength")
