from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RnaProConfig:
    feature_dim: int = 256
    kmer_size: int = 4
    n_models: int = 5
    seed: int = 123
    min_coverage: float = 0.30

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if self.kmer_size <= 0:
            raise ValueError("kmer_size must be > 0")
        if self.n_models <= 0:
            raise ValueError("n_models must be > 0")
        if self.min_coverage <= 0 or self.min_coverage > 1:
            raise ValueError("min_coverage must be in (0,1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RnaProConfig":
        return cls(
            feature_dim=int(payload.get("feature_dim", 256)),
            kmer_size=int(payload.get("kmer_size", 4)),
            n_models=int(payload.get("n_models", 5)),
            seed=int(payload.get("seed", 123)),
            min_coverage=float(payload.get("min_coverage", 0.30)),
        )

