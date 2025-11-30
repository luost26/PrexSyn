import copy
import pathlib
import pickle
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Self, TypedDict, TypeVar, cast

import lmdb
import numpy as np

from prexsyn_engine.synthesis import Synthesis

_DataType = TypeVar("_DataType", bound=Mapping[str, Any])


class ResultDatabase(MutableMapping[str, _DataType]):
    def __init__(self, path: str | pathlib.Path, extra_object_fields: Sequence[str]) -> None:
        super().__init__()
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extra_object_fields = extra_object_fields
        self._keys_cache: set[str] | None = None

    def __enter__(self) -> Self:
        self.db: lmdb.Environment = lmdb.open(
            str(self.path),
            map_size=384 * 1024 * 1024 * 1024,
            subdir=False,
            readonly=False,
            lock=True,
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.db.close()
        del self.db

    def __getitem__(self, key: str) -> _DataType:
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
            if value is None:
                raise KeyError(f"Key {key} not found in the database.")
            data: dict[str, Any] = pickle.loads(value)
            for xfield in self.extra_object_fields:
                if xfield in data:
                    xfield_key = f"_{key}/{xfield}"
                    xfield_value = txn.get(xfield_key.encode("utf-8"))
                    if xfield_value is None:
                        raise KeyError(f"Key {xfield} not found in the database.")
                    data[xfield] = pickle.loads(xfield_value)
        return cast(_DataType, data)

    def get_without_extra(self, key: str) -> _DataType:
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
            if value is None:
                raise KeyError(f"Key {key} not found in the database.")
            data: dict[str, Any] = pickle.loads(value)
            for xfield in self.extra_object_fields:
                if xfield in data:
                    del data[xfield]
            return cast(_DataType, data)

    def __setitem__(self, key: str, data: _DataType) -> None:
        self._keys_cache = None  # Invalidate cache
        data = copy.copy(data)  # Shallow copy to ensure we don't modify the input object
        with self.db.begin(write=True) as txn:
            for xfield in self.extra_object_fields:
                if xfield in data:
                    xfield_key = f"_{key}/{xfield}"
                    txn.put(xfield_key.encode("utf-8"), pickle.dumps(data[xfield]))
                    data[xfield] = 1  # type: ignore[index]
            txn.put(key.encode("utf-8"), pickle.dumps(data))

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("Deletion is not supported in this database.")

    def __iter__(self) -> Iterator[str]:
        if self._keys_cache is not None:
            yield from self._keys_cache
            return

        self._keys_cache = set()
        with self.db.begin(write=False) as txn:
            for key_b in txn.cursor().iternext(values=False):
                key = cast(bytes, key_b).decode("utf-8")
                if not key.startswith("_"):
                    self._keys_cache.add(key)
                    yield key

    def __len__(self) -> int:
        if self._keys_cache is not None:
            return len(self._keys_cache)
        return sum(1 for _ in self)


class AnalogGenerationResult(TypedDict):
    synthesis: list[Synthesis]
    similarity: np.ndarray[Any, Any]
    max_sim_product_idx: np.ndarray[Any, Any]
    time_taken: float


class AnalogGenerationDatabase(ResultDatabase[AnalogGenerationResult]):
    def __init__(self, path: pathlib.Path | str) -> None:
        super().__init__(path, extra_object_fields=["synthesis"])
        self._max_similarities: dict[str, float] = {}

    def __enter__(self) -> Self:
        super().__enter__()
        for key in iter(self):
            data = self.get_without_extra(key)
            self._max_similarities[key] = data["similarity"].max()
        return self

    def __setitem__(self, key: str, data: AnalogGenerationResult) -> None:
        super().__setitem__(key, data)
        self._max_similarities[key] = data["similarity"].max()

    def get_average_similarity(self) -> float:
        return sum(self._max_similarities.values()) / len(self._max_similarities) if self._max_similarities else 0.0

    def get_reconstruction_rate(self) -> float:
        return (
            sum(1 for sim in self._max_similarities.values() if sim == 1.0) / len(self._max_similarities)
            if self._max_similarities
            else 0.0
        )

    def get_time_statistics(self) -> tuple[float, float]:
        times = [self.get_without_extra(key)["time_taken"] for key in iter(self)]
        return (float(np.mean(times)), float(np.std(times))) if times else (0.0, 0.0)
