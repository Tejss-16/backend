import pandas as pd
import uuid

class DataStore:
    def __init__(self):
        self._store = {}

    def save(self, df: pd.DataFrame) -> str:
        dataset_id = str(uuid.uuid4())
        self._store[dataset_id] = df
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame | None:
        return self._store.get(dataset_id)


# singleton instance
data_store = DataStore()