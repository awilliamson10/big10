from torch.utils.data import Dataset
from typing import List
from big10.preprocessing.context import get_processed_game_data

class Big10Dataset(Dataset):
    def __init__(self, game_ids: List[str]):
        self.game_ids = game_ids
        self.data = []
        for game_id in game_ids:
            try:
                game_input, drives = get_processed_game_data(game_id)
                if game_input is not None and drives is not None:
                    self.data.append({
                        "game_input": game_input,
                        "drives_text": drives
                    })
            except ValueError:
                pass
            except Exception as e:
                raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]