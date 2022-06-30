import json
import random
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

PAD_IDX = 0
PAD_SYMBOL = "<pad>"


image_transform = Compose(
    [
        Resize([224, 224]),
        ToTensor(),
        Normalize(mean=[0.8214, 0.8195, 0.8192], std=[0.0910, 0.0970, 0.0996]),
    ]
)


def collate_fn(batch):
    """Sort the batch by the question length, which is the 2nd element of each
    example in the batch. This is required by the MAC model, which uses
    pack_padded_sequence. (cf.
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
    and https://stackoverflow.com/a/51030945)"""
    batch = sorted(batch, key=lambda example: example[2].item(), reverse=True)
    batch_collated = [torch.stack(tup) for tup in list(zip(*batch))]
    return batch_collated


def get_vocab(samples):
    question_words = set()
    answer_words = set()
    for sample in samples:
        question_words = question_words.union(set(sample["question"]))
        answer_words = answer_words.union({sample["answer"]})
    return sorted(question_words | answer_words)


class QDRL3D(Dataset):
    def __init__(
        self,
        data_path,
        transforms=None,
    ):
        self.img_idx2path = {
            int(p.stem): p.as_posix()
            for p in sorted(Path(data_path).glob("**/*.png"), key=lambda p: int(p.stem))
        }
        self.image_transform = image_transform if transforms is None else transforms
        with open(data_path / "questions.json", "r") as f:
            self.samples = json.load(f)
        self.vocab = get_vocab(self.samples)
        self.idx2token = {PAD_IDX: PAD_SYMBOL}
        self.idx2token.update(dict(enumerate(self.vocab, 1)))
        self.token2idx = {token: idx for (idx, token) in self.idx2token.items()}
        self.maxlen = max([len(sample["question"]) for sample in self.samples])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_idx = sample["img_idx"]
        img = Image.open(self.img_idx2path[img_idx]).convert("RGB")
        img_t = self.image_transform(img)
        question = sample["question"]
        question_len = torch.tensor(len(question))
        question_t = torch.tensor(
            [self.token2idx[token] for token in sample["question"]]
            + [PAD_IDX] * (self.maxlen - question_len)
        )
        answer_t = torch.tensor(self.token2idx[sample["answer"]])
        output = (img_t, question_t, question_len, answer_t)
        return output

    def __len__(self):
        return len(self.samples)


class QDRLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        data_path = Path(self.cfg.DATASET.PATH)
        self.dataset = QDRL3D(data_path=data_path)
        with open(data_path / "train_idxs.json", "r") as f:
            train_idxs = json.load(f)
        with open(data_path / "train_idxs_small.json", "r") as f:
            train_idxs_small = json.load(f)
        with open(data_path / "val_idxs.json", "r") as f:
            val_idxs = json.load(f)
        with open(data_path / "test_idxs.json", "r") as f:
            test_idxs = json.load(f)
        reduced_idxs = reduce_idxs(self.dataset, self.cfg.TASK_SIZES)
        train_idxs = sorted(set(reduced_idxs) & set(train_idxs))
        train_idxs_small = sorted(set(reduced_idxs) & set(train_idxs_small))
        val_idxs = sorted(set(reduced_idxs) & set(val_idxs))
        test_idxs = sorted(set(reduced_idxs) & set(test_idxs))
        self.train_dataset = Subset(self.dataset, train_idxs)
        self.train_dataset_small = Subset(self.dataset, train_idxs_small)
        self.val_dataset = Subset(self.dataset, val_idxs)
        self.test_dataset = Subset(self.dataset, test_idxs)

    def _get_loader(self, dataset, shuffle=True):
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        return loader

    def train_dataloader(self):
        return self._get_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        # Collect indices of validation samples categorized by tasks.
        train_small_loaders = self._get_loader_by_task(self.train_dataset_small)
        val_loaders = self._get_loader_by_task(self.val_dataset)
        return train_small_loaders + val_loaders

    def _get_loader_by_task(self, dataset_part):
        indices_by_task = defaultdict(list)
        for i in range(len(dataset_part)):
            index = dataset_part.indices[i]
            task = self.dataset.samples[index]["task"]
            indices_by_task[task].append(i)
        # Create validation dataloaders categorized by tasks.
        loaders = {}
        for task in indices_by_task:
            dataset = Subset(dataset_part, indices_by_task[task])
            loaders[task] = self._get_loader(dataset, shuffle=False)
        # The validation sets are sorted by tasks
        tasks = sorted(loaders.keys())
        combined_loaders = [loaders[task] for task in tasks]
        return combined_loaders

    def test_dataloader(self):
        return self._get_loader(self.test_dataset, shuffle=False)


def get_idxs_by_tasks(dataset):
    idxs_by_task = defaultdict(list)
    for i in range(len(dataset)):
        task = dataset.samples[i]["task"]
        idxs_by_task[task].append(i)
    return idxs_by_task


def get_tasks(dataset):
    return list(get_idxs_by_tasks(dataset).keys())


def get_task_sizes(dataset):
    return {task: len(idxs) for (task, idxs) in get_idxs_by_tasks(dataset).items()}


def reduce_idxs(dataset, task_sizes: dict[str, int]):
    """Reduce the size of the dataset for each task determined by
    `task_sizes`."""
    idxs_by_task = get_idxs_by_tasks(dataset)
    idxs_by_task_capped = {
        task: random.sample(idxs_by_task[task], task_sizes[task]) for task in task_sizes
    }
    idxs = [idx for idxs in idxs_by_task_capped.values() for idx in idxs]
    return idxs
