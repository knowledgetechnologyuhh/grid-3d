import pytorch_lightning as pl
import torch
import torch.nn as nn
from munch import munchify
from torchmetrics.functional import accuracy
from torchvision.models import resnet101
from yacs.config import CfgNode as CN

from grid3d.dataset import QDRLDataModule
from grid3d.models.mac.config.defaults import _C
from grid3d.models.mac.model import MACNetwork
from grid3d.utils import gen_feature_extractor


class MACWrapper(MACNetwork, pl.LightningModule):
    def __init__(self, cfg):
        cfg = munchify(cfg)
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.get_features = gen_feature_extractor(resnet101(pretrained=True), "layer3")
        self.get_features.to(torch.device(cfg.DEVICE))
        tasks = sorted(task for (task, size) in cfg.TASK_SIZES.items() if size > 0)
        self.idx2task = dict(enumerate(tasks + tasks))
        self.idx2phase = dict(enumerate(["train"] * len(tasks) + ["val"] * len(tasks)))

    def _base_step(self, batch, phase, dataloader_idx=None):
        imgs, questions, question_lens, answers = batch
        with torch.no_grad():
            features = self.get_features(imgs)
        preds = self(features, questions, question_lens.to(torch.int64).to("cpu"))
        loss = self.criterion(preds, answers)
        metrics = {"loss": loss, "acc": accuracy(preds, answers)}
        log_options = {
            "on_step": True,
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "add_dataloader_idx": False,
        }
        for metric, value in metrics.items():
            if type(dataloader_idx) is int:
                phase = self.idx2phase[dataloader_idx]
                task = self.idx2task[dataloader_idx]
                self.log(f"{phase}/{task}/{metric}", value, **log_options)
            else:
                if phase == "val":
                    # Only one task is available
                    task = list(self.idx2task.values())[0]
                    self.log(f"{phase}/{task}/{metric}", value, **log_options)
                else:
                    self.log(f"{phase}/{metric}", value, **log_options)
        return metrics

    def training_step(self, batch, idx):
        return self._base_step(batch, phase="train")

    def validation_step(self, batch, idx, dataloader_idx=None):
        return self._base_step(batch, phase="val", dataloader_idx=dataloader_idx)

    def test_step(self, batch, idx):
        return self._base_step(batch, phase="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["SOLVER"]["LR"])
        return optimizer


if __name__ == "__main__":

    cfg = _C
    cfg.MODEL = "MAC"

    cfg.DATASET = CN()
    cfg.DATASET.PATH = "/data/grid3d/"
    cfg.DATALOADER.BATCH_SIZE = 64
    cfg.DATALOADER.NUM_WORKERS = 3

    cfg.MAC.SELF_ATT = False
    cfg.MAC.MEMORY_GATE = False
    cfg.MAC.MEMORY_GATE_BIAS = 0.0
    cfg.MAC.MAX_ITER = 4

    cfg.SOLVER.LR = 0.0001
    cfg.SOLVER.GRAD_CLIP = 0
    cfg.SOLVER.EPOCHS = 50

    cfg.TASK_SIZES = CN()
    cfg.TASK_SIZES.existence_prediction = 59053
    cfg.TASK_SIZES.orientation_prediction = 26344
    cfg.TASK_SIZES.link_prediction = 40770
    cfg.TASK_SIZES.relation_prediction = 69800
    cfg.TASK_SIZES.counting = 92904
    cfg.TASK_SIZES.triple_classification = 166603

    cfg = munchify(cfg)
    qdrl = QDRLDataModule(cfg)
    qdrl.setup()
    cfg.OUTPUT.DIM = len(qdrl.dataset.vocab)
    cfg.INPUT.N_VOCAB = len(qdrl.dataset.vocab) + 1  # consider the padding index
    model = MACWrapper(cfg)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.SOLVER.EPOCHS)
    trainer.fit(model, qdrl)
