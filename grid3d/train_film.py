import pytorch_lightning as pl
import torch
import torch.nn as nn
from munch import munchify
from torchmetrics.functional import accuracy
from torchvision.models import resnet101
from vr.models import FiLMedNet, FiLMGen
from yacs.config import CfgNode as CN

from grid3d.dataset import QDRLDataModule
from grid3d.utils import gen_feature_extractor


class FiLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pg = FiLMGen(**config.FILM_GEN)
        self.ee = FiLMedNet(**config.FILMED_NET)

    def forward(self, questions, feats, question_lens):
        programs_pred = self.pg(questions, question_lens)
        scores = self.ee(feats, programs_pred)
        return scores


class FiLMWrapper(FiLM, pl.LightningModule):
    def __init__(self, cfg):
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
        preds = self(questions, features, question_lens)
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        return optimizer


if __name__ == "__main__":

    filmgen_args = {
        "null_token": 0,
        "start_token": 1,
        "end_token": 2,
        "wordvec_dim": 200,
        "hidden_dim": 4096,
        "rnn_num_layers": 1,
        "rnn_dropout": 0,
        "output_batchnorm": False,
        "bidirectional": False,
        "encoder_type": "gru",
        "decoder_type": "linear",
        "gamma_option": "linear",
        "gamma_baseline": 1.0,
        "num_modules": 4,
        "module_num_layers": 1,
        "module_dim": 128,
        "parameter_efficient": True,
        "debug_every": float("inf"),
    }

    filmednet_args = {
        "feature_dim": (1024, 14, 14),
        "stem_num_layers": 1,
        "stem_batchnorm": True,
        "stem_kernel_size": 3,
        "stem_stride": 1,
        "num_modules": 4,
        "module_num_layers": 1,
        "module_dim": 128,
        "module_residual": True,
        "module_batchnorm": True,
        "module_batchnorm_affine": False,
        "module_dropout": 0.0,
        "module_input_proj": 1,
        "module_kernel_size": 3,
        "classifier_proj_dim": 512,
        "classifier_downsample": "maxpoolfull",
        "classifier_fc_layers": (1024,),
        "classifier_batchnorm": True,
        "classifier_dropout": 0,
        "condition_method": "bn-film",
        "condition_pattern": (1, 1, 1, 1),
        "use_gamma": True,
        "use_beta": True,
        "use_coords": 1,
        "debug_every": float("inf"),
        "print_verbose_every": 20000000,
        "verbose": True,
    }

    cfg = CN()

    cfg.DEVICE = "cuda"

    cfg.MODEL = "FiLM"

    cfg.DATASET = CN()
    cfg.DATASET.PATH = "/data/grid3d/"

    cfg.DATALOADER = CN()
    cfg.DATALOADER.BATCH_SIZE = 64
    cfg.DATALOADER.NUM_WORKERS = 3

    cfg.FILM = CN()
    cfg.FILM_GEN = CN(filmgen_args)
    cfg.FILMED_NET = CN(filmednet_args)

    cfg.SOLVER = CN()
    cfg.SOLVER.LR = 0.0001
    cfg.SOLVER.WEIGHT_DECAY = 1e-5
    cfg.SOLVER.EPOCHS = 50

    cfg.TASK_SIZES = CN()
    cfg.TASK_SIZES.existence_prediction = 59053
    cfg.TASK_SIZES.orientation_prediction = 26344
    cfg.TASK_SIZES.link_prediction = 40770
    cfg.TASK_SIZES.relation_prediction = 69800
    cfg.TASK_SIZES.counting = 92904
    cfg.TASK_SIZES.triple_classification = 166603

    qdrl = QDRLDataModule(cfg)
    qdrl.setup()

    cfg = munchify(cfg)
    cfg.FILM_GEN.encoder_vocab_size = len(qdrl.dataset.vocab) + 3  # (PAD, BOS, EOS)
    vocab = {"answer_idx_to_token": qdrl.dataset.idx2token}
    cfg.FILMED_NET.vocab = vocab

    model = FiLMWrapper(cfg)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.SOLVER.EPOCHS)
    trainer.fit(model, qdrl)
