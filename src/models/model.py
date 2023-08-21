import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F


from src.data.meta_sampler import split_batch


def get_convnet(output_size):
    convnet = torchvision.models.DenseNet(
        growth_rate=32,
        block_config=(6, 6, 6, 6),
        bn_size=2,
        num_init_features=64,
        num_classes=output_size,  # Output dimensionality
    )
    return convnet


class ProtoNet(pl.LightningModule):
    def __init__(self, proto_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_convnet(output_size=self.hparams.proto_dim)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[140, 180], gamma=0.1
        )
        return [optimizer], [scheduler]

    def calculate_prototypes(features, targets):
        classes, _ = torch.unique(targets).sort()
        prototypes = []

        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)
            prototypes.append(p)

        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes

    def classify_feats(self, prototypes, classes, feats, targets):
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(
            dim=2
        )  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc

    def calculate_loss(self, batch, mode):
        imgs, targets = batch
        features = self.model(imgs)  # Encode all images of support and query set
        support_feats, query_feats, support_targets, query_targets = split_batch(
            features, targets
        )
        prototypes, classes = ProtoNet.calculate_prototypes(
            support_feats, support_targets
        )
        preds, labels, acc = self.classify_feats(
            prototypes, classes, query_feats, query_targets
        )
        loss = F.cross_entropy(preds, labels)

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self.calculate_loss(batch, mode="val")
