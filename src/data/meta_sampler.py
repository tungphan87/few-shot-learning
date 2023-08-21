import numpy as np
import torch


class BatchSampler(object):
    def __init__(
        self,
        dataset_targets,
        include_query,
        N_way,
        K_shot,
        shuffle=True,
        shuffle_once=False,
    ):
        super().__init__()
        self.N_way = N_way
        self.K_shot = K_shot
        self.dataset_targets = dataset_targets
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2

        self.shuffle = shuffle
        self.shuffle_once = shuffle_once
        self.batch_size = self.N_way * self.K_shot
        self.classes, counts = np.unique(self.dataset_targets, return_counts=True)
        self.n_classes = len(self.classes)
        self.max_count = max(counts)

        self.indices = torch.Tensor(
            np.empty((self.n_classes, self.max_count), dtype=int) * np.nan
        )
        for i, c in enumerate(self.classes):
            indices_per_class = torch.where(self.dataset_targets == c)[0]
            self.indices[i, :] = indices_per_class

        self.iterations = (
            self.indices.shape[0] * self.indices.shape[1] // self.N_way // self.K_shot
        )
        if shuffle_once or shuffle:
            self.shuffle_data()

        self.n_batches_per_class = self.max_count // self.K_shot

    def shuffle_data(self):
        # shuffle within class
        for i in range(self.indices.shape[0]):
            self.indices[i, :] = self.indices[i, torch.randperm(self.max_count)]

        # shuffle between classes
        idxs = np.random.permutation(self.n_classes)
        self.classes = self.classes[idxs]
        self.indices = self.indices[idxs]

    def __iter__(self):
        # print(self.iterations)
        for it in range(self.iterations):
            row_it, col_it = (
                it // self.n_batches_per_class,
                it % self.n_batches_per_class,
            )
            row_slice = slice(row_it * self.N_way, (row_it + 1) * self.N_way)
            col_slice = slice(col_it * self.K_shot, (col_it + 1) * self.K_shot)

            index_batch = (
                self.indices[row_slice, col_slice]
                .reshape(
                    -1,
                )
                .type(torch.LongTensor)
            )
            # print(index_batch, index_batch.dtype)
            if self.include_query:
                index_batch = torch.concat([index_batch[::2], index_batch[1::2]])

            yield index_batch

    def __len__(self):
        return self.iterations


def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets
