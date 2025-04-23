import numpy as np


# 2. 采样式轮询DataLoader
class BucketBatchSampler:
    def __init__(self, buckets, weights, batch_size):
        self.buckets = list(buckets.values())
        self.weights = np.array(weights) / sum(weights)
        self.batch_size = batch_size
        self.num_batches = sum(len(b) // batch_size for b in self.buckets)

    def __iter__(self):
        # 每个epoch开始时打乱桶内顺序
        for bucket in self.buckets:
            np.random.shuffle(bucket)

        # 生成批次索引
        all_batches = []
        for bidx, bucket in enumerate(self.buckets):
            for i in range(0, len(bucket), self.batch_size):
                all_batches.append((bidx, bucket[i : i + self.batch_size]))

        # 按桶出现概率加权随机排序
        np.random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return self.num_batches
