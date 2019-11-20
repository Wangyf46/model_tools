import torch


class BatchTransformDataLoader():
    # Only 3 channels is supported
    # Mean normalization on batch level instead of individual
    # https://github.com/NVIDIA/apex/blob/59bf7d139e20fb4fa54b09c6592a2ff862f3ac7f/examples/imagenet/main.py#L222
    def __init__(self, loader, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), fp16=False):
        self.loader = loader
        self.sampler = loader.sampler
        self.mean = (torch.tensor(mean) * 255).cuda().view(1, 3, 1, 1)
        self.std = (torch.tensor(std) * 255).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if self.fp16:
            self.mean, self.std = self.mean.half(), self.std.half()

    def __len__(self):
        return len(self.loader)

    def process_tensors(self, input, target, non_blocking=True):
        input = input.cuda(non_blocking=non_blocking)
        if self.fp16:
            input = input.half()
        else:
            input = input.float()

        return input.sub_(self.mean).div_(self.std), target.cuda(non_blocking=non_blocking)

    def update_batch_size(self, bs):
        self.loader.batch_sampler.batch_size = bs

    def __iter__(self):
        return (self.process_tensors(input, target, non_blocking=True) for input, target in self.loader)

