
# TODO: if pytorch v1.0 fix c10 bugs, we should remove deprecated
from mmcv.version_info import USE_TORCHV1
if USE_TORCHV1:
    import torch.nn.parallel.deprecated.distributed as parallel_dist
else:
    import torch.nn.parallel.distributed as parallel_dist

from .scatter_gather import scatter_kwargs


class MMDistributedDataParallel(parallel_dist.DistributedDataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0,
                 broadcast_buffers=True):
        super(MMDistributedDataParallel, self).__init__(module, device_ids=device_ids,
                                                        output_device=output_device, dim=dim,
                                                        broadcast_buffers=broadcast_buffers)
        print("Device IDs: ", self.device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)