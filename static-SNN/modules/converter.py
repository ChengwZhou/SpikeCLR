import torch
import torch.distributed as dist

class BatchNormConverter:
    @classmethod
    def convert_sync_bn_to_bn(cls, module):
        """
        recursively convert SyncBatchNorm back to BatchNorm
        """
        module_output = module
        if isinstance(module, torch.nn.SyncBatchNorm):
            module_output = torch.nn.BatchNorm2d(module.num_features,
                                                 eps=module.eps,
                                                 momentum=module.momentum,
                                                 affine=module.affine,
                                                 track_running_stats=module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight.clone().detach()
                    module_output.bias = module.bias.clone().detach()
            module_output.running_mean = module.running_mean.clone().detach()
            module_output.running_var = module.running_var.clone().detach()
            module_output.num_batches_tracked = module.num_batches_tracked.clone().detach()
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_bn_to_bn(child))
        return module_output