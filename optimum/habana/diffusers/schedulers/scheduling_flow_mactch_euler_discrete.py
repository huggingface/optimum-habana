from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class GaudiFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    # TODO: overwrite orginal func with following one to fix dyn error in gaudi lazy mode
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        # pos = 1 if len(indices) > 1 else 0

        # return indices[pos].item()

        masked = schedule_timesteps == timestep
        tmp = masked.cumsum(dim=0)
        pos = (tmp == 0).sum().item()
        if masked.sum() > 1:
            pos += (tmp == 1).sum().item()
        return pos
