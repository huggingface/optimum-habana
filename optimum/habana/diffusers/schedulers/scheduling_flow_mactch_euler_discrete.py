from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class GaudiFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    # Overwrite orginal function with the following one to handle Gaudi lazy mode
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        masked = schedule_timesteps == timestep
        tmp = masked.cumsum(dim=0)
        pos = (tmp == 0).sum().item()
        if masked.sum() > 1:
            pos += (tmp == 1).sum().item()
        return pos
