import importlib.metadata

from packaging import version

from .models.modeling_base import adapt_PreTrainedModelWrapper_to_gaudi
from .models.modeling_sd_base import GaudiDefaultDDPOStableDiffusionPipeline
from .trainer.ddpo_trainer import GaudiDDPOTrainer
from .trainer.dpo_config import GaudiDPOConfig
from .trainer.dpo_trainer import GaudiDPOTrainer


trl_version = importlib.metadata.version("trl")
if version.parse(trl_version) < version.parse("0.17.0"):
    from .trainer.ppo_config import GaudiPPOConfig
    from .trainer.ppo_trainer import GaudiPPOTrainer
else:
    from .trainer.grpo_config import GaudiGRPOConfig
    from .trainer.grpo_trainer import GaudiGRPOTrainer
from .trainer.reward_trainer import GaudiRewardTrainer, RewardDataCollatorWithPadding
from .trainer.sft_config import GaudiSFTConfig
from .trainer.sft_trainer import GaudiSFTTrainer
