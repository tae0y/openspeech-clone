# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import hydra
import sentencepiece
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from openspeech.dataclass.initialize import hydra_train_init
from openspeech.datasets import DATA_MODULE_REGISTRY
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.utils import get_pl_trainer, parse_configs

from torch.profiler import profile, record_function


@hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    pl.seed_everything(configs.trainer.seed)

    logger, num_devices = parse_configs(configs)

    print(f"configs: {configs}")
    data_module = DATA_MODULE_REGISTRY[configs.dataset.dataset](configs)
    data_module.prepare_data()

    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

    data_module.setup()

    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)

    trainer = get_pl_trainer(configs, num_devices, logger)
    
    #반복되는 OOM 이슈 해결을 위한 프로파일링
    try:
        with profile(record_shapes=True, profile_memory=True) as prof:
            with record_function("model_training"):
                trainer.fit(model, data_module)
    except RuntimeError as e:
        print("An error occurred during training:", str(e))
        prof.export_chrome_trace("trace.json") # Save the profiling data to a file
        print("Profiling data saved to trace.json")
        return
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                
    trainer.test(model, data_module)


if __name__ == "__main__":
    hydra_train_init()
    hydra_main()
