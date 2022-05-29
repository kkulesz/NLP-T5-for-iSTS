from __future__ import annotations
from simpletransformers.t5 import T5Args

import consts


class Config:
    def __init__(self,
                 model_type=consts.MODEL_TYPE,
                 model_name=consts.MODEL_NAME,
                 learning_rate=consts.LEARNING_RATE,

                 model_output_dir=consts.default_model_output_dir,
                 ):
        self.model_type = model_type
        self.model_name = model_name
        self.learning_rate = learning_rate
        # TODO: other params: num of epochs etc

        self.model_output_dir = model_output_dir

    def to_t5_args(self) -> T5Args:
        return T5Args(
            model_type=self.model_type,
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            output_dir=self.model_output_dir,
            num_train_epochs=10,
            train_batch_size=1  # CUDA on my comp out of memory if greater ;/
        )

    @staticmethod
    def from_file(filename: str) -> Config:
        # TODO
        pass
