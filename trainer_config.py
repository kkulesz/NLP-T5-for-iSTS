from simpletransformers.t5 import T5Args

import consts


class TrainerConfig:
    def __init__(self,
                 model_type=consts.MODEL_TYPE,
                 model_output_dir=consts.default_model_output_dir,
                 learning_rate=consts.LEARNING_RATE
                 ):
        self.model_type = model_type
        self.model_output_dir = model_output_dir

        self.learning_rate = learning_rate

    def to_t5_args(self) -> T5Args:
        return T5Args(
            learning_rate=self.learning_rate,
            output_dir=self.model_output_dir
        )
