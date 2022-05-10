from simpletransformers.t5 import T5Args

import consts


class TrainerConfig:
    def __init__(self,
                 model_type=consts.MODEL_TYPE,
                 max_input_size=consts.MAX_INPUT_SIZE,
                 max_target_size=consts.MAX_TARGET_SIZE,
                 learning_rate=consts.LEARNING_RATE
                 ):
        self.model_type = model_type
        self.max_input_size = max_input_size
        self.max_target_size = max_target_size
        self.learning_rate = learning_rate

    def to_t5_args(self) -> T5Args:
        return T5Args(
            learning_rate=self.learning_rate,
            # TODO:
        )
