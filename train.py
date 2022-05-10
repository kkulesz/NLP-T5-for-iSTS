import pandas as pd
import os
import torch

import utils
import consts
from trainer_config import TrainerConfig
from data_processing import for_type, for_score
from t5_wrapper import T5Wrapper

dataset = consts.answers_students_data
train_data_path = os.path.join(dataset, consts.train_data_file_name)

if __name__ == '__main__':
    utils.seed_torch()

    config = TrainerConfig()  # TODO: parse arguments
    t5_args = config.to_t5_args()

    data = pd.read_csv(train_data_path, sep='\t').astype(str)
    score_data = for_score(data)
    type_data = for_type(data)

    model = T5Wrapper.naked(t5_args)

    model.train(score_data)
