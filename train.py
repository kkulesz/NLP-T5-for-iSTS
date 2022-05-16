import pandas as pd
import os

import utils
import consts
from trainer_config import TrainerConfig
from data_processing import for_type, for_score
from t5_wrapper import T5Wrapper

train_data_path = os.path.join(consts.current_dataset, consts.train_data_file_name)

if __name__ == '__main__':
    utils.seed_torch()

    config = TrainerConfig()  # TODO: parse arguments
    t5_args = config.to_t5_args()

    data = pd.read_csv(train_data_path, sep='\t').astype(str)
    score_data = for_score(data)
    type_data = for_type(data)

    model = T5Wrapper.naked(t5_args)

    if consts.train_both:
        print(f"Training both tasks on: {consts.current_dataset}")
        both_data = score_data.append(type_data, ignore_index=True)
        model.train(both_data)
    else:
        if consts.train_type:
            print(f"Training TYPE task on: {consts.current_dataset}")
            model.train(type_data)
        else:
            print(f"Training SCORE task on: {consts.current_dataset}")
            model.train(score_data)
