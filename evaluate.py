import pandas as pd
import os

import utils
import consts
from trainer_config import TrainerConfig
from data_processing import for_type_eval, for_score_eval
from t5_wrapper import T5Wrapper

test_data_path = os.path.join(consts.current_dataset, consts.test_data_file_name)

if __name__ == '__main__':
    utils.seed_torch()

    config = TrainerConfig()  # TODO: parse arguments
    t5_args = config.to_t5_args()

    data = pd.read_csv(test_data_path, sep='\t').astype(str)
    score_data = for_score_eval(data)
    type_data = for_type_eval(data)

    model_dir = consts.default_model_output_dir  # TODO: change to proper dir
    model = T5Wrapper.pretrained(model_dir, t5_args)

    if consts.train_type or consts.train_both:
        print("Evaluating for TYPE...")
        type_labels = data['y_type'].tolist()
        type_predictions = model.predict(type_data)
        print(type_labels)
        print(type_predictions)
        # TODO zapisać do jakiejś csv

    if not consts.train_type or consts.train_both:
        print("Evaluating for SCORE...")
        score_labels = data['y_score'].tolist()
        score_predictions = model.predict(score_data)
        print(score_labels)
        print(score_predictions)
        # TODO zapisać do jakiejś csv
