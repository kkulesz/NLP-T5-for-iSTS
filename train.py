import pandas as pd
import os

import utils
import consts
from data_processing import for_type, for_score
from t5_wrapper import T5Wrapper

train_data_path = os.path.join(consts.current_dataset, consts.train_data_file_name)

if __name__ == '__main__':
    utils.prepare_environment()

    config = utils.parse_arguments()
    t5_args = config.to_t5_args()

    data = pd.read_csv(train_data_path, sep='\t').astype(str)
    score_data = for_score(data)
    type_data = for_type(data)

    model = T5Wrapper.naked(t5_args)

    print(f"Training {consts.current_variant} on: {consts.current_dataset}")
    if consts.current_variant == consts.BOTH:
        both_data = score_data.append(type_data, ignore_index=True)
        model.train(both_data)
    elif consts.current_variant == consts.TYPE:
        model.train(type_data)
    elif consts.current_variant == consts.SCORE:
        model.train(score_data)
    else:
        raise Exception("Invalid training variant!")
