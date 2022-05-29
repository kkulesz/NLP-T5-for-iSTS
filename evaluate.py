import pandas as pd
import os

import utils
import consts
from data_processing import for_type_eval, for_score_eval
from t5_wrapper import T5Wrapper

test_data_path = os.path.join(consts.current_dataset, consts.test_data_file_name)

if __name__ == '__main__':
    utils.prepare_environment()

    config = utils.parse_arguments()
    t5_args = config.to_t5_args()

    data = pd.read_csv(test_data_path, sep='\t').astype(str)
    score_data = for_score_eval(data)
    type_data = for_type_eval(data)

    model_dir = consts.default_model_output_dir
    model = T5Wrapper.pretrained(model_dir, t5_args)

    print(f"Evaluating for {consts.current_variant} on {consts.current_dataset}...")

    type_predictions = model.predict(type_data)
    score_predictions = model.predict(score_data)

    df = pd.DataFrame({'type_predictions': type_predictions, 'score_predictions': score_predictions})
    df.to_csv("out.csv", index=False)

    # TODO: trzeba jakos obliczyc metryki, są gotowe funkcje w pakiecie (chyba) sklearn, ale on dał jakies skrypty perlowe
    #   i nie wiem czy to za ich pomocą powinniśmy to policzyć czy co
