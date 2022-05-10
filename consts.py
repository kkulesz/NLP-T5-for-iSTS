import os

#####################################
# -------- Directories-------- ------
#####################################
data_dir = 'data'

answers_students_data = os.path.join(data_dir, 'answers-students')
headlines_data = os.path.join(data_dir, 'headlines')
images_data = os.path.join(data_dir, 'images')

test_data_file_name = 'test_data.tsv'
train_data_file_name = 'train_data.tsv'

output_dir = 'output'

#####################################
# -------- Training parameters ------
#####################################
MODEL_TYPE = "t5-small"  # model_type: t5-small t5-base t5-large t5-3b t5-11b
MAX_INPUT_SIZE = 512
MAX_TARGET_SIZE = 50
SEED = 2137

LEARNING_RATE = 1e-4



model_args = {  # TODO: from config
    "max_seq_length": 196,
    "train_batch_size": 16,
    "eval_batch_size": 64,
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True
    # "wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
}
