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

default_model_output_dir = "model_storage"

#####################################
# -------- Training parameters ------
#####################################
MODEL_TYPE = "t5-small"  # model_type: t5-small t5-base t5-large t5-3b t5-11b

LEARNING_RATE = 1e-4
