import os


#####################################
# -------- Directories-------- ------
#####################################
data_dir = 'data'

answers_students_data = os.path.join(data_dir, 'answers-students')
headlines_data = os.path.join(data_dir, 'headlines')
images_data = os.path.join(data_dir, 'images')

output_dir = 'output'
model_storage = os.path.join(output_dir, 'model')


#####################################
# -------- Training parameters ------
#####################################
MODEL_TYPE = "t5-small"  # model_type: t5-small t5-base t5-large t5-3b t5-11b
MAX_INPUT_SIZE = 512
MAX_TARGET_SIZE = 50
SEED = 2137

LEARNING_RATE=1e-4
