
# Constants
run_name = "R2"
collector_folder_path = "../../data/collections"
data_folder_name = "saved_data"
model_folder_name = "saved_models"

# Training/validation/testing split
training_split = 0.75
validation_split = 0.20
testing_split = 0.05
assert training_split + validation_split + testing_split == 1