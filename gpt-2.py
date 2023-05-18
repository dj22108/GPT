import gpt_2_simple as gpt2
import os
import shutil
from tqdm import tqdm

# Specify the path to the downloaded GPT-2 model checkpoint
model_name = "355M"
model_checkpoint_dir = "models/355M"

# Download the GPT-2 model if it's not already downloaded
if not os.path.exists(model_checkpoint_dir):
    gpt2.download_gpt2(model_name=model_name)

# Specify the path to your fine-tuning data
data_file = "employee_data.txt"

# Specify the path to the output folder
output_folder = "output"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Fine-tune the GPT-2 model with progress bar
sess = gpt2.start_tf_sess()

# Define the progress bar
total_iterations = 5
progress_bar = tqdm(total=total_iterations, unit='iteration')

# Fine-tuning loop
for _ in range(total_iterations):
    gpt2.finetune(sess, data_file, model_name=model_name, checkpoint_dir=model_checkpoint_dir,learning_rate=0.0001,
                  batch_size=4, steps=4)
    progress_bar.update(1)

# Save the fine-tuned model
checkpoint_path = os.path.join(output_folder, "checkpoint")
os.makedirs(checkpoint_path, exist_ok=True)
shutil.copyfile(os.path.join(model_checkpoint_dir, "hparams.json"), os.path.join(checkpoint_path, "hparams.json"))
gpt2.save_checkpoint(output_folder)

# Close the progress bar
progress_bar.close()
