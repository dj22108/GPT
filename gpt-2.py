import gpt_2_simple as gpt2
import os

# Specify the path to the downloaded GPT-2 model checkpoint
model_name = "124M"
model_checkpoint_dir = "models/124M"

# Download the GPT-2 model if it's not already downloaded
if not os.path.exists(model_checkpoint_dir):
    gpt2.download_gpt2(model_name=model_name)

# Specify the path to your fine-tuning data
data_file = "employee_data.txt"

# Specify the path to the output folder
output_folder = "output/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Fine-tune the GPT-2 model
sess = gpt2.start_tf_sess()
gpt2.finetune(sess, data_file, model_name=model_name, checkpoint_dir=model_checkpoint_dir, model_dir=output_folder)

# Save the fine-tuned model
gpt2.save_checkpoint(output_folder)
