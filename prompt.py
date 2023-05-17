import gpt_2_simple as gpt2

# Specify the path to the fine-tuned model checkpoint
model_checkpoint_dir = "./output"

# Load the fine-tuned model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, checkpoint_dir=model_checkpoint_dir)

# Prompt variable
prompt = input("Enter your input prompt: ")  # You can take user input here or assign a manual input

# Generate text with the fine-tuned model using the prompt
generated_text = gpt2.generate(sess, length=100, prefix=prompt)
print(generated_text)
