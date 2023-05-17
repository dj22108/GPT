import pandas as pd

csv_file = "employee_data.csv"
output_file = "employee_data.txt"
text_columns = ["Name", "Age", "Gender", "Company", "Designation", "Hobby"]  # Specify the column names containing text data

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Combine text from all columns into a single text column
df["Text"] = df[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

# Extract the text data from the "Text" column
text_data = df["Text"].values

# Save the text data to a text file
with open(output_file, "w", encoding="utf-8") as f:
    for text in text_data:
        f.write(str(text) + "\n")
