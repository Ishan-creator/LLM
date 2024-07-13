from datasets import load_dataset

# Load the dataset
ds = load_dataset("Shushant/nepali")
print(ds)

# Define a function to save the text data to a .txt file
def save_entire_dataset_to_txt(dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for split in dataset:
            for item in dataset[split]:
                f.write(item['text'] + '\n')

# Save the entire dataset to a .txt file
save_entire_dataset_to_txt(ds, 'data/nepali_dataset.txt')
