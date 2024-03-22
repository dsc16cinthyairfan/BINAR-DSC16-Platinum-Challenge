import csv

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    with open(csv_file, 'w', newline='') as file:
        fieldnames = ['tweets', 'labels']

        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()

        for line in lines:
            words = line.strip().split("\t")
            writer.writerow({ 'tweets': words[0], 'labels': words[-1]})

    print("File berhasil dikonversi!")

txt_file = 'train_preprocess.txt'
csv_file = 'train_preprocess.csv'

txt_to_csv(txt_file, csv_file)