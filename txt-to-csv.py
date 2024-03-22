# Converting the training data from txt to csv format

import csv

def txt_to_csv(txt_file, csv_file):
    # Membuka file teks untuk dibaca
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Membuka file CSV untuk ditulis
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter ='\t')

        # Menulis setiap baris dari file teks ke file CSV
        for line in lines:
            # Memisahkan baris menjadi kata-kata
            words = line.strip().split()

            # Memisahkan kata terakhir dari baris dan menyimpannya
            last_word = words[-1]

            # Menulis kata-kata ke kolom pertama
            writer.writerow(words[:-1])

            # Menulis kata terakhir ke kolom kedua
            writer.writerow([last_word])

    print("File berhasil dikonversi!")

# Ganti nama file input dan output sesuai kebutuhan Anda
txt_file = 'train_preprocess.txt'
csv_file = 'train_preprocess.csv'

# Panggil fungsi untuk melakukan konversi
txt_to_csv(txt_file, csv_file)
