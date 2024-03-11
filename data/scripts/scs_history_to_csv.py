import os, csv

READ_FILES_DIR = '/home/raj/Documents/cmu/sem2/anlp/assignments/cmu-rag/data/documents/history_of_scs/'

dct = []

for file in sorted(os.listdir(READ_FILES_DIR)):
    if file.endswith(".txt"):
        if file == "00.txt":
            continue
        with open(READ_FILES_DIR + file, 'r') as f:
            data = f.readlines()
            # print(data)
            line1 = data[0]
            line2 = data[1]
            line1 = line1.strip().split(".")[1]
            event, year = ' '.join(line1.split(",")[0:-1]), line1.split(",")[-1]
            description = line2.strip()
            # print(year, event, description)
            dct.append((year, event, description))

with open(READ_FILES_DIR + 'scs_hisotry.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dct)
