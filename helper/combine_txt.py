import os


# os.chdir('..')
ROOT_DIR = '/home/raj/Documents/cmu/sem2/anlp/assignments/sudeep/cmu-rag/data/documents/'
OUTPUT_DIR = '/home/raj/Documents/cmu/sem2/anlp/assignments/sudeep/cmu-rag/helper/'

def combine_txt(file_path, output_path):
    for directory in os.listdir(file_path):
        # print(directory)    
        outfile = open(output_path + directory, 'w')
        for root, dirs, files in os.walk(file_path + directory):
            # print(root, dirs, files)
            for file in sorted(files):
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as infile:
                        # print(file)
                        outfile.write(infile.read())
                        outfile.write('\n')
        outfile.close()
    print('All txt files in the directory have been combined into one txt file')

def split_big_files(input_path, output_path, max_size=5000):
    for file in os.listdir(input_path):
        with open(input_path + file, 'r') as infile:
            lines = infile.readlines()
            size = len(lines)
            # if size < 1000:
            #     continue
            for i in range(0, size, max_size):
                with open(output_path + file + '_part_' + str(i), 'w') as outfile:
                    outfile.writelines(lines[i:min(i+max_size, size)])
# combine_txt(ROOT_DIR, OUTPUT_DIR + 'combined_txt_files/')
                    
split_big_files(OUTPUT_DIR + 'combined_txt_files/', OUTPUT_DIR + 'combined_txt_files_length_normalized/')
