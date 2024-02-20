from pypdf import PdfReader
import os, re

os.chdir("../../..") # Change to the root directory of the project


RAW_DATA_PDF_DIR = "data/raw_data/"
PARSED_DATA_DIR = "data/documents/"
HISTORY_DIR = "history_of_cmu/"
PROGRAM_HANDBOOKS_DIR = "program_handbooks/"
FILE_NAME = "cmu_fact_sheet_02-pdf"

def parse_pdfs(raw_directory, raw_file_names, parsed_directory, parsed_file_names):
    
    assert len(raw_file_names) == len(parsed_file_names)
    
    for i in range(len(raw_file_names)):
        raw_file_name = raw_directory + raw_file_names[i]
        parsed_file = open(parsed_directory + parsed_file_names[i], 'w')
        
        raw_file = open(raw_file_name, 'rb')
        reader = PdfReader(raw_file)

        number_of_pages = len(reader.pages)

        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            parsed_file.write(text)
        
        parsed_file.close()
        raw_file.close()

# parse_pdfs([RAW_DATA_PDF_DIR + FILE_NAME[:-4] + ".pdf"], [PARSED_DATA_DIR + FILE_NAME + ".txt"])
        
raw_file_names = os.listdir(RAW_DATA_PDF_DIR)
parsed_file_names = [re.sub(".pdf", ".txt", file_name) for file_name in raw_file_names]

parse_pdfs(raw_directory=RAW_DATA_PDF_DIR, 
           raw_file_names=raw_file_names, 
           parsed_directory=PARSED_DATA_DIR + PROGRAM_HANDBOOKS_DIR, 
           parsed_file_names=parsed_file_names)
