from pypdf import PdfReader
import os

os.chdir("../../..") # Change to the root directory of the project


RAW_DATA_PDF_DIR = "data/raw_data/"
PARSED_DATA_DIR = "data/documents/"
HISTORY_DIR = "history_of_cmu/"
FILE_NAME = "cmu_fact_sheet_02-pdf"

def parse_pdfs(raw_file_names, parsed_file_names):
    
    assert len(raw_file_names) == len(parsed_file_names)
    
    for i in range(len(raw_file_names)):
        raw_file_name = raw_file_names[i]
        parsed_file = open(parsed_file_names[i], 'w')
        
        raw_file = open(raw_file_name, 'rb')
        reader = PdfReader(raw_file)

        number_of_pages = len(reader.pages)

        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            parsed_file.write(text)
        
        parsed_file.close()
        raw_file.close()

parse_pdfs([RAW_DATA_PDF_DIR + FILE_NAME[:-4] + ".pdf"], [PARSED_DATA_DIR + FILE_NAME + ".txt"])
