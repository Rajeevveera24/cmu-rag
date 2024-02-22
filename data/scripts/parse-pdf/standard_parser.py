from pypdf import PdfReader
import os, re

os.chdir("../../..") # Change to the root directory of the project


RAW_DATA_PDF_DIR = "data/raw_data/"
PARSED_DATA_DIR = "data/documents/"
HISTORY_DIR = "history_of_cmu/"
PROGRAM_HANDBOOKS_DIR = "program_handbooks/"
ACADEMIC_CALENDARS_DIR = "academic_calendars/"
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

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [month.lower() for month in MONTHS]
DAYS_MAPPED = {'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'R': 'Thursday', 'F': 'Friday', 'S': 'Saturday', 'U': 'Sunday'}

def parse_pdfs_academic_calendars(raw_directory, raw_file_names, parsed_directory, parsed_file_names):
        
        assert len(raw_file_names) == len(parsed_file_names)
        
        for i in range(len(raw_file_names)):
            raw_file_name = raw_directory + raw_file_names[i]
            parsed_file = open(parsed_directory + parsed_file_names[i], 'w')
            
            raw_file = open(raw_file_name, 'rb')
            reader = PdfReader(raw_file)
    
            number_of_pages = len(reader.pages)
    
            for i in range(number_of_pages):
                page = reader.pages[i]
                text_list = page.extract_text().split('\n')
                lines_to_write = []
                for text in text_list:
                    words = text.split()
                    words_to_write = []
                    if words[0].lower() in MONTHS:
                        words_to_write.append("Date - ")
                        words_to_write.append(words[0])
                        words_to_write.append(words[1])
                        if words[2].upper() in DAYS_MAPPED:
                            words_to_write.append("; Day -")
                            words_to_write.append(DAYS_MAPPED[words[2]])
                        words_to_write.append("; Event -")
                        words_to_write.extend(words[3:])
                        line_to_write = ' '.join(words_to_write)
                    else:
                        line_to_write = text
                    lines_to_write.append(line_to_write)
                parsed_file.write('\n'.join(lines_to_write))
                # parsed_file.write(text)
            
            parsed_file.close()
            raw_file.close()

# parse_pdfs([RAW_DATA_PDF_DIR + FILE_NAME[:-4] + ".pdf"], [PARSED_DATA_DIR + FILE_NAME + ".txt"])
        
raw_file_names = os.listdir(RAW_DATA_PDF_DIR)
parsed_file_names = [re.sub(".pdf", ".txt", file_name) for file_name in raw_file_names]

parse_pdfs_academic_calendars(raw_directory=RAW_DATA_PDF_DIR, 
           raw_file_names=raw_file_names, 
           parsed_directory=PARSED_DATA_DIR + ACADEMIC_CALENDARS_DIR, 
           parsed_file_names=parsed_file_names)
