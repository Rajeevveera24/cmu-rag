from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re 


####################################################################################################################################
############################################################# Version 1 ############################################################
####################################################################################################################################
cs = webdriver.ChromeService(executable_path='/opt/homebrew/Caskroom/chromedriver/122.0.6261.57/chromedriver-mac-arm64/chromedriver')
driver = webdriver.Chrome(service=cs)

driver.get('https://web.cvent.com/event/ab7f7aba-4e7c-4637-a1fc-dd1f608702c4/websitePage:645d57e4-75eb-4769-b2c0-f201a0bfc6ce?locale=en')
time.sleep(5)

all_texts = []

expandable_elements = driver.find_elements(By.CLASS_NAME, 'AgendaStyles__sessionName___yMBga')

for index, element in enumerate(expandable_elements):
    try:
        element.click()

        WebDriverWait(driver, 10).until(EC.visibility_of(element))

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        current_container = soup.find_all(class_='AgendaStyles__panel___wmdLJ')[index]

        container_text = ' '.join(current_container.stripped_strings)
        all_texts.append(container_text)
        
    except Exception as e:
        print(f"Could not click the element due to: {e}")


# for text in all_texts:
#     date_match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}\s*–\s*\d{1,2}/\d{1,2}/\d{2,4}', text)
#     title_match = re.search(r'^[A-Za-z ]+', text)

#     if date_match and title_match:
#         title = title_match.group()
#         date = date_match.group()
#         body_start_index = date_match.end()
#         body = text[body_start_index:].strip()
        
#         print(f"Title: {title}\nDate: {date}\nBody: {body}\n")
#     else:
#         # print("Could not parse text properly.")
#         print("#############################",text,"#############################")
# for text in all_texts:
#     # Split the text into 'Title', 'Date', and 'Body'
#     parts = text.split(" ET ")
#     if len(parts) > 1:
#         body = parts[1].strip()
#         title_and_date = parts[0].strip()
#         # Further split to isolate the title and date
#         date_match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}\s*–\s*\d{1,2}/\d{1,2}/\d{2,4}', title_and_date)
#         if date_match:
#             date = date_match.group()
#             title = title_and_date.replace(date, '').strip()
#             print(f"Title: {title}\nDate: {date}\nBody: {body}\n")
#         else:
#             print(f"Text does not contain a date: {text}\n")
#     else:
#         print(f"Text does not follow the expected format: {text}\n")

driver.quit()