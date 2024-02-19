import os
import re
import requests
from bs4 import BeautifulSoup


def parse_data(urls, out_dir):
    document_counter = 0
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        header_text = soup.find(id="page-title").text

        view_content = soup.find("div", class_="view-content")
        els = view_content.find_all("td")
        for el in els:
            if el.find("div", class_="views-field-nothing") ==  None:
                continue
            with open(os.path.join(out_dir, f"{document_counter:02}.txt"), "a") as f:
                write_to_file(f, header_text)
                write_to_file(f, f"Name: {el.find("div", class_="views-field-nothing").text.strip()}")
                if el.find("span", class_="label") and el.find("span", class_="label").parent.find("a"):
                    write_to_file(f, f"Email: {el.find("span", class_="label").parent.find("a").text.strip()}")
                if el.find("div", class_="views-field-field-computed-building"):
                    write_to_file(f, f"{re.sub(' +', ' ', el.find("div", class_="views-field-field-computed-building").text.strip())}")
                if el.find("div", class_="views-field-field-research-areas"):
                    write_to_file(f, f"{re.sub(' +', ' ', el.find("div", class_="views-field-field-research-areas").text.strip())}")
                if el.find("div", class_="views-field-field-computed-phone"):
                    write_to_file(f, f"{re.sub(' +', ' ', el.find("div", class_="views-field-field-computed-phone").text.strip())}")
            document_counter += 1
        print(f"Successfully parsed url {url}.")

def write_to_file(f, text):
    f.write(text+"\n")


if __name__ == "__main__":
    urls = [
        "https://lti.cs.cmu.edu/directory/all/154/1",
        "https://lti.cs.cmu.edu/directory/all/154/2",
        "https://lti.cs.cmu.edu/directory/all/154/2728",
        "https://lti.cs.cmu.edu/directory/all/154/200"
    ]
    parse_data(urls=urls, out_dir="documents/lti_faculty")