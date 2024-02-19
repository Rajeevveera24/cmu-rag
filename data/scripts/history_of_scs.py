import os
import re
import requests
from bs4 import BeautifulSoup

def parse_history(out_dir, f_name):
    page = requests.get("https://www.cs.cmu.edu/scs25/history")
    soup = BeautifulSoup(page.content, "html.parser")

    with open(os.path.join(out_dir, f_name), "w") as f:
        write_to_file(f, soup.find(id="page-title").text)
        for child in soup.find("div", class_="field-item"):
            if child.name == "h2":
                write_to_file(f, "")
            write_to_file(f, child.text)

def parse_ideas_from_scs(out_dir):
    page = requests.get("https://www.cs.cmu.edu/scs25/25things")
    soup = BeautifulSoup(page.content, "html.parser")
    for idx, fs in enumerate(soup.find_all("fieldset")):
        with open(os.path.join(out_dir, f"{(idx+1):02}.txt"), "w") as f:
            write_to_file(f, soup.find(id="page-title").text)
            for child in soup.find("div", class_="collapse-text-text").children:
                write_to_file(f, child.text)
            fs_title = fs.find("span", class_="fieldset-legend").text.strip()
            write_to_file(f, fs_title)
            write_to_file(f, fs.find("div", class_="fieldset-wrapper").text.strip())

def write_to_file(f, text):
    f.write(text+"\n")


if __name__ == "__main__":
    parse_history(out_dir="../documents/history_of_scs", f_name="00.txt")
    parse_ideas_from_scs(out_dir="../documents/history_of_scs")