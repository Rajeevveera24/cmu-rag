import os
import re
import requests
from bs4 import BeautifulSoup

def parse_history(out_dir):
    page = requests.get("https://www.cmu.edu/about/history.html")
    soup = BeautifulSoup(page.content, "html.parser")
    for idx, el in enumerate(soup.find_all("div", class_="column2")):
        with open(os.path.join(out_dir, f"{(idx+1):02}.txt"), "w") as f:
            write_to_file(f, el.text.strip())

def write_to_file(f, text):
    f.write(text+"\n")


if __name__ == "__main__":
    parse_history(out_dir="../documents/history_of_cmu")