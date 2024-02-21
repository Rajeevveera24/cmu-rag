import os
import re
import requests
from bs4 import BeautifulSoup

def parse_programs(out_dir):
    
    document_counter = 0
    page = requests.get("https://lti.cs.cmu.edu/learn")
    soup = BeautifulSoup(page.content, "html.parser")

    header = soup.find(id="page-title").text

    pitch = soup.find("h2", class_="pitch").text

    # PhD programs
    program_type = soup.find_all("h2", class_="section_header")[0].text

    document_counter = parse_section(
        div_id="block-quicktabs-phd-lti", 
        header=header, 
        pitch=pitch,
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Handbook", "Additional Info"],
        program_name_idx=0, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )

    document_counter = parse_section(
        div_id="block-quicktabs-dual-degree-in-language-and-info", 
        header=header, 
        pitch=pitch, 
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Additional Info"],
        program_name_idx=1, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )

    # MS programs
    program_type = soup.find_all("h2", class_="section_header")[1].text

    document_counter = parse_section(
        div_id="quicktabs-language_technologies", 
        header=header, 
        pitch=pitch, 
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Handbook", "Additional Info"],
        program_name_idx=2, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )

    document_counter = parse_section(
        div_id="quicktabs-master_of_science_in_intelligent", 
        header=header, 
        pitch=pitch, 
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Handbook", "Additional Info"],
        program_name_idx=3, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )

    document_counter = parse_section(
        div_id="quicktabs-master_of_science_in_computation", 
        header=header, 
        pitch=pitch, 
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Handbook", "Additional Info"],
        program_name_idx=4, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )

    document_counter = parse_section(
        div_id="quicktabs-master_of_science_in_biotechnolo", 
        header=header, 
        pitch=pitch, 
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Handbook", "Additional Info"],
        program_name_idx=5, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )

    # UG programs
    program_type = soup.find_all("h2", class_="section_header")[2].text

    document_counter = parse_section(
        div_id="quicktabs-undergraduate_minor_in_language_", 
        header=header, 
        pitch=pitch, 
        headers = ["Overview", "Requirements", "Curriculum", "Admission", "Additional Info"],
        program_name_idx=6, 
        program_type=program_type,
        soup=soup,
        out_dir=out_dir,
        document_counter=document_counter
    )


def parse_section(div_id, header, pitch, headers, program_name_idx, program_type, soup, out_dir, document_counter):
    with open(os.path.join(out_dir, f"{document_counter:02}.txt"), "a") as f:
        write_to_file(f, header)
        write_to_file(f, pitch)
        program_div = soup.find(id=div_id)
        program_name = soup.find_all("h2", class_="block-title")[program_name_idx].text
        write_to_file(f, f"Program Type: {program_type}")
        write_to_file(f, f"Program Name: {program_name}")
        for header, section in zip(headers, program_div.find_all("div", class_="field-items")):
            write_to_file(f, f"{header}: {section.text.strip()}")
    document_counter += 1
    return document_counter


def write_to_file(f, text):
    f.write(text+"\n")


if __name__ == "__main__":
    parse_programs(out_dir="../documents/lti_programs")