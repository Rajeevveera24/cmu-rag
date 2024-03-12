import os, re, json
# import requests
# from bs4 import BeautifulSoup


# def parse_data(urls, out_dir):
#     document_counter = 0
#     for url in urls:
#         page = requests.get(url)
#         soup = BeautifulSoup(page.content, "html.parser")

#         header_text = soup.find(id="page-title").text

#         view_content = soup.find("div", class_="view-content")
#         els = view_content.find_all("td")
#         for el in els:
#             if el.find("div", class_="views-field-nothing") ==  None:
#                 continue
#             with open(os.path.join(out_dir, f"{document_counter:02}.txt"), "a") as f:
#                 write_to_file(f, header_text)
#                 write_to_file(f, f"Name: {el.find("div", class_="views-field-nothing").text.strip()}")
#                 if el.find("span", class_="label") and el.find("span", class_="label").parent.find("a"):
#                     write_to_file(f, f"Email: {el.find("span", class_="label").parent.find("a").text.strip()}")
#                 if el.find("div", class_="views-field-field-computed-building"):
#                     write_to_file(f, f"{re.sub(' +', ' ', el.find("div", class_="views-field-field-computed-building").text.strip())}")
#                 if el.find("div", class_="views-field-field-research-areas"):
#                     write_to_file(f, f"{re.sub(' +', ' ', el.find("div", class_="views-field-field-research-areas").text.strip())}")
#                 if el.find("div", class_="views-field-field-computed-phone"):
#                     write_to_file(f, f"{re.sub(' +', ' ', el.find("div", class_="views-field-field-computed-phone").text.strip())}")
#             document_counter += 1
#         print(f"Successfully parsed url {url}.")

# def write_to_file(f, text):
#     f.write(text+"\n")


if __name__ == "__main__":
    urls = [
        "https://lti.cs.cmu.edu/directory/all/154/1",
        "https://lti.cs.cmu.edu/directory/all/154/1?page=1",
        "https://lti.cs.cmu.edu/directory/all/154/2728",
        "https://lti.cs.cmu.edu/directory/all/154/200",
        "https://lti.cs.cmu.edu/directory/all/154/2",
        "https://lti.cs.cmu.edu/directory/all/154/2?page=1",
        "https://lti.cs.cmu.edu/directory/all/154/2731",
        "https://lti.cs.cmu.edu/directory/all/154/2731?page=1",
        "https://lti.cs.cmu.edu/directory/all/154/2730"
    ]
    # parse_data(urls=urls, out_dir="../documents/lti_faculty")
    FILE_DIR = '/home/raj/Documents/cmu/sem2/anlp/assignments/cmu-rag/data/documents/Faculty/lti_faculty/'

    cnts = set()
    for file in sorted(os.listdir(FILE_DIR + 'raw')):
        with open(os.path.join(FILE_DIR + '/raw', file), 'r') as f:
            cnts.add(len(f.readlines()))
    # print(cnts)
    # assert len(cnts) == 1
    # print('All files have same number of lines')

    dct_list = []
    for file in sorted(os.listdir(FILE_DIR+'raw/')):
        # print(file)
        with open(os.path.join(FILE_DIR+'/raw/', file), 'r') as f:
            lines = f.readlines()
            keys = [line.split(':')[0] for line in lines]
            assert len(keys) == len(set(keys))
            values = [''.join(line.split(':')[1:]).rstrip('\n') for line in lines]
            assert len(values) == len(keys)
            dct = {}
            for key, value in zip(keys, values):
                dct[key] = value
            dct_list.append(dct)
    with open(FILE_DIR + 'csv/' + 'lti_faculty.json', 'w') as f:
        json.dump(dct_list, f, indent=4)
    with open(FILE_DIR + 'txt/' + 'lti_faculty.txt', 'w') as f:
        for dct in dct_list:
            write_str = ''
            write_str += dct.get('Name', '') + "'s email is " + dct.get('Email', 'not provided')
            write_str += ' and phone number is ' + dct.get('Phone', 'not provided') + '. ' + dct.get('Name', '') + '\'s office is ' + dct.get('Office', 'unknown') + '. '
            write_str += dct.get('Name', '') + "'s research areas and interests are " + dct.get('Research Areas', 'none') + '.\n'
            f.write(write_str)
    print('Parsed lti faculty data')