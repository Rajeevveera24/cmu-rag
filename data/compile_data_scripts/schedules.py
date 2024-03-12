import os, json, csv
# import re
# import requests
# from bs4 import BeautifulSoup


def parse_schedule(urls, out_dir):
    document_counter = 0
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        for el in soup.find_all("b"):
            if el.parent and el.parent.name == "b" and "Semester: " in el.text:
                semester = el.text

        table = soup.find("table")
        trs = table.find_all("tr")
        headings = None
        current_category = None
        course_data = None
        last_section = None
        last_instructor = None
        for tr in trs:
            tds = tr.find_all("td")
            if len(tds) < 2: continue
            if not headings:
                headings = []
                for td in tds:
                    headings.append(td.text)
            elif tds[1].text.strip() == "" and tds[-1].text.strip() == "":
                current_category = tds[0].text
            else:
                with open(os.path.join(out_dir, f"{document_counter:02}.txt"), "w") as f:
                    if tds[0].text.strip() != "":
                        course_data = [tds[0].text.strip(), tds[1].text.strip(), tds[2].text.strip()]
                    write_to_file(f, semester)
                    write_to_file(f, f"Category: {current_category}")
                    for i in range(3):
                        if tds[i].text.strip() == "":
                            write_to_file(f, f"{headings[i]}: {course_data[i]}")
                        else:
                            write_to_file(f, f"{headings[i]}: {tds[i].text}")
                    for idx, (heading, td) in enumerate(zip(headings[3:], tds[3:])):
                        if idx == 0:
                            if td.text.strip() != "":
                                last_section = td.text.strip()
                                write_to_file(f, f"{heading}: {td.text}")
                            else:
                                write_to_file(f, f"{heading}: {last_section}")
                        elif idx == 6:
                            if td.text.strip() != "":
                                last_instructor = td.text.strip()
                                write_to_file(f, f"{heading}: {td.text}")
                            else:
                                write_to_file(f, f"{heading}: {last_instructor}")
                        else:
                            if len(tds) == 9 and idx == 5:
                                write_to_file(f, f"Instructor(s): {last_instructor}")
                            write_to_file(f, f"{heading}: {td.text}")
                document_counter += 1

def write_to_file(f, text):
    f.write(text+"\n")



def days(day_str):
    day_str = day_str.strip()
    if day_str == "MWF":
        return "Monday, Wednesday, Friday"
    elif day_str == "TR":
        return "Tuesday, Thursday"
    elif day_str == "MW":
        return "Monday, Wednesday"
    elif day_str == "F":
        return "Friday"
    elif day_str == "M":
        return "Monday"
    elif day_str == "T":
        return "Tuesday"
    elif day_str == "W":
        return "Wednesday"
    elif day_str == "R":
        return "Thursday"
    else:
        return day_str

if __name__ == "__main__":
    # parse_schedule(
    #     urls=[
    #         "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_spring.htm",
    #         "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_1.htm",
    #         "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_2.htm",
    #         "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_fall.htm"
    #     ], 
    #     out_dir="../documents/schedules"
    # )
    FILE_DIR = '/home/raj/Documents/cmu/sem2/anlp/assignments/cmu-rag/data/documents/Courses_CMU/schedules/'

    cnts = set()
    for file in sorted(os.listdir(FILE_DIR + 'raw')):
        with open(os.path.join(FILE_DIR + '/raw', file), 'r') as f:
            cnts.add(len(f.readlines()))
    # print(cnts)
    assert len(cnts) == 1
    print('All files have same number of lines')

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
    with open(FILE_DIR + 'csv/' + 'schedules.json', 'w') as f:
        json.dump(dct_list, f, indent=4)
    with open(FILE_DIR + 'txt/' + 'schedules.txt', 'w') as f:
        for dct in dct_list:
            assert len(dct.keys()) == 12
            write_str = dct["Instructor(s)"] + " teaches course number " + dct["Course"] + " titled " + dct["Title"] + " in semester " + dct["Semester"]
            write_str += " under category " + dct["Category"] + " with section " + dct["Lec/Sec"] + ". The course is of " + dct["Units"] + " units and is taught on days: "
            write_str += days(dct["Days"]) + " in " + dct["Bldg/Room"] + " located at " + dct["Location"] + ". " + "The course begins at " + dct["Begin"] + " and ends at " + dct["End"] + "." 
            f.write(write_str + "\n")
        

                
            


