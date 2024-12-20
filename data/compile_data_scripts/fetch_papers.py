# from semanticscholar import SemanticScholar
# from print_schema import print_schema
# import requests
import os, json


# lti_faculty_map = {
#     "Yonatan Bisk": ["3312309"],
#     "Ralf D. Brown": ["2109449533"],
#     "Jamie Callan": ["144987107", "17038253"],
#     "Justine Cassell": ["145431806", "145431807", "2065308530", "2256351350"],
#     "Mona T. Diab": ["1700007", "2138579860", "2250935476"],
#     "Fernando Diaz": ["145472333"],
#     "S. Fahlman": ["1758714"],
#     "R. Frederking": ["2260563"],
#     "Daniel Fried": ["47070750", "2261278355"],
#     "Anatole Gershman": ["145001267", "2099699170"],
#     "Alexander Hauptmann": ["7661726", "145788702", "2257210215", "2257000091", "121653201"],
#     "Daphne Ippolito": ["7975935"],
#     "Lori Levin": ["1686960", "145585627", "1736785962", "2263388984"],
#     "Lei Li": ["143900005"],
#     "Teruko Mitamura": ["1706595", "2240346320", "2259321079", "2067218320"],
#     "Louis-Philippe Morency": ["49933077", "2065275646"],
#     "David Mortensen": ["3407646"],
#     "Graham Neubig": ["1700325", "2265547593"],
#     "Eric Nyberg": ["144287919", "46841006"],
#     "Kemal Oflazer": ["1723120", "2250930714"],
#     "Bhiksha Ramakrishnan": ["2070312757", "1880336"],
#     "Carolyn Rose": ["35959897", "2256564956", "2243225250", "2053634036", "2261985726", "11149490"],
#     "Alexander Rudnicky": ["1783635", "3156164", "2257301543"],
#     "Maarten Sap": ["2729164"],
#     "Michael Shamos": ["1890127"],
#     "Rita Singh": ["153915824"],
#     "Emma Strubell": ["2268272"],
#     "Alexander Waibel": ["1724972", "2064429921", "2271781054", "2257374937"],
#     "Shinji Watanabe": ["1746678", "2187876006"],
#     "Sean Welleck": ["2129663"],
#     "Eric P. Xing": ["143977260", "2251052375", "2064963077", "2246852356", "2243336934", "2243234805", "2238075244"],
#     "Chenyan Xiong": ["144628574", "2139787803"],
#     "Yiming Yang": ["35729970", "46286308"]
# }

# def get_papers(out_dir, raw_out_dir):
#     sch = SemanticScholar()
#     all_papers = []
#     for author_str, author_ids in lti_faculty_map.items():
#         for author_id in author_ids:
#             author = sch.get_author(author_id)
#             papers = author.papers
#             filtered_papers = []
#             for paper in papers:
#                 if paper.isOpenAccess and paper.year == 2023:
#                     filtered_papers.append(paper)
#             print(f"Filtered papers for {author.name} from {len(papers)} to {len(filtered_papers)}.")
#             for paper in filtered_papers:
#                 all_papers.append(paper)
#         #     break
#         # break
#             # print_schema(dict(papers[0]), indent=3, dense=False)
#     for idx, paper in enumerate(all_papers):
#         # Save metadata
#         print(f"Saving metadata {idx:02}.txt.")
#         with open(os.path.join(out_dir, f"{idx:02}.txt"), "w") as f:
#             if paper.title:
#                 write_to_file(f, f"Title: {paper.title}")
#             else:
#                 write_to_file(f, f"Title: Not Available")
#             if paper.abstract:
#                 write_to_file(f, f"Abstract: {paper.abstract}")
#             else:
#                 write_to_file(f, f"Abstract: Not Available")
#             if paper.authors:
#                 write_to_file(f, f"Authors: {", ".join([a['name'] for a in paper.authors])}")
#             else:
#                 write_to_file(f, f"Authors: Not Available")
#             if paper.publicationVenue:
#                 publication_venues = []
#                 try:
#                     publication_venues.append(paper.publicationVenue['name'])
#                 except:
#                     pass
#                 try:
#                     publication_venues += paper.publicationVenue['alternate_names']
#                 except:
#                     pass
#                 if len(publication_venues) > 0:
#                     write_to_file(f, f"Publication Venue: {", ".join(publication_venues)}")
#                 else:
#                     write_to_file(f, f"Publication Venue: Not Available")
#             else:
#                 write_to_file(f, f"Publication Venue: Not Available")
#             if paper.year:
#                 write_to_file(f, f"Year of Publication: {paper.year}")
#             else:
#                 write_to_file(f, f"Year of Publication: Not Available")
#             if paper.tldr:
#                 write_to_file(f, f"Summary: {paper.tldr}")
#             else:
#                 write_to_file(f, f"Summary: Not Available")
#         # Save pdf
#         try:
#             print(f"Saving paper {paper.openAccessPdf['url']}.")
#             with open(os.path.join(raw_out_dir, f"{idx:02}.pdf"), "wb") as f:
#                 response = requests.get(paper.openAccessPdf['url'])
#                 f.write(response.content)
#         except:
#             print(f"Unable to save paper for {idx:02}.txt.")
#     print(f"{len(all_papers)} total papers.")

# def write_to_file(f, text):
#     f.write(text+"\n")


if __name__ == "__main__":
    # get_papers(
    #     out_dir="../documents/lti_papers_metadata", 
    #     raw_out_dir="../raw_data/lti_papers"
    # )
    FILE_DIR = '/home/raj/Documents/cmu/sem2/anlp/assignments/cmu-rag/data/documents/Acads_LTI/lti_papers_metadata'

    dct_list = []
    for file in sorted(os.listdir(FILE_DIR+'/raw/')):
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
    with open(FILE_DIR + '/csv/' + 'schedules.json', 'w') as f:
        json.dump(dct_list, f, indent=4)
