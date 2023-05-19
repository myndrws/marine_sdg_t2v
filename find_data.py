# This script is for scraping the .gov site

import requests
import xmltodict
from bs4 import BeautifulSoup
import pickle
from datetime import date

rooturl = "https://www.gov.uk/search/all.atom?content_purpose_supergroup%5B%5D=research_and_statistics" \
          "&content_purpose_supergroup%5B%5D=policy_and_engagement&content_purpose_supergroup%5B%5D=transparency" \
          "&level_one_taxon=3cf97f69-84de-41ae-bc7b-7e2cc238fa58&public_timestamp%5Bfrom%5D=01%2F01%2F2021" \
          "&public_timestamp%5Bto%5D=31%2F12%2F2021"

# parsing
master_dict = {}
page = requests.get(rooturl)
page_count = 0

while page.status_code == 200:
    try:
        url = rooturl + f"&page={page_count}"
        page = requests.get(url)
        data_dict = xmltodict.parse(page.text)

        for i in range(len(data_dict['feed']['entry'])):
            master_dict[f'{page_count}_{i}'] = {
                'title': data_dict['feed']['entry'][i]['title'],
                'url': data_dict['feed']['entry'][i]['link']['@href'],
                'updated': data_dict['feed']['entry'][i]['updated']
            }
    except KeyError:
        break

    page_count += 1

# now we have master dict, time to inspect and draw in data from the urls
# this is not particularly fast, but it gets the job done.
successful_content_count = 0
unwanted_formats = ('.xls', '.ods', '.zip', '.xlsm', 'xlsx', 'odt')
for key, value in master_dict.items():
    try:
        on_page_dict = value['url']
        page = requests.get(on_page_dict)
        parsed_html = BeautifulSoup(page.text, features="lxml")
        found = parsed_html.body.find('div', attrs={'class': 'attachment-details'})
        children = found.find_all('a', href=True)

        for child in children:
            link = child['href']
            if not link.endswith(unwanted_formats) and \
                    not link.startswith('mailto:'):
                if link.startswith('/'):
                    link = "https://www.gov.uk" + link
                # add in error catching for basically if the link
                # doesn't work when we try to request, don't save.
                try:
                    attempt = requests.get(link)
                    if attempt.status_code != 200:
                        continue
                except:
                    continue
                value['first_content_url'] = link
                successful_content_count += 1
                print(successful_content_count)
                break
    except AttributeError:
        print(value['url'])

print(f"Successful content count: {successful_content_count} out of {len(master_dict)}")

filename = f"master_dict_{date.today()}.pkl"
print(filename)

with open(filename, 'wb') as f:
    pickle.dump(master_dict, f)

