
import requests
from bs4 import BeautifulSoup
import time


url = 'https://www.kurdfonts.com/browse/categories'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')


ul = soup.find('ul', {'class': 'list-group'})


for page in ul.find_all('a'):
    soup = BeautifulSoup(requests.get(page.get('href')).content, 'html.parser')
    for link in soup.find_all('a'):
        if link.get('class') is not None and 'btn-success' in link.get('class'):
            print(link.get('href'))
            resp = requests.get(link.get('href'))
            soup = BeautifulSoup(resp.content, 'html.parser')
            link = soup.select('a[href^="https://www.kurdfonts.com/dl/"]')
            # print(link[0].get('href'))
            
            print("Downloading: " + link[0].get('href'))

            resp = requests.get(link[0].get('href'))
            # write font file to disk
            with open("./data/fonts" +link[0].get('href').split('/')[-1], 'wb') as f:
                f.write(resp.content)
            time.sleep(1)
            print("Downloaded: " + link[0].get('href').split('/')[-1])

