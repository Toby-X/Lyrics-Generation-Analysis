# %%
import requests
import re
from bs4 import BeautifulSoup
# %%
url = 'https://www.azlyrics.com/lyrics/shayneward/icry.html'
url = 'https://www.azlyrics.com/lyrics/taylorswift/lovestory.html'
response = requests.get(url=url)
print(response.status_code)
# %%
html_soup = BeautifulSoup(response.content, 'html.parser')
singer = html_soup.find('div', class_='lyricsh').h2.text
song_name = html_soup.find('div', class_='col-xs-12 col-lg-8 text-center').find_all(
    'div', class_='div-share')[1].text.split('"')[1]
lyrics = html_soup.find(
    'div', class_='col-xs-12 col-lg-8 text-center').find_all('div')[5].text

print('Singer Name -> {}'.format(singer))
print('Song Name -> {}'.format(song_name))
print('Lyrics is -> {}'.format(lyrics))
# %%
print(response.links)

# %%
