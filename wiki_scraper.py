# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import json

#%%
def get_genre(soup):
    """
    Get the genre information from a song's Wikipedia page
    """
    genre_element = soup.find(text=re.compile('Genre'))

    if genre_element:
        genre_text = genre_element.find_next().text
        genres = [genre.strip().split('[')[0] for genre in re.split(r"\n|,", genre_text) if genre]
    else:
        genres = []

    # genres = []
    # ul_element = genre_element.find_next('ul')
    # for li_element in ul_element.find_all('li'):
    #     a_element = li_element.find('a')
    #     if a_element:
    #         genres.append(a_element.get('title') or a_element.text)

    return genres

def parse_year(the_year, yeartext_dict):
    soup = BeautifulSoup(yeartext_dict[str(the_year)], 'html.parser')
    tables_wikitable = soup.find_all('table', 'wikitable')
    rows = [row for row in tables_wikitable[0].find_all('tr')][1:]
    yearinfo = [get_single_dict(row) for row in rows]
    return yearinfo

def get_single_dict(row):
    children = [child for child in row.children]
    children = list(filter(lambda x: x != '\n', children))
    ranking = children[0].string
    band_singers = children[2].find_all('a')
    band_singer = [band.string for band in band_singers]
    url = [url['href'] for url in band_singers]
    songs = children[1].find_all('a')
    songurl = [song['href'] for song in songs]

    if songurl == []:
        songurl = [None]
    song = [song.string for song in songs]
    if not song:
        song = children[1].string

    if type(song) == list:
        title = '/'.join(str(s) for s in song)
    else:
        title = song

    # get the genre information from the song's Wikipedia page
    if songurl[0]:
        page = requests.get('https://en.wikipedia.org' + songurl[0])
        song_soup = BeautifulSoup(page.content, 'html.parser')
        genre = get_genre(song_soup)
        # print(f"{song[0]}: {genre}")
    else:
        genre = []

    single_dict = {'band_singer': band_singer, 'ranking': ranking, 'song': song,
                   'song_url':songurl, 'title':title, 'singer_url': url, 'genre': genre}
    return single_dict

#%%
# generate list of urls
urls = ['http://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{0}'.format(str(i)) for i in range(1959, 2023)]


# download text for all years
yearstext = {}
for url in urls:
    req = requests.get(url)
    yearstext[url.split('_')[-1]] = req.text
    time.sleep(1)

#%% 
# create JSON file for each year's Billboard Hot 100s: 1959-2023
for year in range(1959, 2023):
    print("Extracting year {0}".format(str(year)))
    yearinfo = parse_year(year, yearstext)
    fd = open("data/year_info/{0}info.json".format(str(year)),"w")
    json.dump(yearinfo,fd)
    fd.close()
    print("Finished extracting year {0}".format(str(year)))


# %%
#create DataFrame for Billboard Hot 100's: 1959-2023
flatframe = pd.DataFrame(columns=['band_singer', 'ranking', 'song', 'song_url', 'title', 'singer_url', 'genre','year'])
for year in range(1959,2023):
    with open("data/year_info/{0}info.json".format(str(year)), "r") as f:
        curyearinfo = json.load(f)
        year_df = pd.DataFrame(curyearinfo)
        year_df['year'] = year
        flatframe = pd.concat([flatframe, year_df], axis=0, ignore_index=True,copy=True)


flatframe['ranking'] = flatframe['ranking'].replace('Tie', 100)
flatframe['ranking'] = flatframe['ranking'].astype(int)


def extract_substring(x):
    match = re.search(r"'([^']+)'", str(x))
    if match:
        return match.group(1)
    else:
        return x


flatframe[['band_singer','song']] = flatframe[['band_singer','song']].applymap(extract_substring)
flatframe = flatframe.drop(columns=['song_url','singer_url'])

flatframe.to_csv('data/billboard.csv', index=False)
# %%
