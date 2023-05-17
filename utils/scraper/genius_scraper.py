#%%
import lyricsgenius
import pandas as pd
import numpy as np
import json
import time
import random

#%%
token = 'PAAfUI-Awcje3MevYpru7rodySHbua1zIXoTJvNoe-68c6op2e7pOGjZZog6RxLf'
LyricsGenius = lyricsgenius.Genius(token,sleep_time=0.2, verbose=True)

df = pd.read_csv("data/billboard.csv")
df = df[['ranking','band_singer','title','year']]

def reduce(x):
    contents = x.split("'")
    return contents[1] if len(contents) > 1 else None

df['band_singer'] = df['band_singer'].apply(reduce)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
df['index'] = df.index
df['song_id'] = pd.DataFrame(range(len(df)))

df['ranking'] = df['ranking'].astype(int)

#%%
dfs = np.array_split(df,100)  # split into 30 dfs

#%%
song_ids = []
all_lyrics = []

for j,mini_df in enumerate(dfs):
        time.sleep(0)
        song_ids_ = []
        lyrics_ = []
        for i,row in mini_df.iterrows():
            try:
                song = LyricsGenius.search_song(title=row['title'],artist=row['band_singer'])  # search for lyrics

                if song:
                    df.loc[row['index'],'song_id'] = song.id  # add lyrics to df
                    song_ids_.append(song.id)
                    song_ids.append(song.id)
                    all_lyrics.append(song.lyrics)
                    lyrics_.append(song.lyrics)

                    # save lyrics to json file
                    fd = open("data/lyrics/{0}.json".format(str(song.id)),"w")
                    json.dump(song.lyrics, fd)
                    fd.close()

                else:
                    df.loc[row['index'],'song_id'] = None
                    song_ids_.append(song.id)
                    song_ids.append(song.id)
                    all_lyrics.append(None)
                    lyrics_.append(None)

            except:
                print("Failed to search for song: {0} by {1}".format(row['title'],row['band_singer']))
                df.loc[row['index'],'song_id'] = None
                song_ids_.append(None)
                song_ids.append(None)
                all_lyrics.append(None)
                lyrics_.append(None)
                time.sleep(3)
                

        mini_df['song_id'] = song_ids_
        mini_df.to_csv(f'data/billboard_song_ids_{j}.csv',index=False)
        lyrics_df = pd.DataFrame({'song_id':song_ids_,'lyrics':lyrics_})
        lyrics_df.to_csv(f'data/lyrics_{j}.csv',index=False)


df['song_id'] = song_ids
df.to_csv('data/billboard_song_ids.csv',index=False)
lyrics_df.to_csv('data/lyrics.csv',index=False)

# %%
all_lyrics = pd.DataFrame(columns=['song_id','lyrics'])
for i in range(100):
    df = pd.read_csv(f'data/lyrics_{i}.csv')
    all_lyrics = pd.concat([all_lyrics,df],axis=0)
    
all_lyrics.dropna().to_csv('data/lyrics_nonull.csv',index=False)

billboard_all_lyrics = pd.read_csv('data/billboard_song_ids.csv').merge(all_lyrics,how='left',on='song_id').dropna().drop(columns=['index'])
billboard_all_lyrics['song_id'] = billboard_all_lyrics['song_id'].astype(int)
billboard_all_lyrics.to_csv('data/billboard_all_lyrics.csv',index=False)
# %%
