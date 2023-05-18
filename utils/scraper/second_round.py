#%%
import pandas as pd
import lyricsgenius
import json
import time
# %%
df = pd.read_csv("data/billboard_song_ids.csv")
df['song_id'] = df['song_id'].astype(str)
df.to_csv("data/billboard_song_ids.csv",index=False)
# %%
token = 'PAAfUI-Awcje3MevYpru7rodySHbua1zIXoTJvNoe-68c6op2e7pOGjZZog6RxLf'
LyricsGenius = lyricsgenius.Genius(token,sleep_time=0.2, verbose=True)

for i,row in df.iterrows():
    if row['song_id'] == 'nan':
        try:
            song = LyricsGenius.search_song(title=row['title'],artist=row['band_singer'])  # search for lyrics
            if song:
                df.loc[i,'song_id'] = song.id
                fd = open("data/lyrics/{0}.json".format(str(song.id)),"w")
                json.dump(song.lyrics, fd)
                fd.close()
            else:
                df.loc[i,'song_id'] = None
        except:
            df.loc[i,'song_id'] = None
            print("Failed to search for song: {0} by {1}".format(row['title'],row['band_singer']))
            time.sleep(5)


df.to_csv("data/billboard_song_ids.csv",index=False)

# %%
df[df['song_id'] == 'nan'].count()
    
# %%
