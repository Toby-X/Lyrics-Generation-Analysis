# %%
import pandas as pd
# %%
billboard_lyrics = pd.read_csv("data/billboard_top10_song_ids_0.csv")

billboard_top10 = pd.DataFrame(columns=billboard_lyrics.columns)
top10_lyrics = pd.DataFrame()
for i in range(30):
    df = pd.read_csv(f"data/billboard_top10_song_ids_{i}.csv")
    lyrics = pd.read_csv(f"data/top10_lyrics_{i}.csv")
    billboard_top10 = pd.concat([billboard_top10, df])
    top10_lyrics = pd.concat([top10_lyrics, lyrics])

billboard_top10.to_csv('data/billboard_top10_song_ids_withnull.csv')

billboard_top10_lyrics = pd.merge(billboard_top10, top10_lyrics, on='song_id')

billboard_top10_lyrics = billboard_top10_lyrics.drop(columns=['index'])


billboard_top10_lyrics.dropna(inplace=True)

billboard_top10_lyrics['song_id'] = billboard_top10_lyrics['song_id'].astype(
    int)

billboard_top10_lyrics.to_csv('data/billboard_top10_lyrics.csv', index=False)
# %%
billboard_lyrics = pd.read_csv("data/billboard_all_lyrics.csv")
billboard_genres = pd.read_csv("data/billboard.csv")
billboard_genres['genre'] = billboard_genres['genre']
# %%
lyrics_genres = pd.merge(billboard_lyrics, billboard_genres[['band_singer', 'title', 'genre']], on=[
                         'band_singer', 'title'], how='inner').drop_duplicates(subset=['band_singer', 'title'])
# %%
lyrics_genres.to_csv("data/billboard_lyrics_genres.csv", index=False)
# %%
