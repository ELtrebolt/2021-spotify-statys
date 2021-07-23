import datetime
import pandas as pd

PERCENTILE_COLS = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']

def _get_100_songs(session, tracks, playlist):
    
    song_meta={'id':[], 'name':[], 
            'artist':[], 'album':[], 'explicit':[],'popularity':[], 
            'playlist':[], 'date_added':[], 'artist_ids':[]}
    
    i = 1
    for item in tracks['items']:
        meta = item['track']
        
        song_meta['id'] += [meta['id']]

        # song name
        song=meta['name']
        song_meta['name']+=[song]
        
        # artists name
        artist= ', '.join([singer['name'] for singer in meta['artists']])
        song_meta['artist'].append(artist)

        # artists id
        artist_ids = ', '.join([singer['id'] for singer in meta['artists']])
        song_meta['artist_ids'].append(artist_ids)
        
        # album name
        album=meta['album']['name']
        song_meta['album']+=[album]

        # explicit: lyrics could be considered offensive or unsuitable for children
        explicit=meta['explicit']
        song_meta['explicit'].append(explicit)

        # song popularity
        popularity=meta['popularity']
        song_meta['popularity'].append(popularity)
        
        # playlist name
        song_meta['playlist'].append(playlist)
        
        # date added to playlist
        d1 = datetime.datetime.strptime(item['added_at'], "%Y-%m-%dT%H:%M:%SZ")
        new_format = "%Y-%m-%d"
        date_added = d1.strftime(new_format)
        song_meta['date_added'].append(date_added)

        # artist genres
        # artist_ids = [a['id'] for a in meta['artists']]
        # genres = [SPOTIFY.artist(i)['genres'] for i in artist_ids]
        # song_meta['genres'].append(genres)
        
        i += 1
        
    song_meta_df=pd.DataFrame.from_dict(song_meta)
    
    # check the song feature
    features = session['SPOTIFY'].audio_features(song_meta['id'])
    # change dictionary to dataframe
    features_df=pd.DataFrame.from_dict(features)

    # convert milliseconds to mins
    # duration_ms: The duration of the track in milliseconds.
    # 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
    features_df['duration']=features_df['duration_ms']/1000
    features_df.drop(columns='duration_ms', inplace=True)

    # combine two dataframe
    final_df=song_meta_df.merge(features_df)
    
    return final_df

def _get_playlist(session, name, _id):
    SPOTIFY = session['SPOTIFY']
    df = pd.DataFrame()

    results = SPOTIFY.playlist(_id)
    tracks = results['tracks']
    df = pd.concat([df, _get_100_songs(session, tracks, name)])
    
    while tracks['next']:
        tracks = SPOTIFY.next(tracks)
        df = pd.concat([df, _get_100_songs(session, tracks, name)])
        
    return df.reset_index()

def setup_data(session):
    SPOTIFY = session['SPOTIFY']

    PLAYLISTS = SPOTIFY.current_user_playlists()
    USER_ID = SPOTIFY.me()['id']

    PLAYLIST_DICT = {}
    for playlist in PLAYLISTS['items']:
        if playlist['owner']['id'] == USER_ID:
            PLAYLIST_DICT[playlist['name']] = playlist['id']

    ALL_SONGS_DF = pd.DataFrame()
    for name, _id in PLAYLIST_DICT.items():
        df = _get_playlist(session, name, _id)
        ALL_SONGS_DF = pd.concat([ALL_SONGS_DF, df])
    ALL_SONGS_DF.drop(columns='index',inplace=True)

    # Changing cols will be condensed into a list = ex: unique song will have a col "playlist" = [playlist1, playlist2]
    changing_cols = ['playlist', 'date_added']
    UNIQUE_SONGS_DF = ALL_SONGS_DF.groupby([i for i in ALL_SONGS_DF.columns if i not in changing_cols], as_index=False)\
        [['playlist', 'date_added']].agg(lambda x: list(x))
    for col in PERCENTILE_COLS:
        sz = UNIQUE_SONGS_DF[col].size-1
        UNIQUE_SONGS_DF[col + '_percentile'] = UNIQUE_SONGS_DF[col].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)

    #SONGS_TIMELINE_DF = ALL_SONGS_DF[['name', 'playlist', 'date_added', 'artist']].groupby(['date_added'], as_index=False)\
    #    [['name', 'playlist', 'artist']].agg(lambda x: list(x))

    TOP_ARTISTS_SHORT = [i['name'] for i in SPOTIFY.current_user_top_artists(time_range='short_term', limit = 50)['items']]
    TOP_ARTISTS_MED = [i['name'] for i in SPOTIFY.current_user_top_artists(time_range='medium_term', limit = 50)['items']]
    TOP_ARTISTS_LONG = [i['name'] for i in SPOTIFY.current_user_top_artists(time_range='long_term', limit = 50)['items']]
    TOP_ARTISTS = [TOP_ARTISTS_SHORT, TOP_ARTISTS_MED, TOP_ARTISTS_LONG]

    for top_list, col_name in zip(TOP_ARTISTS, ['artists_short_rank', 'artists_med_rank', 'artists_long_rank']):
        new_list = []
        for i in UNIQUE_SONGS_DF['artist']:
            if any(item in top_list for item in i.split(', ')):
                rank = []
                for item in i.split(', '):
                    if item in top_list:
                        rank.append(top_list.index(item)+1)
                    else:
                        rank.append('N/A')
                rank = ', '.join([str(i) for i in rank]) if len(rank) > 1 else str(rank[0])
            else:
                rank = 'N/A'
            new_list.append(rank)
        UNIQUE_SONGS_DF.insert(len(UNIQUE_SONGS_DF.columns), col_name, new_list, True)

    TOP_SONGS_SHORT = {i['name']:(', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(SPOTIFY.current_user_top_tracks(time_range='short_term', limit=50)['items'])}
    TOP_SONGS_MED = {i['name']:(', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(SPOTIFY.current_user_top_tracks(time_range='medium_term', limit=50)['items'])}
    TOP_SONGS_LONG = {i['name']:(', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(SPOTIFY.current_user_top_tracks(time_range='long_term', limit=50)['items'])}
    TOP_SONGS = [TOP_SONGS_SHORT, TOP_SONGS_MED, TOP_SONGS_LONG]

    for top_dict, col_name in zip(TOP_SONGS, ['songs_short_rank', 'songs_med_rank', 'songs_long_rank']):
        new_list = []
        for i in UNIQUE_SONGS_DF.index:
            song = UNIQUE_SONGS_DF['name'][i]
            if song in top_dict.keys() and UNIQUE_SONGS_DF['artist'][i] == top_dict[song][0]:
                rank = top_dict[song][1]
            else:
                rank = 'N/A'
            new_list.append(rank)
        UNIQUE_SONGS_DF.insert(len(UNIQUE_SONGS_DF.columns), col_name, new_list, True)

    UNIQUE_SONGS_DF['num_playlists'] = [len(i) for i in UNIQUE_SONGS_DF['playlist']]

    # Get Genres for Each Song By Artist Genres - Takes 2 minutes for 1000 unique artists
    unique_artist_ids = set()
    for i in UNIQUE_SONGS_DF['artist_ids']:
        unique_artist_ids = unique_artist_ids | set(i.split(', '))

    # Save artist genres from Spotify API all at once using dict(), then make new cols for dataframes
    genres_dict = dict()
    total = len(unique_artist_ids)
    unique_artist_ids = list(unique_artist_ids)
    
    # Getting 50 artists at a time is a LOT faster than getting 1 artist at a time
    for i in range(0, total, 40):
        listy = session['SP'].artists(unique_artist_ids[i:i+40])
        for a in listy['artists']:
            try:
                g = a['genres']
            except:
                g = []
            if g:
                genres_dict[a['id']] = g
            else:
                genres_dict[a['id']] = ['N/A']

    for df in [ALL_SONGS_DF, UNIQUE_SONGS_DF]:
        genres_list = []
        for i in df['artist_ids']:
            song_genres = []
            for j in i.split(', '):
                song_genres.append(genres_dict[j])
            genres_list.append(song_genres)
        df['genres']=genres_list

    session['PLAYLISTS'] = PLAYLISTS
    session['USER_ID'] = USER_ID
    session['PLAYLIST_DICT'] = PLAYLIST_DICT
    session['ALL_SONGS_DF'] = ALL_SONGS_DF
    session['UNIQUE_SONGS_DF'] = UNIQUE_SONGS_DF
    session['TOP_ARTISTS'] = TOP_ARTISTS
    session['TOP_SONGS'] = TOP_SONGS      