# SetupData.py = Class for Managing creation of DataFrames from SpotifyAPI
from datetime import datetime
import pandas as pd
import pickle
import os
from visualization import HomePage, AboutPage, Top50Page, MyPlaylistsPage
import traceback

# Test Local
# REDIRECT_URI = 'http://127.0.0.1:5000/'
# Run Heroku
REDIRECT_URI = 'https://spotify-statys.herokuapp.com/'
PERCENTILE_COLS = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']

def _dump(path, obj):
    with open(path, 'wb') as f:   # Pickling
        pickle.dump(obj, f)

def _load(path):
    with open(path, 'rb') as f:  # Unpickling
        return pickle.load(f)

class SetupData():
    def __init__(self, session):
        self.SPOTIFY = session['SPOTIFY']
        self.SP = session['SP']

        self.USER_ID = self.SPOTIFY.me()['id']
        self.PLAYLIST_DICT = self._get_all_playlists_dict()

        self.path = f'.data/{self.USER_ID}/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        status = {'SETUP1': False, 'SETUP2': False, 'SETUP3': False}
        _dump(f'{self.path}collection.pkl', status)


    def _get_50_playlist_dict(self, playlists):
        dicty = {}
        for playlist in playlists:
            if playlist['owner']['id'] == self.USER_ID:
                dicty[playlist['name']] = playlist['id']
        return dicty
    

    def _get_all_playlists_dict(self):
        dicty = {}

        results = self.SPOTIFY.current_user_playlists()
        items = results['items']
        dicty = {**dicty, **self._get_50_playlist_dict(items)}
        total = 50

        off = 0
        while total < results['total']:
            off += 50
            items = self.SPOTIFY.current_user_playlists(offset=off)['items']
            dicty = {**dicty, **self._get_50_playlist_dict(items)}
            total += 50

        return dicty


    # Tracks can only get 100 items at a time
    def _get_100_songs(self, tracks, playlist):

        # Empty Playlist
        if len(tracks['items']) == 0:
            return pd.DataFrame()

        song_meta = {'id': [], 'name': [],
                     'artist': [], 'album': [], 'explicit': [], 'popularity': [],
                     'playlist': [], 'date_added': [], 'artist_ids': []}

        for item in tracks['items']:
            meta = item['track']

            # For Bernardo and Adam, this was the error - meta was None
            # For Beni meta['id'] was None due to Kanye West leaked unrelease - Spread Your Wings & Flowers
            if meta is not None and meta['id'] is not None:

                song_meta['id'].append(meta['id'])

                song = meta['name']
                song_meta['name'].append(song)

                artist = ', '.join([singer['name']
                                   for singer in meta['artists']])
                song_meta['artist'].append(artist)

                artist_ids = ', '.join(filter(None, [singer['id'] if singer else ''
                                       for singer in meta['artists']]))
                song_meta['artist_ids'].append(artist_ids)

                album = meta['album']['name']
                song_meta['album'].append(album)

                explicit = meta['explicit']
                song_meta['explicit'].append(1 if explicit == True else 0)

                popularity = meta['popularity']
                song_meta['popularity'].append(popularity)

                song_meta['playlist'].append(playlist)

                # date added to playlist
                d1 = datetime.strptime(item['added_at'], '%Y-%m-%dT%H:%M:%SZ')
                # for whatever reason converting timezone doesn't show same date added as Spotify
                date_added = d1.strftime('%Y-%m-%d')
                song_meta['date_added'].append(date_added)

        song_meta_df = pd.DataFrame.from_dict(song_meta)

        features = self.SP.audio_features(song_meta['id'])
        # Song Features = None for podcast episodes like Greendale Is Where I Belong
        if None in features:
            features2 = []
            for i,j in enumerate(features):
                if j is None:
                    song_meta_df.drop([i], inplace=True)
                else:
                    features2.append(j)
        else:
            features2 = features
        features_df = pd.DataFrame.from_dict(features2)

        # convert milliseconds to mins
        # duration_ms: The duration of the track in milliseconds.
        # 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
        features_df['duration'] = features_df['duration_ms']/1000
        features_df.drop(columns='duration_ms', inplace=True)

        final_df = song_meta_df.merge(features_df)

        return final_df


    # Get all of a playlists songs
    def _get_playlist(self, name, _id):
        df = pd.DataFrame()

        try:
            results = self.SPOTIFY.playlist(_id, fields='tracks,next')
        except:
            return 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'
        
        tracks = results['tracks']
        df = pd.concat([df, self._get_100_songs(tracks, name)])

        while tracks['next']:
            tracks = self.SPOTIFY.next(tracks)
            df = pd.concat([df, self._get_100_songs(tracks, name)])

        return df.reset_index()


    # Get all User Playlists songs
    def setup_1(self):
        try:
            count = 1
            total = len(self.PLAYLIST_DICT)
            yield 'data:<h1>Collecting Your Playlists</h1>\n\n'

            ALL_SONGS_DF = pd.DataFrame()
            for name, _id in list(self.PLAYLIST_DICT.items()):
                df = self._get_playlist(name, _id)
                if type(df) == str:
                    yield df
                else:
                    ALL_SONGS_DF = pd.concat([ALL_SONGS_DF, df])
                    yield 'data:' + name + '   ' + str(count) + '/' + str(total) + '<br/>\n\n\n'
                count += 1

            # Yung Yi had the problem of 'index' not found in axis
            if 'index' in ALL_SONGS_DF.columns:
                ALL_SONGS_DF.drop(columns='index', inplace=True)
            ALL_SONGS_DF.to_pickle(f"{self.path}all_songs_df.pkl")

            status = {'SETUP1': True, 'SETUP2': False, 'SETUP3': False}
            _dump(f'{self.path}collection.pkl', status)

            yield 'data:REDIRECT_URI=' + REDIRECT_URI + '\n\n'
        except Exception as e:
            yield 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'


    def _get_unique_songs_df(self):
        # Changing cols will be condensed into a list = ex: unique song will have a col "playlist" = [playlist1, playlist2]
        ALL_SONGS_DF = pd.read_pickle(f'{self.path}all_songs_df.pkl')

        changing_cols = ['id', 'playlist', 'date_added']
        UNIQUE_SONGS_DF = ALL_SONGS_DF.groupby([i for i in ALL_SONGS_DF.columns if i not in changing_cols], as_index=False)[
            changing_cols].agg(lambda x: list(x))
        UNIQUE_SONGS_DF.drop_duplicates(
            subset=['name', 'artist'], inplace=True)

        for col in PERCENTILE_COLS:
            sz = UNIQUE_SONGS_DF[col].size-1
            UNIQUE_SONGS_DF[col + '_percentile'] = UNIQUE_SONGS_DF[col].rank(
                method='max').apply(lambda x: 100.0*(x-1)/sz)

        UNIQUE_SONGS_DF['num_playlists'] = [
            len(i) for i in UNIQUE_SONGS_DF['playlist']]

        UNIQUE_SONGS_DF.to_pickle(f"{self.path}unique_songs_df.pkl")


    def _get_top_artists(self):
        # [Artist1, Artist2, Artist3]
        TOP_ARTISTS_SHORT = [i['name'] for i in self.SPOTIFY.current_user_top_artists(
            time_range='short_term', limit=50)['items']]
        TOP_ARTISTS_MED = [i['name'] for i in self.SPOTIFY.current_user_top_artists(
            time_range='medium_term', limit=50)['items']]
        TOP_ARTISTS_LONG = [i['name'] for i in self.SPOTIFY.current_user_top_artists(
            time_range='long_term', limit=50)['items']]
        TOP_ARTISTS = [TOP_ARTISTS_SHORT, TOP_ARTISTS_MED, TOP_ARTISTS_LONG]

        _dump(f'{self.path}top_artists.pkl', TOP_ARTISTS)


    # INT or "N/A" if single, STR separated by commas if multiple
    def _add_top_artists_rank(self):
        UNIQUE_SONGS_DF = pd.read_pickle(f'{self.path}unique_songs_df.pkl')
        TOP_ARTISTS = _load(f'{self.path}top_artists.pkl')

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
                    rank = ', '.join([str(i) for i in rank]) if len(
                        rank) > 1 else rank[0]
                else:
                    rank = 'N/A'
                new_list.append(rank)
            UNIQUE_SONGS_DF[col_name] = new_list
        UNIQUE_SONGS_DF.to_pickle(f"{self.path}unique_songs_df.pkl")


    def _get_top_songs(self):
        # {Song: ('Artists Separated By Commas', Rank)}
        TOP_SONGS_SHORT = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
            self.SPOTIFY.current_user_top_tracks(time_range='short_term', limit=50)['items'])}
        TOP_SONGS_MED = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
            self.SPOTIFY.current_user_top_tracks(time_range='medium_term', limit=50)['items'])}
        TOP_SONGS_LONG = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
            self.SPOTIFY.current_user_top_tracks(time_range='long_term', limit=50)['items'])}
        TOP_SONGS = [TOP_SONGS_SHORT, TOP_SONGS_MED, TOP_SONGS_LONG]

        _dump(f'{self.path}top_songs.pkl', TOP_SONGS)


    def _add_top_songs_rank(self):
        UNIQUE_SONGS_DF = pd.read_pickle(f'{self.path}unique_songs_df.pkl')
        TOP_SONGS = _load(f'{self.path}top_songs.pkl')

        for top_dict, col_name in zip(TOP_SONGS, ['songs_short_rank', 'songs_med_rank', 'songs_long_rank']):
            new_list = []
            for i in UNIQUE_SONGS_DF.index:
                song = UNIQUE_SONGS_DF.loc[i, 'name']
                if song in top_dict.keys() and UNIQUE_SONGS_DF.loc[i, 'artist'] == top_dict[song][0]:
                    rank = top_dict[song][1]
                else:
                    rank = 'N/A'
                new_list.append(rank)
            UNIQUE_SONGS_DF[col_name] = new_list
        UNIQUE_SONGS_DF.to_pickle(f"{self.path}unique_songs_df.pkl")


    def _add_genres(self):
        ALL_SONGS_DF = pd.read_pickle(f'{self.path}all_songs_df.pkl')
        UNIQUE_SONGS_DF = pd.read_pickle(f'{self.path}unique_songs_df.pkl')

        # Get Genres for Each Song By Artist Genres - Takes 2 minutes for 1000 unique artists
        unique_artist_ids = set()
        for i in UNIQUE_SONGS_DF['artist_ids']:
            unique_artist_ids = unique_artist_ids | set(i.split(', '))
        # if '' in unique_artist_ids:
        #     unique_artist_ids.remove('')

        # Save artist genres from Spotify API all at once using dict(), then make new cols for dataframes
        # {artist1: genres_list, artist2: ['N/A']}
        genres_dict = dict()
        total = len(unique_artist_ids)
        unique_artist_ids = list(unique_artist_ids)

        # Getting 50 artists at a time is a LOT faster than getting 1 artist at a time
        for i in range(0, total, 50):
            listy = self.SP.artists(unique_artist_ids[i:i+50])
            for a in listy['artists']:
                try:
                    g = a['genres']
                except:
                    g = []
                if g:
                    genres_dict[a['id']] = g
                else:
                    genres_dict[a['id']] = ['N/A']

        # Populate Dataframes
        for df in [ALL_SONGS_DF, UNIQUE_SONGS_DF]:
            genres_list = []
            for i in df['artist_ids']:
                song_genres = []
                for j in i.split(', '):
                    # try:
                    song_genres.append(genres_dict[j])
                    # except Exception as e:
                    #    print(j, e)
                    #    yield f'data:No Artist Id Found{j}<br>\n\n\n'
                    #    song_genres.append('N/A')
                genres_list.append(song_genres)
            df['genres'] = genres_list

        ALL_SONGS_DF.to_pickle(f"{self.path}all_songs_df.pkl")
        UNIQUE_SONGS_DF.to_pickle(f"{self.path}unique_songs_df.pkl")


    # Group Data with Top Artists/Songs, Genres, and setup Pages
    def setup_2(self, ALL_SONGS_DF):
        total = 11
        try:
            yield 'data:<h1>Grouping Your Data</h1>\n\n'

            yield f'data:Getting Unique Songs...1/{total}<br>\n\n\n'
            self._get_unique_songs_df()

            yield f'data:Getting Top Artists...2/{total}<br>\n\n\n'
            self._get_top_artists()

            yield f'data:Getting Top Songs...3/{total}<br>\n\n\n'
            self._get_top_songs()

            yield f'data:Adding Top Artists Rank...4/{total}<br>\n\n\n'
            self._add_top_artists_rank()

            yield f'data:Adding Top Songs Rank...5/{total}<br>\n\n\n'
            self._add_top_songs_rank()

            yield f'data:Getting Artist Genres...6/{total}<br>\n\n\n'
            self._add_genres()

            yield f'data:Setting Up Home Page...7/{total}<br>\n\n\n'
            UNIQUE_SONGS_DF = pd.read_pickle(f'{self.path}unique_songs_df.pkl')
            home_page = HomePage(self.path, ALL_SONGS_DF, UNIQUE_SONGS_DF)
            _dump(f'{self.path}home_page.pkl', home_page)

            yield f'data:Setting Up About Me Page...8/{total}<br>\n\n\n'
            artists = self.SPOTIFY.current_user_followed_artists()[
                'artists']['items']
            top_artists = _load(f'{self.path}top_artists.pkl')
            top_songs = _load(f'{self.path}top_songs.pkl')
            about_page = AboutPage(
                self.path, ALL_SONGS_DF, UNIQUE_SONGS_DF, artists)
            _dump(f'{self.path}about_page.pkl', about_page)

            yield f'data:Setting up Top50 Page...9/{total}<br>\n\n\n'
            top50_page = Top50Page(
                self.path, UNIQUE_SONGS_DF, top_artists)
            _dump(f'{self.path}top50_page.pkl', top50_page)

            yield f'data:Setting up My Playlists Page...10/{total}<br>\n\n\n'
            myplaylists_page = MyPlaylistsPage(
                self.path, ALL_SONGS_DF, top_artists, top_songs)
            _dump(f'{self.path}myplaylists_page.pkl', myplaylists_page)

            yield f'data:Finalizing Data Collection...{total}/{total}\n\n'
            status = {'SETUP1': True, 'SETUP2': True, 'SETUP3': False}
            _dump(f'{self.path}collection.pkl', status)

            yield 'data:REDIRECT_URI=' + REDIRECT_URI + '\n\n'
        except Exception as e:
            yield 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'