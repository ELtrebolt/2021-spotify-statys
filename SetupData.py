# SetupData.py = Class for Managing creation of DataFrames from SpotifyAPI
from datetime import datetime
import pandas as pd
import pickle
import os
import json
import gzip
from visualization import HomePage, AboutPage, Top50Page, MyPlaylistsPage
import traceback

PERCENTILE_COLS = ['popularity', 'duration']
# DEPRECATED 2025
#FEATURE_COLS = ['id', 'danceability', 'energy', 'loudness', 'speechiness',
#                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']
# don't need mode, key, type, uri, track_href, analysis_url, time_signature
        

def _dump(path, obj):
    with open(path, 'wb') as f:   # Pickling
        pickle.dump(obj, f)

def _load(path):
    with open(path, 'rb') as f:  # Unpickling
        return pickle.load(f)

class SetupData():
    def __init__(self, session):
        self.SPOTIFY = session['SPOTIFY']
        
        self.USER_ID = self.SPOTIFY.me()['id']
        self.PLAYLIST_DICT = self._get_all_playlists_dict()

        self.path = f'.data/{self.USER_ID}/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        status = {'SETUP1': False, 'SETUP2': False, 'SETUP3': False}
        try:
            with open(os.path.join(self.path, 'setup_status.json'), 'w', encoding='utf-8') as f:
                json.dump(status, f)
        except Exception:
            pass

        # Filesystem-first metadata (JSON). Keep pickles for backward-compat.
        meta_path = os.path.join(self.path, 'meta.json')
        try:
            display_name = self.SPOTIFY.me().get('display_name', '')
        except Exception:
            display_name = ''
        meta = {
            'user_id': self.USER_ID,
            'display_name': display_name,
            'last_login': datetime.utcnow().isoformat() + 'Z'
        }
        self._atomic_json_write(meta_path, meta)
        self._atomic_json_write(os.path.join(self.path, 'setup_status.json'), {
            'setup1': False,
            'setup2': False,
            'setup3': False
        })

    def _atomic_json_write(self, path, obj):
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)


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
        try:
            if len(tracks['items']) == 0:
                return pd.DataFrame()
        except KeyError as e:
            # when this happens that means tracks = {'next':None}
            return pd.DataFrame()

        song_meta = {'id': [], 'name': [],
                     'artist': [], 'album': [], 'explicit': [], 'popularity': [],
                     'playlist': [], 'date_added': [], 'artist_ids': [], 'duration': []}

        for item in tracks['items']:
            meta = item['track']

            # For Bernardo and Adam, this was the error - meta was None
            # For Beni meta['id'] was None due to Kanye West leaked unrelease - Spread Your Wings & Flowers
            if meta is not None and meta['id'] is not None:
                # For special cases like the podcast 'greendale is where i belong'
                if meta['type'] == 'episode':
                    # print('Episode Found:', meta['name'])
                    continue

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

                # convert milliseconds to secs
                # duration_ms: The duration of the track in milliseconds.
                # 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
                song_meta['duration'].append(meta['duration_ms']/1000)

        song_meta_df = pd.DataFrame.from_dict(song_meta)

        return song_meta_df

    def _write_tracks_relations(self, all_songs_df: pd.DataFrame):
        """Write compressed NDJSON files for tracks and playlist relations."""
        # Tracks (deduplicated by id)
        track_cols = ['id', 'name', 'album', 'explicit', 'popularity', 'duration']
        tracks = (all_songs_df[track_cols]
                  .drop_duplicates(subset=['id'])
                  .sort_values('id'))
        tracks_path = os.path.join(self.path, 'tracks.ndjson.gz')
        self._atomic_gzip_ndjson_write(tracks_path, tracks)

        # Relations: playlist -> track with added_at
        rel_cols = ['playlist', 'id', 'date_added']
        rel = all_songs_df[rel_cols].rename(columns={'id': 'track_id', 'date_added': 'added_at'})
        rel_path = os.path.join(self.path, 'playlist_tracks.ndjson.gz')
        self._atomic_gzip_ndjson_write(rel_path, rel)

    def _atomic_gzip_ndjson_write(self, path: str, df: pd.DataFrame):
        """Atomically write a DataFrame as gzipped NDJSON."""
        tmp = path + '.tmp'
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            # Use orient='records', lines=True output
            for record in df.to_dict(orient='records'):
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')
        os.replace(tmp, path)


    # Get all of a playlists songs
    def _get_playlist(self, name, _id):
        df = pd.DataFrame()

        try:
            tracks = self.SPOTIFY.playlist_items(_id, offset=0)
        except:
            return 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'
        
        df = pd.concat([df, self._get_100_songs(tracks, name)])

        offset = len(tracks['items'])
        while tracks['next']:
            # issue with getting more than 100 tracks = {'next': url}, doesn't have items
            # now have to use playlist_tracks and offset instead of self.SPOTIFY.next(tracks)
            tracks = self.SPOTIFY.playlist_items(_id, offset=offset)
            df = pd.concat([df, self._get_100_songs(tracks, name)])
            offset += len(tracks['items'])

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
                    yield 'data:' + name + ' --> ' + str(count) + '/' + str(total) + '<br/>\n\n\n'
                    if len(ALL_SONGS_DF) > 8888:
                        yield 'data:***Total Song Count Reached Max Limit***<br/>\n\n\n'
                        break
                count += 1

            # Yung Yi had the problem of 'index' not found in axis
            if 'index' in ALL_SONGS_DF.columns:
                ALL_SONGS_DF.drop(columns='index', inplace=True)
            # Write compact filesystem-friendly files
            self._write_tracks_relations(ALL_SONGS_DF)

            status = {'SETUP1': True, 'SETUP2': False, 'SETUP3': False}
            self._atomic_json_write(os.path.join(self.path, 'setup_status.json'), status)

        except Exception as e:
            yield 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'


    def _get_unique_songs_df(self):
        # Changing cols will be condensed into a list = ex: unique song will have a col "playlist" = [playlist1, playlist2]
        ALL_SONGS_DF = pd.read_json(f'{self.path}all_songs.ndjson.gz', lines=True, compression='gzip')

        changing_cols = ['id', 'playlist', 'date_added']
        UNIQUE_SONGS_DF = ALL_SONGS_DF.groupby([i for i in ALL_SONGS_DF.columns if i not in changing_cols], as_index=False)[
            changing_cols].agg(lambda x: list(x))
        duplicates = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF.duplicated(subset=['name', 'artist'], keep=False)]
        duplicates.sort_values(by=['name', 'popularity'], ascending=[True, False], inplace=True)
        duplicates_with_lower_popularity = duplicates.iloc[1::2]
        # duplicates_with_lower_popularity.to_csv('dups.csv')
        # All Night by Vamps should have 65 instead of 0 popularity
        UNIQUE_SONGS_DF.drop(duplicates_with_lower_popularity.index, inplace=True)
        
        for col in PERCENTILE_COLS:
            sz = UNIQUE_SONGS_DF[col].size-1
            UNIQUE_SONGS_DF[col + '_percentile'] = UNIQUE_SONGS_DF[col].rank(
                method='max').apply(lambda x: 100.0*(x-1)/sz)

        UNIQUE_SONGS_DF['num_playlists'] = [
            len(i) for i in UNIQUE_SONGS_DF['playlist']]

        # Also write compressed NDJSON for filesystem mode
        self._atomic_gzip_ndjson_write(os.path.join(self.path, 'unique_songs.ndjson.gz'), UNIQUE_SONGS_DF)


    def _get_top_artists(self):
        # [Artist1, Artist2, Artist3]
        TOP_ARTISTS_SHORT = {i['name']:i['popularity'] for i in self.SPOTIFY.current_user_top_artists(
            time_range='short_term', limit=50)['items']}
        TOP_ARTISTS_MED = {i['name']:i['popularity'] for i in self.SPOTIFY.current_user_top_artists(
            time_range='medium_term', limit=50)['items']}
        TOP_ARTISTS_LONG = {i['name']:i['popularity'] for i in self.SPOTIFY.current_user_top_artists(
            time_range='long_term', limit=50)['items']}
        TOP_ARTISTS_NAMES = [list(TOP_ARTISTS_SHORT.keys()), list(TOP_ARTISTS_MED.keys()), list(TOP_ARTISTS_LONG.keys())]
        TOP_ARTISTS_POPULARITY = [list(TOP_ARTISTS_SHORT.values()), list(TOP_ARTISTS_MED.values()), list(TOP_ARTISTS_LONG.values())]

        # Legacy pickle writes removed - JSON handles this now
        # Filesystem JSON mirrors
        self._atomic_json_write(os.path.join(self.path, 'top_artists.json'), TOP_ARTISTS_NAMES)
        self._atomic_json_write(os.path.join(self.path, 'top_artists_pop.json'), TOP_ARTISTS_POPULARITY)


    # INT or "N/A" if single, STR separated by commas if multiple
    def _add_top_artists_rank(self):
        UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
        with open(f'{self.path}top_artists.json', 'r', encoding='utf-8') as f:
            TOP_ARTISTS = json.load(f)

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
        # Persist updated ranks to NDJSON (pickle not required for runtime)
        self._atomic_gzip_ndjson_write(os.path.join(self.path, 'unique_songs.ndjson.gz'), UNIQUE_SONGS_DF)


    def _get_top_songs(self):
        # {Song: ('Artists Separated By Commas', Rank)}
        TOP_SONGS_SHORT = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
            self.SPOTIFY.current_user_top_tracks(time_range='short_term', limit=50)['items'])}
        TOP_SONGS_MED = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
            self.SPOTIFY.current_user_top_tracks(time_range='medium_term', limit=50)['items'])}
        TOP_SONGS_LONG = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
            self.SPOTIFY.current_user_top_tracks(time_range='long_term', limit=50)['items'])}
        TOP_SONGS = [TOP_SONGS_SHORT, TOP_SONGS_MED, TOP_SONGS_LONG]

        # Persist Top Songs as JSON (pickle not required)
        self._atomic_json_write(os.path.join(self.path, 'top_songs.json'), TOP_SONGS)


    def _add_top_songs_rank(self):
        UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
        with open(f'{self.path}top_songs.json', 'r', encoding='utf-8') as f:
            TOP_SONGS = json.load(f)

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
        # Persist updated ranks to NDJSON (pickle not required for runtime)
        self._atomic_gzip_ndjson_write(os.path.join(self.path, 'unique_songs.ndjson.gz'), UNIQUE_SONGS_DF)


    def _add_genres(self):
        # Add 'genres' column to ALL_SONGS and UNIQUE_SONGS like ['pop', 'punk']
        ALL_SONGS_DF = pd.read_json(f'{self.path}all_songs.ndjson.gz', lines=True, compression='gzip')
        UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')

        # Get Genres for Each Song By Artist Genres - Takes 2 minutes for 1000 unique artists
        unique_artist_names, unique_artist_ids = set(), set()
        for i in zip(UNIQUE_SONGS_DF['artist'], UNIQUE_SONGS_DF['artist_ids']):
            unique_artist_names = unique_artist_names | set(i[0].split(', '))
            unique_artist_ids = unique_artist_ids | set(i[1].split(', '))
        # Store unique artist names as JSON for search features
        try:
            with open(os.path.join(self.path, 'unique_artist_names.json'), 'w', encoding='utf-8') as f:
                json.dump(sorted(list(unique_artist_names)), f)
        except Exception:
            pass
        # if '' in unique_artist_ids:
        #     unique_artist_ids.remove('')

        # Save artist genres from Spotify API all at once using dict(), then make new cols for dataframes
        # {artist1: genres_list, artist2: ['N/A']}
        genres_dict = dict()
        total = len(unique_artist_ids)
        unique_artist_ids = list(unique_artist_ids)

        # Getting 50 artists at a time is a LOT faster than getting 1 artist at a time
        for i in range(0, total, 50):
            listy = self.SPOTIFY.artists(unique_artist_ids[i:i+50])
            for a in listy['artists']:
                try:
                    g = a['genres']
                except Exception as e:
                    print(e)
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
                    # Potential error here? ufo ufo does not have la pop as a genre
                    try:
                        song_genres.append(genres_dict[j])
                    except Exception as e:
                        print('Artist ID DUP FOUND', j, e)
                        a = self.SPOTIFY.artist(j)
                        if a['genres']:
                            song_genres.append(a['genres'])
                        else:
                            song_genres.append('N/A')
                genres_list.append(song_genres)
            df['genres'] = genres_list

        # Filesystem-friendly outputs for analytics
        self._atomic_gzip_ndjson_write(os.path.join(self.path, 'all_songs.ndjson.gz'), ALL_SONGS_DF)
        self._atomic_gzip_ndjson_write(os.path.join(self.path, 'unique_songs.ndjson.gz'), UNIQUE_SONGS_DF)


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

            # Takes a while
            yield f'data:Getting Artist Genres...6/{total}<br>\n\n\n'
            self._add_genres()

            yield f'data:Setting Up Home Page...7/{total}<br>\n\n\n'
            UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
            home_page = HomePage(self.path, ALL_SONGS_DF, UNIQUE_SONGS_DF)
            # Page objects no longer persisted; views compute on demand

            yield f'data:Setting Up About Me Page...8/{total}<br>\n\n\n'
            artists = self.SPOTIFY.current_user_followed_artists()[
                'artists']['items']
            with open(f'{self.path}top_artists.json', 'r', encoding='utf-8') as f:
                top_artists = json.load(f)
            with open(f'{self.path}top_songs.json', 'r', encoding='utf-8') as f:
                top_songs = json.load(f)
            about_page = AboutPage(
                self.path, ALL_SONGS_DF, UNIQUE_SONGS_DF, artists)
            # Not persisted

            # Takes a while
            yield f'data:Setting Up Top50 Page...9/{total}<br>\n\n\n'
            with open(f'{self.path}top_artists_pop.json', 'r', encoding='utf-8') as f:
                top_artists_pop = json.load(f)
            top50_page = Top50Page(
                self.path, UNIQUE_SONGS_DF, top_artists, top_artists_pop)
            # Not persisted

            yield f'data:Setting Up My Playlists Page...10/{total}<br>\n\n\n'
            myplaylists_page = MyPlaylistsPage(
                self.path, ALL_SONGS_DF, top_artists, top_songs)
            # Not persisted

            yield f'data:Finalizing Data Collection...{total}/{total}\n\n'
            status = {'SETUP1': True, 'SETUP2': True, 'SETUP3': False}
            self._atomic_json_write(os.path.join(self.path, 'setup_status.json'), status)

        except Exception as e:
            yield 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'