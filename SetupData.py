# SetupData.py = Class for Managing creation of DataFrames from SpotifyAPI
import os
import pandas as pd
import json
from datetime import datetime
import gzip
from visualization import HomePage, AboutPage, Top50Page, MyPlaylistsPage
import traceback
import requests
import time

PERCENTILE_COLS = ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'duration', 'tempo', 'loudness']
# DEPRECATED 2025
#FEATURE_COLS = ['id', 'danceability', 'energy', 'loudness', 'speechiness',
#                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']
# don't need mode, key, type, uri, track_href, analysis_url, time_signature
MAX_SONGS = 8888
        

class SetupData():
    def __init__(self, session):
        self.SPOTIFY = session['SPOTIFY']
        self.USER_ID = self.SPOTIFY.me()['id']
        self.PLAYLIST_DICT = self._get_all_playlists_dict()

        self.path = f'.data/{self.USER_ID}/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Filesystem-first metadata (JSON) - consolidated with setup data
        meta_path = os.path.join(self.path, 'user_meta.json')
        try:
            display_name = self.SPOTIFY.me().get('display_name', '')
        except Exception:
            display_name = ''
        meta = {
            'user_id': self.USER_ID,
            'display_name': display_name,
            'last_login': datetime.utcnow().isoformat() + 'Z',
            'playlist_dict': self.PLAYLIST_DICT
        }
        self._atomic_json_write(meta_path, meta)
        
        # Initialize setup status with consistent lowercase keys (only if file doesn't exist)
        status_path = os.path.join(self.path, 'setup_status.json')
        if not os.path.exists(status_path):
            status = {'setup1': False, 'setup2': False, 'setup3': False}
            self._atomic_json_write(status_path, status)

    def _atomic_json_write(self, path, obj):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)


    def _get_50_playlist_dict(self, playlists):
        dicty = {}
        for playlist in playlists:
            try:
                if playlist['owner']['id'] == self.USER_ID:
                    dicty[playlist['name']] = playlist['id']
            except Exception as e:
                pass
        return dicty
    

    def _get_all_playlists_dict(self):
        dicty = {}

        try:
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
        except Exception as e:
            return {}


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
        temp_path = None
        try:
            # Ensure path is absolute and normalized
            path = os.path.abspath(path)
            
            # Check if directory exists
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            # Create temporary file path with unique timestamp to avoid conflicts
            timestamp = str(int(time.time() * 1000))
            temp_path = f"{path}.{timestamp}.tmp"
            
            # Write to temporary file first
            with gzip.open(temp_path, 'wb') as f:
                # Use orient='records', lines=True output
                records = df.to_dict(orient='records')
                
                for i, record in enumerate(records):
                    try:
                        json_str = json.dumps(record, ensure_ascii=False)
                        # Encode to bytes for binary gzip
                        f.write(json_str.encode('utf-8'))
                        f.write(b'\n')
                    except Exception as e:
                        # Clean up temp file on error
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass  # Ignore cleanup errors
                        raise
            
            # Close the file handle explicitly to ensure it's released
            f = None
            
            # Small delay to ensure file handles are fully released
            time.sleep(0.2)
            
            # Atomically rename temp file to target file
            if os.path.exists(path):
                try:
                    os.remove(path)
                    time.sleep(0.1)  # Small delay after removal
                except Exception as remove_error:
                    # If we can't remove the target, try to use a different name
                    backup_path = f"{path}.backup"
                    try:
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                        os.rename(path, backup_path)
                        time.sleep(0.1)
                    except:
                        pass  # Continue with the write attempt
            
            # Now rename the temp file
            os.rename(temp_path, path)
            temp_path = None  # Mark as successfully moved
            
            # Small delay to ensure file is fully written to disk
            time.sleep(0.1)
            
            # Verify the file can be read back (basic integrity check)
            try:
                if os.path.exists(path):
                    with gzip.open(path, 'rb') as f:
                        # Try to read first few bytes to verify gzip integrity
                        f.read(1024)
                else:
                    raise Exception(f"Target file {path} does not exist after rename")
            except Exception as verify_error:
                # If verification fails, remove the corrupted file
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
                raise Exception(f"File verification failed after write: {verify_error}")
            
        except Exception as e:
            # Clean up temp file on any error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass  # Ignore cleanup errors
            raise

    def _get_audio_features_batch(self, song_ids: list, batch_num: int, total_batches: int, progress_callback=None):
        """
        Get audio features for a batch of 40 songs using ReccoBeats API.
        
        This is a two-step process with only 2 API calls per batch:
        1. Get audio features for all Spotify IDs in the batch
        2. Get track details for all ReccoBeats IDs in the batch
        
        Args:
            song_ids: List of Spotify track IDs to get features for
            batch_num: Current batch number for progress tracking
            total_batches: Total number of batches for progress tracking
            progress_callback: Optional function to call for progress updates
            
        Returns:
            DataFrame with song IDs and audio features, or None if error
        """
        if not song_ids:
            return None
            
        try:
            # Step 1: Get audio features for all Spotify IDs in the batch
            features_url = "https://api.reccobeats.com/v1/audio-features"
            ids_param = ",".join(song_ids)
            features_params = {"ids": ids_param}
            
            if progress_callback:
                progress_callback(f"Requesting audio features for batch {batch_num}/{total_batches}...")
            
            features_response = requests.get(features_url, params=features_params, timeout=30)
            
            if features_response.status_code == 200:
                # Success - parse audio features
                features_data = features_response.json()
                if progress_callback:
                    progress_callback(f"Received audio features response for batch {batch_num}")
                
                # Create a mapping from Spotify ID to position in the input list
                spotify_id_to_position = {spotify_id: i for i, spotify_id in enumerate(song_ids)}
                
                # Initialize DataFrame with song IDs
                features_df = pd.DataFrame({'id': song_ids})
                
                # Audio features columns to extract
                audio_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                
                # Initialize audio feature columns with None
                for col in audio_cols:
                    features_df[col] = None
                
                # Initialize reccobeats_id column
                features_df['reccobeats_id'] = None
                
                # Extract features and collect ReccoBeats IDs for batch processing
                reccobeats_ids = []
                audio_features_map = {}  # Map ReccoBeats ID to audio features
                
                if isinstance(features_data, dict) and 'content' in features_data:
                    content_data = features_data['content']
                    if progress_callback:
                        progress_callback(f"Processing {len(content_data)} audio feature records...")
                    
                    # Process each audio feature record and collect ReccoBeats IDs
                    for track_data in content_data:
                        if 'id' in track_data:
                            reccobeats_id = track_data['id']
                            reccobeats_ids.append(reccobeats_id)
                            
                            # Store audio features for this ReccoBeats ID
                            audio_features_map[reccobeats_id] = {}
                            for col in audio_cols:
                                if col in track_data:
                                    audio_features_map[reccobeats_id][col] = track_data[col]
                
                if not reccobeats_ids:
                    if progress_callback:
                        progress_callback(f"Warning: No ReccoBeats IDs found in batch {batch_num}")
                    return features_df
                
                if progress_callback:
                    progress_callback(f"Requesting track details for {len(reccobeats_ids)} ReccoBeats IDs...")
                
                # Step 2: Get track details for all ReccoBeats IDs in the batch
                track_url = "https://api.reccobeats.com/v1/track"
                track_ids_param = ",".join(reccobeats_ids)
                track_params = {"ids": track_ids_param}
                
                track_response = requests.get(track_url, params=track_params, timeout=30)
                
                if track_response.status_code == 200:
                    track_info = track_response.json()
                    if progress_callback:
                        progress_callback(f"Received track details for batch {batch_num}")
                    
                    # Process track details to map ReccoBeats IDs to Spotify IDs
                    reccobeats_to_spotify = {}
                    
                    if isinstance(track_info, dict) and 'content' in track_info:
                        track_content = track_info['content']
                        if progress_callback:
                            progress_callback(f"Mapping {len(track_content)} track details to Spotify IDs...")
                        
                        for track in track_content:
                            if 'id' in track and 'href' in track:
                                reccobeats_id = track['id']
                                href = track['href']
                                
                                # Extract Spotify ID from href
                                if 'spotify.com/track/' in href:
                                    spotify_id = href.split('/track/')[-1].split('?')[0]
                                    reccobeats_to_spotify[reccobeats_id] = spotify_id
                    
                    # Now update the features DataFrame with the correct mappings
                    if progress_callback:
                        progress_callback(f"Updating features for {len(reccobeats_to_spotify)} matched songs...")
                    
                    for reccobeats_id, spotify_id in reccobeats_to_spotify.items():
                        # Find the row in our DataFrame by Spotify ID
                        spotify_row = features_df[features_df['id'] == spotify_id]
                        if not spotify_row.empty:
                            row_idx = spotify_row.index[0]
                            
                            # Store the ReccoBeats ID
                            features_df.loc[row_idx, 'reccobeats_id'] = reccobeats_id
                            
                            # Store the audio features
                            if reccobeats_id in audio_features_map:
                                for col, value in audio_features_map[reccobeats_id].items():
                                    features_df.loc[row_idx, col] = value
                
                else:
                    if progress_callback:
                        progress_callback(f"Warning: Track details API failed for batch {batch_num} (status: {track_response.status_code})")
                
                if progress_callback:
                    progress_callback(f"Batch {batch_num} completed with {len(features_df)} songs processed")
                return features_df
                
            elif features_response.status_code == 429:
                # Rate limited - check Retry-After header
                retry_after = features_response.headers.get('Retry-After', 60)
                retry_seconds = int(retry_after)
                if progress_callback:
                    progress_callback(f"Rate limited on batch {batch_num}, waiting {retry_seconds} seconds...")
                
                # Wait for the specified time
                time.sleep(retry_seconds)
                
                # Retry the same batch
                return self._get_audio_features_batch(song_ids, batch_num, total_batches, progress_callback)
                
            else:
                # Other error
                if progress_callback:
                    progress_callback(f"Warning: Audio features API failed for batch {batch_num} (status: {features_response.status_code})")
                return None
                
        except requests.exceptions.RequestException as e:
            if progress_callback:
                progress_callback(f"Warning: Network error on batch {batch_num}: {str(e)}")
            return None
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Unexpected error on batch {batch_num}: {str(e)}")
            return None



    def _merge_audio_features_to_all_songs(self, all_songs_df: pd.DataFrame, unique_songs_df: pd.DataFrame):
        """
        Merge audio features from unique songs back to all songs DataFrame.
        
        Args:
            all_songs_df: DataFrame containing all songs
            unique_songs_df: DataFrame containing unique songs with audio features
            
        Returns:
            Updated all_songs_df with audio features
        """
        if all_songs_df.empty or unique_songs_df.empty:
            return all_songs_df
            
        # Audio features columns to merge
        audio_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        # Initialize audio feature columns with None in all_songs_df
        for col in audio_cols:
            all_songs_df[col] = None
        
        # Create a mapping from (name, artist) to audio features
        features_map = {}
        for _, row in unique_songs_df.iterrows():
            key = (row['name'], row['artist'])
            features = {}
            for col in audio_cols:
                if col in row and pd.notna(row[col]):
                    features[col] = row[col]
            features_map[key] = features
        
        # Apply audio features to all songs based on name and artist match
        for idx, row in all_songs_df.iterrows():
            key = (row['name'], row['artist'])
            if key in features_map:
                for col, value in features_map[key].items():
                    all_songs_df.loc[idx, col] = value
        
        return all_songs_df






    # Get all of the user's liked/saved songs
    def _get_liked_songs(self):
        df = pd.DataFrame()
        playlist_name = "Liked Songs"

        try:
            tracks = self.SPOTIFY.current_user_saved_tracks(limit=50, offset=0)
        except:
            return 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'
        
        df = pd.concat([df, self._get_100_songs(tracks, playlist_name)])

        offset = len(tracks['items'])
        while tracks['next']:
            tracks = self.SPOTIFY.current_user_saved_tracks(limit=50, offset=offset)
            df = pd.concat([df, self._get_100_songs(tracks, playlist_name)])
            offset += len(tracks['items'])

        return df.reset_index()

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
            total = len(self.PLAYLIST_DICT)  # don't +1 include liked songs
            yield 'data:<h1>Collecting Your Playlists</h1>\n\n'

            ALL_SONGS_DF = pd.DataFrame()
            for name, _id in list(self.PLAYLIST_DICT.items()):
                df = self._get_playlist(name, _id)
                if type(df) == str:
                    yield df
                else:
                    ALL_SONGS_DF = pd.concat([ALL_SONGS_DF, df])
                    yield 'data:' + name + ' --> ' + str(count) + '/' + str(total) + '<br/>\n\n\n'
                    if len(ALL_SONGS_DF) > MAX_SONGS:
                        yield 'data:***Total Song Count Reached Max Limit***<br/>\n\n\n'
                        break
                count += 1

            # Collect liked songs if we haven't hit the limit
            if len(ALL_SONGS_DF) <= MAX_SONGS:
                yield 'data:PROGRESS: Processing Liked Songs...<br/>\n\n\n'
                liked_df = self._get_liked_songs()
                if type(liked_df) == str:
                    yield liked_df
                else:
                    ALL_SONGS_DF = pd.concat([ALL_SONGS_DF, liked_df])
                    yield 'data:Finished GettingLiked Songs!<br/>\n\n\n'

            # Yung Yi had the problem of 'index' not found in axis
            if 'index' in ALL_SONGS_DF.columns:
                ALL_SONGS_DF.drop(columns='index', inplace=True)
            # Write compact filesystem-friendly files
            self._write_tracks_relations(ALL_SONGS_DF)
            
            # Save all_songs for setup2 to use if it needs to restart
            yield 'data:PROGRESS: Saving All Songs...<br/>\n\n\n'
            self._atomic_gzip_ndjson_write(f'{self.path}all_songs.ndjson.gz', ALL_SONGS_DF)
            yield 'data:All songs data saved successfully!<br/>\n\n\n'

            # Export to CSV for specific user ID
            if self.USER_ID == 'qf26s87ilixm0wn6njz7amx2f':
                try:
                    csv_path = '.data/all_songs_df.csv'
                    ALL_SONGS_DF.to_csv(csv_path, index=False)
                except Exception as e:
                    yield f'data:Warning: CSV export failed: {str(e)}<br/>\n\n\n'

            status = {'setup1': True, 'setup2': False, 'setup3': False}
            self._atomic_json_write(os.path.join(self.path, 'setup_status.json'), status)

        except Exception as e:
            yield 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'

    def setup_2(self):
        """
        Setup 2: Audio Features Processing
        - Get unique songs DataFrame
        - Collect audio features from ReccoBeats API
        - Add audio features to both DataFrames
        - Calculate percentiles
        - Export to ndjson.gz
        - Set completion flag for Setup3
        """
        try:

            
            # Check for existing progress in setup_status.json
            status_file = f'{self.path}setup_status.json'
            resume_from_batch = 0
            processed_count = 0
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                    resume_from_batch = status_data.get('setup2_last_batch', 0)
                    processed_count = status_data.get('setup2_processed_count', 0)

                except Exception as e:
                    yield f"data:Warning: Could not read status file: {str(e)}<br>\n\n\n"
            
            # Step 1: Getting All of your Unique Songs...
            yield "data:PROGRESS: 1/6 Getting Your Unique Songs...<br>\n\n\n"
            UNIQUE_SONGS_DF = self._get_unique_songs_df()
            
            # Step 2: Retrieving Audio Features Database...
            yield "data:PROGRESS: 2/6 Retrieving Audio Features Database...<br>\n\n\n"
            
            # Check if db_all_tracks.csv exists
            db_tracks_file = '.data/db_all_tracks.csv'
            if os.path.exists(db_tracks_file):
                try:
                    db_tracks_df = pd.read_csv(db_tracks_file)
                    
                    # Create a mapping from Spotify ID to audio features
                    audio_features_map = {}
                    for _, row in db_tracks_df.iterrows():
                        spotify_id = row.get('id', '')
                        if pd.notna(spotify_id) and spotify_id:
                            # Store audio features
                            features = {}
                            for col in ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                                       'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                                       'tempo', 'valence', 'duration_ms']:
                                if col in row and pd.notna(row[col]):
                                    features[col] = row[col]
                            
                            if features:  # Only store if we have some features
                                audio_features_map[spotify_id] = features
                    

                    
                    # Apply existing audio features to UNIQUE_SONGS_DF
                    features_applied = 0
                    for idx, row in UNIQUE_SONGS_DF.iterrows():
                        # UNIQUE_SONGS_DF['id'] is now a single string, not a list
                        song_id = row.get('id', '')
                        
                        if song_id in audio_features_map:
                            features = audio_features_map[song_id]
                            for col, value in features.items():
                                if col == 'duration_ms':
                                    # Convert duration_ms to duration (seconds)
                                    UNIQUE_SONGS_DF.at[idx, 'duration'] = value / 1000 if pd.notna(value) else None
                                else:
                                    UNIQUE_SONGS_DF.at[idx, col] = value
                            features_applied += 1
                    
                    # Step 3: Matching Database with Your Library...
                    yield f"data:PROGRESS: 3/6 Matching Database with Your Library...<br>\n\n\n"

                    
                except Exception as e:
                    yield f"data:Warning: Could not load db_all_tracks.csv: {str(e)}<br>\n\n\n"
                    audio_features_map = {}
                    # Step 3: Matching Database with Your Library... (no database)
                    yield f"data:PROGRESS: 3/6 Matching Audio Features with Your Library...<br>\n\n\n"
            else:
                yield f"data:PROGRESS: 3/6 Matching Audio Features with Your Library...<br>\n\n\n"
                audio_features_map = {}
            
            # Initialize audio feature columns (only for columns that don't already have data)
            for col in ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                       'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                       'tempo', 'valence', 'reccobeats_id']:
                if col not in UNIQUE_SONGS_DF.columns:
                    UNIQUE_SONGS_DF[col] = None
            
            # Get song IDs for API calls (only for songs that are NOT in db_all_tracks.csv at all)
            song_ids = []
            songs_needing_features = []
            songs_in_db_but_no_features = 0
            
            for idx, row in UNIQUE_SONGS_DF.iterrows():
                song_id = row['id']
                # UNIQUE_SONGS_DF['id'] is now a single string, not a list
                
                # Check if this song is in the database (regardless of whether it has features)
                is_in_database = song_id in audio_features_map
                
                if is_in_database:
                    # Song is in database - check if it has features
                    has_features = False
                    for col in ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                               'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                               'tempo', 'valence']:
                        if col in row and pd.notna(row[col]):
                            has_features = True
                            break
                    
                    if not has_features:
                        songs_in_db_but_no_features += 1
                        # Skip API call - already tried ReccoBeats for this song
                else:
                    # Song is NOT in database - needs API call
                    song_ids.append(song_id)
                    songs_needing_features.append(idx)
            
            # Filter out None values and get songs that need features
            valid_song_ids = [sid for sid in song_ids if sid is not None]
            total_songs_needing_features = len(valid_song_ids)
            total_songs_total = len(UNIQUE_SONGS_DF)
            
            # Log the filtering results

            
            if total_songs_needing_features == 0:

                # Skip to percentile calculation
                yield f"data:PROGRESS: 4/6 Found 100% of your songs in the database!<br>\n\n\n"
            else:
                # Make API calls for songs not in database, but skip songs already in database (even if no features)

                
                # Step 4: Getting X/Total Songs (%) Without Match... (dynamic updates)
                yield f"data:PROGRESS: 4/6 Getting 0/{total_songs_needing_features} Songs (0%) Without Match...<br>\n\n\n"
                
                # Process in batches of 40
                batch_size = 40
                total_batches = (len(valid_song_ids) + batch_size - 1) // batch_size
                processed_count = total_songs_total - total_songs_needing_features  # Start with songs that already have features
                
                # Start from the resume point
                for batch_num in range(resume_from_batch, total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(valid_song_ids))
                    batch_song_ids = valid_song_ids[start_idx:end_idx]
                    
                    # Update progress for step 4 - show current batch progress (skip first batch to avoid duplicate 0%)
                    if batch_num > 0:  # Skip first batch since we already showed 0% initially
                        current_processed = batch_num * batch_size
                        percentage = (current_processed / total_songs_needing_features * 100) if total_songs_needing_features > 0 else 0
                        yield f"data:PROGRESS: 4/6 Getting {current_processed}/{total_songs_needing_features} Songs ({percentage:.1f}%) Without Match...<br>\n\n\n"
                    
                    # Use the previous robust implementation
                    def progress_callback(msg):
                        # Store progress messages for later yielding
                        progress_callback.messages.append(msg)
                    
                    progress_callback.messages = []
                    
                    # Get audio features for this batch
                    batch_features_df = self._get_audio_features_batch(batch_song_ids, batch_num + 1, total_batches, progress_callback)
                    
                    # Yield any progress messages from the callback

                    
                    if batch_features_df is not None and not batch_features_df.empty:
                        # Merge the features back to UNIQUE_SONGS_DF
                        for _, feature_row in batch_features_df.iterrows():
                            spotify_id = feature_row['id']
                            
                            # Find the song in our dataframe and update it
                            # UNIQUE_SONGS_DF['id'] is now a single string, not a list
                            for idx, row in UNIQUE_SONGS_DF.iterrows():
                                song_id = row['id']
                                if song_id == spotify_id:
                                    # Update with new features
                                    audio_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                                                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                                    for col in audio_cols:
                                        if col in feature_row and pd.notna(feature_row[col]):
                                            UNIQUE_SONGS_DF.at[idx, col] = feature_row[col]
                                    
                                    # Also update reccobeats_id if available
                                    if 'reccobeats_id' in feature_row and pd.notna(feature_row['reccobeats_id']):
                                        UNIQUE_SONGS_DF.at[idx, 'reccobeats_id'] = feature_row['reccobeats_id']
                                    
                                    processed_count += 1
                                    break
                    
                    # Save progress after each batch
                    try:
                        with open(status_file, 'r') as f:
                            status_data = json.load(f)
                    except:
                        status_data = {'setup1': False, 'setup2': False, 'setup3': False}
                    
                    status_data['setup2_last_batch'] = batch_num
                    status_data['setup2_processed_count'] = processed_count
                    
                    with open(status_file, 'w') as f:
                        json.dump(status_data, f)
                    
                    # Small delay between batches
                    time.sleep(1)
                
                # Final progress update for step 4
                yield f"data:PROGRESS: 4/6 Getting {total_songs_needing_features}/{total_songs_needing_features} Songs (100%) Without Match...<br>\n\n\n"
            
            # Calculate final statistics
            songs_with_features = UNIQUE_SONGS_DF[['acousticness', 'danceability', 'energy', 'instrumentalness', 
                                                 'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                                                 'tempo', 'valence']].notna().any(axis=1).sum()
            features_percentage = (songs_with_features / total_songs_total * 100) if total_songs_total > 0 else 0
            

            
            # Step 5: Calculating Overall Percentiles...
            yield "data:PROGRESS: 5/6 Calculating Overall Percentiles...<br>\n\n\n"
            percentile_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                              'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                              'tempo', 'valence', 'popularity']
            
            # VECTORIZED PERCENTILE CALCULATION: Use pandas operations instead of apply for better performance
            for col in percentile_cols:
                if col in UNIQUE_SONGS_DF.columns and not UNIQUE_SONGS_DF[col].isna().all():
                    # Vectorized percentile calculation
                    ranks = UNIQUE_SONGS_DF[col].rank(method='max')
                    UNIQUE_SONGS_DF[f'{col}_percentile'] = 100.0 * (ranks - 1) / (len(UNIQUE_SONGS_DF) - 1)
            
            # Process ALL_SONGS_DF - merge audio features and percentiles
            ALL_SONGS_DF = self._get_all_songs_df()
            
            # VECTORIZED PERCENTILE CALCULATION for ALL_SONGS_DF
            for col in percentile_cols:
                if col in ALL_SONGS_DF.columns and not ALL_SONGS_DF[col].isna().all():
                    # Vectorized percentile calculation
                    ranks = ALL_SONGS_DF[col].rank(method='max')
                    ALL_SONGS_DF[f'{col}_percentile'] = 100.0 * (ranks - 1) / (len(ALL_SONGS_DF) - 1)
            
            # Add audio feature columns to ALL_SONGS_DF
            for col in ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                       'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                       'tempo', 'valence', 'reccobeats_id']:
                if col in UNIQUE_SONGS_DF.columns:
                    ALL_SONGS_DF[col] = None
            
            # Add percentile columns to ALL_SONGS_DF
            for col in percentile_cols:
                if f'{col}_percentile' in UNIQUE_SONGS_DF.columns:
                    ALL_SONGS_DF[f'{col}_percentile'] = None
            
            # Merge audio features and percentiles
            try:
                # Check if required columns exist in both DataFrames
                if 'id' not in ALL_SONGS_DF.columns:
                    yield f"data:Warning: Required column 'id' not found in ALL_SONGS_DF. Available columns: {list(ALL_SONGS_DF.columns)}<br>\n\n\n"
                    raise Exception("Missing required column 'id' in ALL_SONGS_DF")
                
                if 'id' not in UNIQUE_SONGS_DF.columns:
                    yield f"data:Warning: Required column 'id' not found in UNIQUE_SONGS_DF. Available columns: {list(UNIQUE_SONGS_DF.columns)}<br>\n\n\n"
                    raise Exception("Missing required column 'id' in UNIQUE_SONGS_DF")
                
                # VECTORIZED OPERATION: Use map-based approach for efficient merging
                # Create a mapping dictionary from UNIQUE_SONGS_DF for fast lookups
                audio_feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                                    'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                                    'tempo', 'valence', 'reccobeats_id']
                
                # Select columns that exist in UNIQUE_SONGS_DF
                available_audio_cols = [col for col in audio_feature_cols if col in UNIQUE_SONGS_DF.columns]
                available_percentile_cols = [col for col in percentile_cols if f'{col}_percentile' in UNIQUE_SONGS_DF.columns]
                
                # Create mapping dictionaries for fast lookups
                id_to_features = {}
                for _, row in UNIQUE_SONGS_DF.iterrows():
                    song_id = row['id']
                    # Check for actual audio feature values (not empty strings or None)
                    audio_features = {}
                    for col in available_audio_cols:
                        if col in row:
                            value = row[col]
                            # Check if the value is actually meaningful (not None, NaN, or empty string)
                            if pd.notna(value) and value != '' and value != 'nan':
                                audio_features[col] = value
                    
                    percentiles = {}
                    for col in available_percentile_cols:
                        if f'{col}_percentile' in row:
                            value = row[f'{col}_percentile']
                            if pd.notna(value) and value != '' and value != 'nan':
                                percentiles[col] = value
                    
                    if audio_features or percentiles:  # Only store if we have meaningful data
                        id_to_features[song_id] = {
                            'audio': audio_features,
                            'percentiles': percentiles
                        }
                

                
                # Vectorized update using map and boolean indexing
                matches_found = 0
                features_updated = 0
                percentiles_updated = 0
                
                for idx, row in ALL_SONGS_DF.iterrows():
                    # IDs are already strings, not lists
                    spotify_id = row['id']
                    if spotify_id in id_to_features:
                        matches_found += 1
                        
                        # Update audio features
                        for col in available_audio_cols:
                            if col in id_to_features[spotify_id]['audio']:
                                value = id_to_features[spotify_id]['audio'][col]
                                if pd.notna(value) and value != '' and value != 'nan':
                                    ALL_SONGS_DF.at[idx, col] = value
                                    features_updated += 1
                        
                        # Update percentiles
                        for col in available_percentile_cols:
                            if col in id_to_features[spotify_id]['percentiles']:
                                value = id_to_features[spotify_id]['percentiles'][col]
                                if pd.notna(value) and value != '' and value != 'nan':
                                    ALL_SONGS_DF.at[idx, f'{col}_percentile'] = value
                                    percentiles_updated += 1
                

                
            except Exception as e:
                yield f"data:Warning: Could not merge to all songs DataFrame: {str(e)}<br>\n\n\n"
            
            # Step 6: Caching Your Data...
            yield "data:PROGRESS: 6/6 Caching Your Data...<br>\n\n\n"
            
            # Export UNIQUE_SONGS_DF
            self._atomic_gzip_ndjson_write(f'{self.path}unique_songs.ndjson.gz', UNIQUE_SONGS_DF)
            
            # Export ALL_SONGS_DF
            self._atomic_gzip_ndjson_write(f'{self.path}all_songs.ndjson.gz', ALL_SONGS_DF)
            
            # Export to CSV for specific user ID
            if self.USER_ID == 'qf26s87ilixm0wn6njz7amx2f':
                try:
                    # Export unique_songs_df to CSV
                    unique_csv_path = '.data/unique_songs_df.csv'
                    UNIQUE_SONGS_DF.to_csv(unique_csv_path, index=False)
                    
                    # Export all_songs_df to CSV (in case it was updated)
                    all_csv_path = '.data/all_songs_df.csv'
                    ALL_SONGS_DF.to_csv(all_csv_path, index=False)
                except Exception as e:
                    yield f'data:Warning: CSV export failed: {str(e)}<br/>\n\n\n'
            
            # Update main setup status file

            status = {'setup1': True, 'setup2': True, 'setup3': False}
            # Clean up progress data since setup is complete
            if 'setup2_last_batch' in status:
                del status['setup2_last_batch']
            if 'setup2_processed_count' in status:
                del status['setup2_processed_count']
            self._atomic_json_write(os.path.join(self.path, 'setup_status.json'), status)

            
            # Progress data is now stored in setup_status.json and cleaned up above
            
            yield "data:PROGRESS: Setup 2 - Audio Features completed successfully!<br>\n\n\n"
            
        except Exception as e:
            # No file saving on error - only save files at Step 6
            yield f"data:Setup 2 failed with error: {str(e)}<br>\n\n\n"
            yield f"data:Traceback: {traceback.format_exc().replace(chr(10), '<br>')}<br>\n\n\n"
            raise

    def _get_unique_songs_df(self):
        # Changing cols will be condensed into a list = ex: unique song will have a col "playlist" = [playlist1, playlist2]
        try:
            ALL_SONGS_DF = pd.read_json(f'{self.path}all_songs.ndjson.gz', lines=True, compression='gzip')
        except Exception as e:
            # Check if file is corrupted by trying to read first few lines
            try:
                with gzip.open(f'{self.path}all_songs.ndjson.gz', 'rt', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line.strip():
                        raise Exception("all_songs.ndjson.gz file is empty or corrupted. Please restart Setup 1.")
                    # Try to parse first line as JSON
                    try:
                        json.loads(first_line.strip())
                        raise Exception(f"JSON parsing error in all_songs.ndjson.gz: {str(e)}. Please restart Setup 1.")
                    except json.JSONDecodeError:
                        raise Exception(f"Invalid JSON format in all_songs.ndjson.gz: {str(e)}. Please restart Setup 1.")
            except Exception as gzip_error:
                if "JSON parsing error" in str(gzip_error) or "Invalid JSON format" in str(gzip_error):
                    raise gzip_error
                else:
                    raise Exception(f"File corruption detected in all_songs.ndjson.gz: {str(e)}. Please restart Setup 1.")

        # OPTIMIZED GROUPBY: Use specific aggregation functions instead of lambda for better performance
        changing_cols = ['id', 'playlist', 'date_added']
        
        # Define aggregation strategy for each column type
        agg_dict = {}
        for col in ALL_SONGS_DF.columns:
            if col in changing_cols:
                if col == 'id':
                    agg_dict[col] = 'first'  # Keep first ID instead of creating list
                else:
                    agg_dict[col] = list  # Keep lists for playlist and date_added
            else:
                agg_dict[col] = 'first'  # Keep first value for non-changing columns
        
        # Perform optimized groupby with specific aggregation functions
        UNIQUE_SONGS_DF = ALL_SONGS_DF.groupby(['name', 'artist', 'album'], as_index=False).agg(agg_dict)
        
        duplicates = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF.duplicated(subset=['name', 'artist'], keep=False)]
        duplicates.sort_values(by=['name', 'popularity'], ascending=[True, False], inplace=True)
        duplicates_with_lower_popularity = duplicates.iloc[1::2]
        # duplicates_with_lower_popularity.to_csv('dups.csv')
        # All Night by Vamps should have 65 instead of 0 popularity
        UNIQUE_SONGS_DF.drop(duplicates_with_lower_popularity.index, inplace=True)
        
        # Audio features will be added in a separate step for better progress visibility
        
        # VECTORIZED PERCENTILE CALCULATION: Use pandas operations instead of apply for better performance
        existing_percentile_cols = [col for col in ['popularity', 'duration'] if col in UNIQUE_SONGS_DF.columns]
        for col in existing_percentile_cols:
            # Vectorized percentile calculation
            ranks = UNIQUE_SONGS_DF[col].rank(method='max')
            UNIQUE_SONGS_DF[col + '_percentile'] = 100.0 * (ranks - 1) / (len(UNIQUE_SONGS_DF) - 1)

        # VECTORIZED PLAYLIST COUNT: Use pandas operations instead of list comprehension
        UNIQUE_SONGS_DF['num_playlists'] = UNIQUE_SONGS_DF['playlist'].str.len()

        # Also write compressed NDJSON for filesystem mode
        self._atomic_gzip_ndjson_write(os.path.join(self.path, 'unique_songs.ndjson.gz'), UNIQUE_SONGS_DF)
        
        return UNIQUE_SONGS_DF


    def _get_all_songs_df(self):
        """Get the all songs DataFrame from the compressed file."""
        try:
            return pd.read_json(f'{self.path}all_songs.ndjson.gz', lines=True, compression='gzip')
        except Exception as e:
            print(f"Error reading all_songs.ndjson.gz: {e}")
            return pd.DataFrame()


    def _get_top_artists(self):
        # [Artist1, Artist2, Artist3]
        try:
            TOP_ARTISTS_SHORT = {i['name']:i['popularity'] for i in self.SPOTIFY.current_user_top_artists(
                time_range='short_term', limit=50)['items']}
            time.sleep(0.5)  # Rate limiting protection
            
            TOP_ARTISTS_MED = {i['name']:i['popularity'] for i in self.SPOTIFY.current_user_top_artists(
                time_range='medium_term', limit=50)['items']}
            time.sleep(0.5)  # Rate limiting protection
            
            TOP_ARTISTS_LONG = {i['name']:i['popularity'] for i in self.SPOTIFY.current_user_top_artists(
                time_range='long_term', limit=50)['items']}
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                raise Exception("Rate limited by Spotify API. Please wait a few minutes and try again.")
            raise
        TOP_ARTISTS_NAMES = [list(TOP_ARTISTS_SHORT.keys()), list(TOP_ARTISTS_MED.keys()), list(TOP_ARTISTS_LONG.keys())]
        TOP_ARTISTS_POPULARITY = [list(TOP_ARTISTS_SHORT.values()), list(TOP_ARTISTS_MED.values()), list(TOP_ARTISTS_LONG.values())]

        # JSON handles this now
        # Filesystem JSON mirrors
        self._atomic_json_write(os.path.join(self.path, 'top_artists.json'), TOP_ARTISTS_NAMES)
        self._atomic_json_write(os.path.join(self.path, 'top_artists_pop.json'), TOP_ARTISTS_POPULARITY)


    # INT or "N/A" if single, STR separated by commas if multiple
    def _add_top_artists_rank(self):
        try:
            # Check if file exists first
            if not os.path.exists(f'{self.path}unique_songs.ndjson.gz'):
                raise Exception("unique_songs.ndjson.gz file not found. Please complete Setup 2 first.")
            
            # Try to read the file with better error handling
            try:
                UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
                
                # Check if 'artist' column exists, if not, try alternative column names
                if 'artist' not in UNIQUE_SONGS_DF.columns:
                    available_cols = list(UNIQUE_SONGS_DF.columns)
                    raise Exception(f"Column 'artist' not found in UNIQUE_SONGS_DF. Available columns: {available_cols}. Please restart Setup 2.")
                    
            except Exception as json_error:
                # Check if file is corrupted by trying to read first few lines
                try:
                    with gzip.open(f'{self.path}unique_songs.ndjson.gz', 'rt', encoding='utf-8') as f:
                        first_line = f.readline()
                        if not first_line.strip():
                            raise Exception("unique_songs.ndjson.gz file is empty or corrupted. Please restart Setup 2.")
                        # Try to parse first line as JSON
                        json.loads(first_line.strip())
                        raise Exception(f"JSON parsing error in unique_songs.ndjson.gz: {str(json_error)}. Please restart Setup 2.")
                except Exception as gzip_error:
                    if "JSON parsing error" in str(gzip_error):
                        raise gzip_error
                    raise Exception(f"File corruption detected in unique_songs.ndjson.gz: {str(gzip_error)}. Please restart Setup 2.")
            
            if not os.path.exists(f'{self.path}top_artists.json'):
                raise Exception("top_artists.json file not found. Please complete Setup 1 first.")
                
            with open(f'{self.path}top_artists.json', 'r', encoding='utf-8') as f:
                TOP_ARTISTS = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read required files in _add_top_artists_rank: {str(e)}. Please ensure Setup 1 and Setup 2 are completed successfully.")

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
        try:
            TOP_SONGS_SHORT = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
                self.SPOTIFY.current_user_top_tracks(time_range='short_term', limit=50)['items'])}
            time.sleep(0.5)  # Rate limiting protection
            
            TOP_SONGS_MED = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
                self.SPOTIFY.current_user_top_tracks(time_range='medium_term', limit=50)['items'])}
            time.sleep(0.5)  # Rate limiting protection
            
            TOP_SONGS_LONG = {i['name']: (', '.join([j['name'] for j in i['artists']]), k+1) for k, i in enumerate(
                self.SPOTIFY.current_user_top_tracks(time_range='long_term', limit=50)['items'])}
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                raise Exception("Rate limited by Spotify API. Please wait a few minutes and try again.")
            raise
        TOP_SONGS = [TOP_SONGS_SHORT, TOP_SONGS_MED, TOP_SONGS_LONG]

        # Persist Top Songs as JSON
        self._atomic_json_write(os.path.join(self.path, 'top_songs.json'), TOP_SONGS)


    def _add_top_songs_rank(self):
        try:
            UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
        except Exception as e:
            raise Exception(f"Failed to read unique_songs.ndjson.gz: {str(e)}. The file may be corrupted. Please restart Setup 2.")
        
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
        try:
            ALL_SONGS_DF = pd.read_json(f'{self.path}all_songs.ndjson.gz', lines=True, compression='gzip')
        except Exception as e:
            raise Exception(f"Failed to read all_songs.ndjson.gz: {str(e)}. The file may be corrupted. Please restart Setup 2.")
        
        try:
            UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
        except Exception as e:
            raise Exception(f"Failed to read unique_songs.ndjson.gz: {str(e)}. The file may be corrupted. Please restart Setup 2.")

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
            try:
                listy = self.SPOTIFY.artists(unique_artist_ids[i:i+50])
                for a in listy['artists']:
                    try:
                        g = a['genres']
                    except Exception as e:
                        g = []
                    if g:
                        genres_dict[a['id']] = g
                    else:
                        genres_dict[a['id']] = ['N/A']
                
                # Yield progress update (single line that gets updated)
                collected = min(i + 50, total)
                yield f'data:PROGRESS:Step 5/10: Getting Artist Genres... ({collected}/{total} artists collected)<br>\n\n\n'
                
                # Rate limiting protection - wait between batches
                time.sleep(1.0)  # 1 second between batches of 50 artists
                
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    # If rate limited, wait longer and retry
                    time.sleep(30)
                    # Retry the same batch
                    try:
                        listy = self.SPOTIFY.artists(unique_artist_ids[i:i+50])
                        for a in listy['artists']:
                            try:
                                g = a['genres']
                            except Exception as e:
                                g = []
                            if g:
                                genres_dict[a['id']] = g
                            else:
                                genres_dict[a['id']] = ['N/A']
                        
                        # Yield progress update after retry (single line that gets updated)
                        collected = min(i + 50, total)
                        yield f'data:PROGRESS:Step 5/10: Getting Artist Genres... ({collected}/{total} artists collected)<br>\n\n\n'
                    except Exception as retry_error:
                        raise Exception(f"Rate limited by Spotify API. Please wait a few minutes and try again. Error: {str(retry_error)}")
                else:
                    raise

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
    def setup_3(self, ALL_SONGS_DF):
        total = 10
        try:
            # Check Setup 2 completion status from main status file
            status_file = f'{self.path}setup_status.json'
            
            # Wait for Setup 2 to complete (check every 2 seconds)
            max_wait_time = 300  # 5 minutes max wait
            wait_count = 0
            
            while wait_count < max_wait_time:
                try:
                    with open(status_file, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                    
                    if status.get('setup2', False):
                        break  # Setup 2 is complete, proceed directly to step 1
                    else:
                        yield f'data:Waiting for Setup 2 to complete... ({wait_count}s elapsed)<br>\n\n\n'
                        time.sleep(2)
                        wait_count += 2
                except Exception as e:
                    yield f'data:Waiting for Setup 2 to complete... ({wait_count}s elapsed)<br>\n\n\n'
                    time.sleep(2)
                    wait_count += 2
            
            if wait_count >= max_wait_time:
                yield 'data:ERROR=Setup 2 did not complete within expected time. Please restart Setup 2.<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise Exception("Setup 2 completion timeout")

            yield f'data:Step 1/10: Getting Top Artists...<br>\n\n\n'
            try:
                self._get_top_artists()
            except Exception as e:
                yield f'data:ERROR=Failed to get top artists: {str(e)}<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise

            yield f'data:Step 2/10: Getting Top Songs...<br>\n\n\n'
            try:
                self._get_top_songs()
            except Exception as e:
                yield f'data:ERROR=Failed to get top songs: {str(e)}<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise

            yield f'data:Step 3/10: Adding Top Artists Rank...<br>\n\n\n'
            try:
                self._add_top_artists_rank()
            except Exception as e:
                yield f'data:ERROR=Failed to add top artists rank: {str(e)}<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise

            yield f'data:Step 4/10: Adding Top Songs Rank...<br>\n\n\n'
            try:
                self._add_top_songs_rank()
            except Exception as e:
                yield f'data:ERROR=Failed to add top songs rank: {str(e)}<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise

            # Takes a while
            yield f'data:Step 5/10: Getting Artist Genres...<br>\n\n\n'
            try:
                for progress_message in self._add_genres():
                    yield progress_message
            except Exception as e:
                yield f'data:ERROR=Failed to add artist genres: {str(e)}<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise

            yield f'data:Step 6/10: Preparing data for pages...<br>\n\n\n'
            # Load required data files for page rendering (pages will be created on-demand)
            try:
                UNIQUE_SONGS_DF = pd.read_json(f'{self.path}unique_songs.ndjson.gz', lines=True, compression='gzip')
            except Exception as e:
                yield f'data:ERROR=Failed to load data files: {str(e)}<br>\n\n\n'
                yield f'data:Traceback: {traceback.format_exc().replace(chr(10), "<br>")}<br>\n\n\n'
                raise
            
            yield f'data:Step 7/10: Loading artist data...<br>\n\n\n'
            # Get followed artists for About page
            try:
                artists = self.SPOTIFY.current_user_followed_artists()['artists']['items']
                # Save to file for later use
                with open(f'{self.path}followed_artists.json', 'w', encoding='utf-8') as f:
                    json.dump(artists, f, ensure_ascii=False, indent=2)
                time.sleep(0.5)  # Rate limiting protection
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    yield f'data:Warning: Rate limited while loading followed artists. Skipping this step.<br>\n\n\n'
                else:
                    yield f'data:Warning: Could not load followed artists: {str(e)}<br>\n\n\n'
                artists = []

            yield f'data:Step 8/10: Finalizing data preparation...<br>\n\n\n'
            # Verify all required files exist
            required_files = [
                'top_artists.json', 'top_songs.json', 'top_artists_pop.json',
                'unique_songs.ndjson.gz', 'all_songs.ndjson.gz'
            ]
            missing_files = []
            for file in required_files:
                if not os.path.exists(f'{self.path}{file}'):
                    missing_files.append(file)
            
            if missing_files:
                yield f'data:Warning: Missing required files: {", ".join(missing_files)}<br>\n\n\n'
            else:
                yield f'data:All required data files verified successfully<br>\n\n\n'

            yield f'data:Step 9/10: Data preparation complete...<br>\n\n\n'

            yield f'data:Step 10/10: Finalizing Data Collection...<br>\n\n\n'
            
            # Write status immediately to ensure completion is recorded
            status = {'setup1': True, 'setup2': True, 'setup3': True}
            status_path = os.path.join(self.path, 'setup_status.json')
            self._atomic_json_write(status_path, status)
            
            # Verify the status was written
            if os.path.exists(status_path):
                yield 'data:Setup 3 - Top50 completed successfully!<br>\n\n\n'
            else:
                yield 'data:WARNING: Status file not found after completion.<br>\n\n\n'

        except Exception as e:
            # If we encounter a corrupted file error, clean up and reset setup status
            if "corrupted" in str(e).lower() or "zlib.error" in str(e) or "invalid stored block lengths" in str(e):
                yield f'data:ERROR=Corrupted data files detected. Cleaning up and resetting setup status...<br>\n\n\n'
                
                # Remove corrupted files
                corrupted_files = [
                    f'{self.path}unique_songs.ndjson.gz',
                    f'{self.path}all_songs.ndjson.gz'
                ]
                
                for file_path in corrupted_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            yield f'data:Removed corrupted file: {os.path.basename(file_path)}<br>\n\n\n'
                    except Exception as cleanup_error:
                        yield f'data:Warning: Could not remove {os.path.basename(file_path)}: {str(cleanup_error)}<br>\n\n\n'
                
                # Reset setup status to allow restart from setup_2
                try:
                    status = {'setup1': True, 'setup2': False, 'setup3': False}
                    status_path = os.path.join(self.path, 'setup_status.json')
                    self._atomic_json_write(status_path, status)
                    yield f'data:Setup status reset. Please restart Setup 2 to regenerate data files.<br>\n\n\n'
                except Exception as status_error:
                    yield f'data:Warning: Could not reset setup status: {str(status_error)}<br>\n\n\n'
                
                yield f'data:Please restart Setup 2 to regenerate the corrupted data files.<br>\n\n\n'
            else:
                yield 'data:ERROR=' + traceback.format_exc().replace('\n', '<br>') + '\n\n'