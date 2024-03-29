# App.py setup started from Spotipy template, see at
# https://github.com/plamere/spotipy/blob/master/examples/app.py

# -------------------------------Imports-----------------------------------------------

import pickle
import pandas as pd
from visualization import CurrentlyPlayingPage, AnalyzePlaylistsPage, AnalyzePlaylistPage, AnalyzeArtistPage, AnalyzeArtistsPage
from SetupData import SetupData
import datetime
import spotipy
import os
import uuid
import shutil
import tempfile
import traceback
from flask_caching import Cache
from flask_session import Session
from flask import Flask, session, request, redirect, render_template, Response, stream_with_context

CLIENT_ID = 'YOUR-CLIENT-ID-HERE'
CLIENT_SECRET = 'YOUR-CLIENT-SECRET-HERE'

# Test Local
REDIRECT_URI = 'http://127.0.0.1:5000/'
# Run Heroku
# REDIRECT_URI = 'https://spotify-statys.herokuapp.com/'

# -------------------------------Data-----------------------------------------------

# Matplotlib Temp Files
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name

# Unpickle Python Objects
def _load(path, name):
    with open(path + name, 'rb') as f: 
        return pickle.load(f)

# Creating Flask App
app = Flask(__name__, template_folder='templates')

# this is a server-side type session without the client-side limit of 4 KB
app.config['SESSION_TYPE'] = 'filesystem'
# directory for storing user data
app.config['SESSION_FILE_DIR'] = './.flask_session/'
# if user doesn't log out their data will only be saved for a limited time
app.config["SESSION_PERMANENT"] = False

app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

cache.init_app(app)
# activates the Flask-Session object AFTER 'app' has been configured
Session(app)

# Setting environment variables
os.environ['SPOTIPY_CLIENT_ID'] = CLIENT_ID
os.environ['SPOTIPY_CLIENT_SECRET'] = CLIENT_SECRET
os.environ['SPOTIPY_REDIRECT_URI'] = REDIRECT_URI

# Creating Cache Folders
cache_folder = './.spotify_caches/'
cache_folder2 = './.sp_caches/'
for i in [cache_folder, cache_folder2]:
    if not os.path.exists(i):
        os.makedirs(i)

def spotify_cache_path():
    return cache_folder + session.get('uuid')

def sp_cache_path():
    return cache_folder2 + session.get('uuid')

# -------------------------------Web Page Routes-----------------------------------------------

# Setup1 = Collecting Playlist Data, Called by setup.html JS as EventSource
@app.route('/setup_1')
def setup_1():
    return Response(session['setup'].setup_1(), content_type='text/event-stream')


# Setup2 = Grouping Data, Called by setup.html JS as EventSource
@app.route('/setup_2')
def setup_2():
    return Response(session['setup'].setup_2(session['ALL_SONGS_DF']), content_type='text/event-stream')


# Login and Setup
@app.route('/')
def index():
    if not session.get('uuid'):
        # Step 1. Visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())

    session['CACHE_HANDLER'] = spotipy.cache_handler.CacheFileHandler(
        cache_path=spotify_cache_path())
    session['AUTH_MANAGER'] = spotipy.oauth2.SpotifyOAuth(
        scope='user-read-currently-playing playlist-read-private user-top-read user-follow-read',
                                                          cache_handler=session['CACHE_HANDLER'],
                                                          show_dialog=True)

    if request.args.get("code"):
        # Step 3. Being redirected from Spotify auth page
        session['AUTH_MANAGER'].get_access_token(request.args.get("code"))
        # Have to refresh page otherwise gets stuck loading at Spotify auth page
        return '<script>window.location.href="' + REDIRECT_URI + '"</script>'

    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        # Step 2. Display sign in link when no token
        auth_url = session['AUTH_MANAGER'].get_authorize_url()
        return f'<h2><a href="{auth_url}">Sign in</a></h2>'

    # Step 4. Signed in, Setup1 = Collect Playlists
    if not session.get('setup'):
        # SPOTIFY = for Current User Data
        session['SPOTIFY'] = spotipy.Spotify(auth_manager=session['AUTH_MANAGER'])
        # SP = for Public Data, need a separate Cache Handler otherwise cannot write token to .cache
        client_credentials = spotipy.oauth2.SpotifyClientCredentials(
            cache_handler=spotipy.cache_handler.CacheFileHandler(cache_path=sp_cache_path()))
        session['SP'] = spotipy.Spotify(auth_manager=client_credentials)

        session['setup'] = SetupData(session)

        session['USER_ID'] = session['setup'].USER_ID
        user_id = session['USER_ID']
        session['PATH'] = f'.data/{user_id}/'
        session['PLAYLIST_DICT'] = session['setup'].PLAYLIST_DICT

        return render_template('setup.html', setup2 = "false")

    status = _load(session['PATH'], 'collection.pkl')

    # Off case that Setup1 was in progress but did not finish
    if not status['SETUP1']:
        return render_template('setup.html', setup2 = "false")

    # Setup2 = Group Data
    if status['SETUP1'] and not status['SETUP2']:
        path = session['PATH']
        session['ALL_SONGS_DF'] = pd.read_pickle(f'{path}all_songs_df.pkl')
        
        return render_template('setup.html', setup2 = "true")

    if status['SETUP2'] and not status['SETUP3']:
        path = session['PATH']
        session['ALL_SONGS_DF'] = pd.read_pickle(f'{path}all_songs_df.pkl')
        session['UNIQUE_SONGS_DF'] = pd.read_pickle(f'{path}unique_songs_df.pkl')

        session['TOP_ARTISTS'] = _load(path, 'top_artists.pkl')
        session['TOP_SONGS'] = _load(path, 'top_songs.pkl')

        session['HOME_PAGE'] = _load(path, 'home_page.pkl')
        session['ABOUT_PAGE'] = _load(path, 'about_page.pkl')
        session['TOP50_PAGE'] = _load(path, 'top50_page.pkl')

    return redirect('/home')


# Home Page
@app.route('/home')
@cache.cached(timeout=300)
def home():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')

    page = session['HOME_PAGE']

    on_this_date = page.load_on_this_date()
    full_timeline = page.load_timeline()
    last_added_playlist = page.load_last_added()
    overall_data = page.load_totals()
    first_times = page.load_first_times()

    return render_template('home.html', collection=False,
                           name=session['SPOTIFY'].me()['display_name'], today=str(
                               datetime.datetime.now().astimezone().date())[5:],
                           on_this_date=on_this_date,
                           full_timeline=full_timeline,
                           last_added_playlist=last_added_playlist,
                           overall_data=overall_data, first_times=first_times
                           )


# Error Page
@app.route('/retry')
def retry(traceback):
    return render_template('retry.html', traceback=traceback)


# Delete all user data and return to sign-in
@app.route('/sign-out')
def sign_out():
    try:
        os.remove(spotify_cache_path())
        os.remove(sp_cache_path())
        shutil.rmtree(session['PATH'])
        session.clear()
        cache.clear()
    except OSError as e:
        return render_template('retry.html', traceback=traceback.format_exc())
    return redirect('/')


# Currently Playing Page
@app.route('/currently-playing')
def currently_playing():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    track = session['SPOTIFY'].current_user_playing_track()

    if not track:
        return "<h3>No track currently playing</h3>"

    artist = ', '.join([listy['name'] for listy in track['item']['artists']])
    song = track['item']['name']

    # See if User is playing one of their playlists
    if not track['context'] is None and not track['context']['uri'] is None:
        playlist_id = track['context']['uri']
        playlist_id = playlist_id[playlist_id.rfind(':')+1:]
        lookup = {value: key for key,
                    value in session['PLAYLIST_DICT'].items()}
        if playlist_id in lookup:
            playlist = lookup[str(playlist_id)]
        else:
            playlist = None
    else:
        playlist = None

    df = session['UNIQUE_SONGS_DF']
    df = df[(df['name'] == song) & (df['artist'] == artist)]
    song_in_playlist = True if len(df) > 0 else False
    if not song_in_playlist:
        return f'<h3>You are playing the song {song} by {artist} - but it\'s not in any of your created playlists :(</h3>' \
            f'<h3>Start listening to any song from your playlists to see cool stats!</h3>'
    else:

        page = CurrentlyPlayingPage(
            song, artist, playlist, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

        top_rank_table = page.graph_top_rank_table()

        song_features_radar = page.graph_song_features_vs_avg()
        song_features_percentiles_bar = page.graph_song_percentiles_vs_avg()

        playlist_date_gantt = page.graph_date_added_to_playlist()
        playlist_timeline = page.graph_count_timeline()

        artist_top_graphs = page.graph_all_artists()

        artist_genres = page.graph_artist_genres()
        genres_playlist_percentiles = page.graph_song_genres_vs_avg(
            playlist=True) if playlist else None
        genres_overall_percentiles = page.graph_song_genres_vs_avg()

        return render_template('currently_playing.html', song=song, artist=artist, playlist=playlist,
                                top_rank_table=top_rank_table,
                                song_features_radar=song_features_radar, song_features_percentiles_bar=song_features_percentiles_bar,
                                playlist_date_gantt=playlist_date_gantt, playlist_timeline=playlist_timeline,
                                artist_top_graphs=artist_top_graphs,
                                artist_genres=artist_genres, genres_playlist_percentiles=genres_playlist_percentiles, genres_overall_percentiles=genres_overall_percentiles
                                )


# About Me Page
@app.route('/about-me')
@cache.cached(timeout=300)
def about_me():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')

    page = session['ABOUT_PAGE']

    top_genres_by_followed_artists_bar = page.load_followed_artists()
    top_playlists_by_songs_bubble = page.load_playlists_by_songs()

    top_playlists_by_artists_bubble = page.load_playlists_by_artists()
    top_songs_by_num_playlists = page.load_top_songs()
    top_artists_by_all_playlists = page.load_top_artists()
    top_albums_by_all_playlists = page.load_top_albums()

    return render_template('about_me.html', top_genres_by_followed_artists_bar=top_genres_by_followed_artists_bar,
                           top_playlists_by_songs_bubble=top_playlists_by_songs_bubble, top_playlists_by_artists_bubble=top_playlists_by_artists_bubble,
                           top_songs_by_num_playlists=top_songs_by_num_playlists,
                           top_artists_by_all_playlists=top_artists_by_all_playlists,
                           top_albums_by_all_playlists=top_albums_by_all_playlists)


# Top 50 Page
@app.route('/top-50')
@cache.cached(timeout=300)
def top_50():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')

    page = session['TOP50_PAGE']
    plots_by_time_range = page.load_dynamic_graph()

    return render_template('top_50.html', plots_by_time_range=plots_by_time_range)


# Search Page
@app.route('/search')
def search():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    return render_template('search.html')


# Single Playlist
def analyze_playlist(playlist_name):
    page = AnalyzePlaylistPage(
        playlist_name, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

    timeline = page.graph_count_timeline()
    genres = page.graph_playlist_genres()

    top_artists = page.graph_top_artists()
    top_albums = page.graph_top_albums()

    features_boxplot = page.graph_song_features_boxplot()

    similar_playlists = page.graph_similar_playlists()

    return render_template('analyze_playlist.html', playlist_name=playlist_name,
                           timeline=timeline, genres=genres,
                           top_artists=top_artists, top_albums=top_albums,
                           features_boxplot=features_boxplot,
                           similar_playlists=similar_playlists
                           )


# Multiple Playlists
def analyze_playlists(playlists):
    page = AnalyzePlaylistsPage(
        playlists, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

    intersection = page.get_intersection_of_playlists()

    boxplot = intersection[0]
    length = intersection[1]
    names = ', '.join(playlists)

    playlist_timelines = page.graph_playlist_timelines()
    genres = page.graph_genres_by_playlists()

    return render_template('analyze_playlists.html', boxplot=boxplot, length=length, playlists=names,
                           playlist_timelines=playlist_timelines, genres=genres
                           )


# Single Artist
def analyze_artist(artist_name):
    page = AnalyzeArtistPage(
        artist_name, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

    timeline = page.graph_count_timeline()
    top_rank_table = page.graph_top_rank_table()

    top_playlists = page.graph_top_playlists_by_artist()
    top_songs = page.graph_top_songs_by_artist()

    features_boxplot = page.graph_song_features_boxplot()

    artist_genres = page.artist_genres()
    playlists_genres = page.graph_playlists_by_artist_genres()

    return render_template('analyze_artist.html', artist_name=artist_name,
                           timeline=timeline, top_rank_table=top_rank_table,
                           top_playlists=top_playlists, top_songs=top_songs,
                           features_boxplot=features_boxplot,
                           artist_genres=artist_genres,
                           playlists_genres=playlists_genres
                           )


# Multiple Artists
def analyze_artists(artists):
    page = AnalyzeArtistsPage(
        artists, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

    artist_timelines = page.graph_artist_timelines()
    genres = page.graph_artist_genres()

    return render_template('analyze_artists.html', artists=', '.join(artists), artist_timelines=artist_timelines,
                           genres=genres)


# Parse Search Query and Load Proper Page
@app.route('/analyze-search', methods=['GET', 'POST'])
def analyze_search():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    if request.method == 'GET':
        return f"The URL is accessed directly. Try going to '/search' to submit form"
    if request.method == 'POST':
        query = [i.strip() for i in request.form['query'].split(',')]

        lowercase_playlists = {i.lower(): i
                               for i in session['PLAYLIST_DICT'].keys()}
        lowercase_artists = {i.lower(): i
                             for i in session['ALL_SONGS_DF']['artist'].unique()}
        found_list = []
        found_playlist = False
        found_artist = False
        for i in query:
            if i.lower() in lowercase_playlists:
                found_list.append(lowercase_playlists[i])
                found_playlist = True
            elif i.lower() in lowercase_artists:
                found_list.append(lowercase_artists[i])
                found_artist = True
            else:
                return 'Query not found - make sure you spelled correctly!'

            if found_playlist and found_artist:
                return 'Query not found - make sure you inputted only playlists or only artists'

        if found_playlist:
            if len(found_list) == 1:
                return analyze_playlist(found_list[0])
            else:
                return analyze_playlists(found_list)
        elif found_artist:
            if len(found_list) == 1:
                return analyze_artist(found_list[0])
            else:
                return analyze_artists(found_list)

# -------------------------------Main Method-----------------------------------------------

if __name__ == '__main__':
    # threaded=True is default, meaning that multiple requests can be handled at once
    app.run()
