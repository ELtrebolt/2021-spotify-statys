# App setup based on Spotipy template, see at
# https://github.com/plamere/spotipy/blob/master/examples/app.py

# Windows Command Prompt for Local "flask run"
# SET SPOTIPY_CLIENT_ID=YOUR-ID-HERE
# SET SPOTIPY_CLIENT_SECRET=YOUR-SECRET-HERE
# SET SPOTIPY_REDIRECT_URI=http://127.0.0.1:5000/

# Spotify Constants - remove all instances if running local, otherwise keep for Heroku
import pickle
import pandas as pd
from visualization import CurrentlyPlayingPage, ComparePlaylistsPage, AnalyzePlaylistPage, Top50Page
from SetupData import SetupData
import datetime
import spotipy
import tempfile
import os
import uuid
import shutil
from flask_caching import Cache
from flask_session import Session
from flask import Flask, session, request, redirect, render_template, Response
CLIENT_ID = 'YOUR-KEY-HERE'
CLIENT_SECRET = 'YOUR-SECRET-HERE'
# Test Local = http://127.0.0.1:5000/
# Run Heroku = https://YOUR-APP-HERE.herokuapp.com/
REDIRECT_URI = 'https://YOUR-APP-HERE.herokuapp.com/'

# -------------------------------Imports-----------------------------------------------

# Web Integration
# import dash
# import dash_core_components as dcc
# import dash_html_components as html

temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name

# Data

def _load(path, name):
    with open(path + name, 'rb') as f:  # Unpickling
        return pickle.load(f)

# -------------------------------Web Page Routes-----------------------------------------------


# Creating Flask App
app = Flask(__name__, template_folder='templates')
# app.config['SECRET_KEY'] = os.urandom(64)               # Do NOT use for Heroku, instead set the secret key as an environment variable in Heroku
# this is a server-side type session without the client-side limit of 4 KB
app.config['SESSION_TYPE'] = 'filesystem'
# directory for storing user data
app.config['SESSION_FILE_DIR'] = './.flask_session/'
# if user doesn't log out their data will only be saved for a limited time
app.config["SESSION_PERMANENT"] = False
# SECRET_KEY = os.getenv('SECRET_KEY', 'Optional default value')

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
# app.config.from_mapping(SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev_key')

# Creating Cache Folder
caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

# Get files for specific user at cache-folder/user-id/


def session_cache_path():
    return caches_folder + session.get('uuid')

# Home Page - check login first


@app.route('/')
def index():
    if not session.get('uuid'):
        # Step 1. Visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())

    session['CACHE_HANDLER'] = spotipy.cache_handler.CacheFileHandler(
        cache_path=session_cache_path())
    session['AUTH_MANAGER'] = spotipy.oauth2.SpotifyOAuth(scope='user-read-currently-playing playlist-read-private user-top-read user-follow-read',
                                                          cache_handler=session['CACHE_HANDLER'],
                                                          show_dialog=True)

    if request.args.get("code"):
        # Step 3. Being redirected from Spotify auth page
        session['AUTH_MANAGER'].get_access_token(request.args.get("code"))
        return redirect('/')

    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        # Step 2. Display sign in link when no token
        auth_url = session['AUTH_MANAGER'].get_authorize_url()
        return f'<h2><a href="{auth_url}">Sign in</a></h2>'

    # Step 4. Signed in, display data
    if not session.get('setup'):
        session['SPOTIFY'] = spotipy.Spotify(
            auth_manager=session['AUTH_MANAGER'])
        client_credentials = spotipy.oauth2.SpotifyClientCredentials()
        session['SP'] = spotipy.Spotify(auth_manager=client_credentials)

        session['setup'] = SetupData(session)

        session['USER_ID'] = session['setup'].USER_ID
        user_id = session['USER_ID']
        session['PATH'] = f'.data/{user_id}/'
        session['PLAYLIST_DICT'] = session['setup'].PLAYLIST_DICT

        return Response(session['setup'].setup_1(), mimetype='text/html')
        # return render_template('index.html', collection=True)
        # return render_template('get_playlists.html.jinja')

    status = _load(session['PATH'], 'collection.pkl')

    if not status['SETUP1']:
        return Response(session['setup'].setup_1(), mimetype='text/html')

    if status['SETUP1'] and not status['SETUP2']:
        path = session['PATH']
        session['ALL_SONGS_DF'] = pd.read_pickle(f'{path}all_songs_df.pkl')

        return Response(session['setup'].setup_2(session['ALL_SONGS_DF']), mimetype='text/html')

    if status['SETUP2'] and not status['SETUP3']:
        path = session['PATH']
        session['ALL_SONGS_DF'] = pd.read_pickle(f'{path}all_songs_df.pkl')
        session['UNIQUE_SONGS_DF'] = pd.read_pickle(
            f'{path}unique_songs_df.pkl')

        session['TOP_ARTISTS'] = _load(path, 'top_artists.pkl')
        session['TOP_SONGS'] = _load(path, 'top_songs.pkl')

        session['HOME_PAGE'] = _load(path, 'home_page.pkl')
        session['ABOUT_PAGE'] = _load(path, 'about_page.pkl')
        session['TOP50_PAGE'] = _load(path, 'top50_page.pkl')

    return redirect('/home')
    # except Exception as e:
    #    return render_template('retry.html', error=e, function='Home Page')
    # f'<h2>Hi {spotify.me()["display_name"]}, ' \
    # f'<small><a href="/sign_out">[sign out]<a/></small></h2>' \
    # f'<a href="/playlists">my playlists</a> | ' \
    # f'<a href="/currently_playing">currently playing</a> | ' \
    # f'<a href="/current_user">me</a>' \


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

    return render_template('home.html', collection=False,
                           name=session['SPOTIFY'].me()['display_name'], today=str(
                               datetime.datetime.now().astimezone().date())[5:],
                           on_this_date=on_this_date,
                           full_timeline=full_timeline,
                           last_added_playlist=last_added_playlist,
                           overall_data=overall_data,
                           )


@app.route('/retry/<function>/<error>')
def retry(function, error):
    return render_template('retry.html', error=error, function=function)


@app.route('/sign_out')
def sign_out():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        os.remove(session_cache_path())
        shutil.rmtree(session['PATH'])
        session.clear()
        cache.clear()
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    return redirect('/')


@app.route('/currently_playing')
def currently_playing():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    track = session['SPOTIFY'].current_user_playing_track()

    if not track is None:
        artist = ', '.join([listy['name']
                           for listy in track['item']['artists']])
        song = track['item']['name']
        # song_id = track['item']['id']

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

        if playlist_id:
            return render_template('currently_playing.html', song=song, artist=artist, playlist=playlist,
                                   top_rank_table=top_rank_table,
                                   song_features_radar=song_features_radar, song_features_percentiles_bar=song_features_percentiles_bar,
                                   playlist_date_gantt=playlist_date_gantt, playlist_timeline=playlist_timeline,
                                   artist_top_graphs=artist_top_graphs,
                                   artist_genres=artist_genres, genres_playlist_percentiles=genres_playlist_percentiles, genres_overall_percentiles=genres_overall_percentiles,

                                   )

        # div = graph_rank_table(UNIQUE_SONGS_DF, 'energy', 10)

    return "No track currently playing."


@app.route('/about_me')
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


@app.route('/choose_playlists', methods=['GET', 'POST'])
def choose_playlists():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    return render_template('choose_playlists.html', playlists=session['PLAYLIST_DICT'].keys())


@app.route('/compare_playlists', methods=['POST', 'GET'])
def compare_playlists():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    if request.method == 'GET':
        return f"The URL /compare_playlists is accessed directly. Try going to '/choose_playlists' to submit form"
    if request.method == 'POST':
        playlists = [i.strip() for i in request.form['query'].split(',')]
        playlist_names = []
        for i in playlists:
            found = False
            for j in session['PLAYLIST_DICT'].keys():
                if i.lower() == j.lower():
                    playlist_names.append(j)
                    found = True
                    break
            if not found:
                return 'Playlists not found - make sure you spelled correctly!'
        playlists = playlist_names

        page = ComparePlaylistsPage(
            playlists, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

        df = page.get_intersection_of_playlists()
        playlist_timelines = page.graph_playlist_timelines()
        playlist_timelines_continuous = page.graph_playlist_timelines(
            continuous=True)

        songs = df[0]
        length = df[1]
        names = ', '.join(playlists)

        return render_template('compare_playlists.html', df=songs, length=length, playlists=names,
                               playlist_timelines=playlist_timelines, playlist_timelines_continuous=playlist_timelines_continuous
                               )


@app.route('/choose_playlist', methods=['GET', 'POST'])
def choose_playlist():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    PLAYLIST_DICT = session['PLAYLIST_DICT']
    if request.method == 'POST':
        query = request.form['query']
        search_results = [
            i for i in PLAYLIST_DICT if query.lower() in i.lower()]
        return render_template('choose_playlist.html', playlists=PLAYLIST_DICT.keys(), search_results=search_results)
    return render_template('choose_playlist.html', playlists=PLAYLIST_DICT.keys())


@app.route('/analyze_playlist', methods=['POST', 'GET'])
def analyze_playlist():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')
    if request.method == 'GET':
        return f"The URL /analyze_playlist is accessed directly. Try going to '/choose_playlist' to submit form"
    if request.method == 'POST':
        playlist = request.form['query']
        for i in session['PLAYLIST_DICT'].keys():
            if playlist.lower() == i.lower():
                page = AnalyzePlaylistPage(
                    i, session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

                timeline = page.graph_count_timeline()
                genres = page.graph_playlist_genres()

                top_artists = page.graph_top_artists()
                top_albums = page.graph_top_albums()

                features_boxplot = page.graph_song_features_boxplot()

                similar_playlists = page.graph_similar_playlists()

                return render_template('analyze_playlist.html', playlist_name=i,
                                       timeline=timeline, genres=genres,
                                       top_artists=top_artists, top_albums=top_albums,
                                       features_boxplot=features_boxplot,
                                       similar_playlists=similar_playlists
                                       )

        return 'Playlist not found - make sure there are no typos and remove any unnecessary spaces'


@app.route('/top_50')
@cache.cached(timeout=300)
def top_50():
    if not session['AUTH_MANAGER'].validate_token(session['CACHE_HANDLER'].get_cached_token()):
        return redirect('/')

    page = session['TOP50_PAGE']

    plots_by_time_range = page.load_dynamic_graph()

    return render_template('top_50.html', plots_by_time_range=plots_by_time_range)

# -------------------------------Web Page Routes-----------------------------------------------


'''
Following lines allow application to be run more conveniently with
`python app.py` (Make sure you're using python3)
(Also includes directive to leverage pythons threading capacity.)
'''
if __name__ == '__main__':
    # threaded=True is default, meaning that multiple requests can be handled at once
    app.run()
