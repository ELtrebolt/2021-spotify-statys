# App.py setup started from Spotipy template, see at
# https://github.com/plamere/spotipy/blob/master/examples/app.py

# -------------------------------Imports-----------------------------------------------

import pickle
import pandas as pd
from visualization import CurrentlyPlayingPage, AnalyzePlaylistsPage, AnalyzePlaylistPage,AnalyzeArtistPage, AnalyzeArtistsPage, SingleSongPage, MyPlaylistsPage
from SetupData import SetupData
import datetime
import spotipy
import os
import uuid
import shutil
import tempfile
import traceback
import json
import threading
import base64
import hashlib
import hmac
from flask_caching import Cache
from flask import Flask, request, redirect, render_template, Response, jsonify, make_response, g, session
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from spotipy.cache_handler import CacheFileHandler

load_dotenv()

# -------------------------------Data-----------------------------------------------

# Matplotlib Temp Files
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name

# Global progress tracking for PythonAnywhere compatibility
progress_data = {}

# Secret key for signing cookies (in production, use a secure random key)
SECRET_KEY = 'your-secret-key-here-change-in-production'

# Creating Flask App
app = Flask(__name__, template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config['PREFERRED_URL_SCHEME'] = 'https'

# Create local folders for session and cache persistence
session_folder = './.flask_session/'
cache_fs_folder = './.flask_cache/'
for d in [session_folder, cache_fs_folder]:
    if not os.path.exists(d):
        os.makedirs(d)

# Flask secret key (use env var in production) – keep for any flashes/CSRF, but disable Flask-Session
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', SECRET_KEY)

# File-system caching so it persists across reloads
app.config['CACHE_TYPE'] = 'FileSystemCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
app.config['CACHE_DIR'] = cache_fs_folder
cache = Cache(app)
cache.init_app(app)

# -------------------------------Middleware-----------------------------------------------

@app.before_request
def before_request():
    """Check and refresh tokens before each request to prevent 401 errors."""
    # Skip for authentication routes
    if request.endpoint in ['callback', 'login', 'static']:
        return
    
    # Check if user is authenticated
    if 'access_token' in session:
        access_token = session.get('access_token')
        refresh_token = session.get('refresh_token')
        expires_at = session.get('expires_at')
        
        if access_token and refresh_token and expires_at:
            # Check if token will expire soon (within 2 minutes)
            current_time = datetime.datetime.utcnow().timestamp()
            if current_time >= (expires_at - 120):  # 2 minutes buffer
                # Proactively refresh token
                new_token_info = refresh_user_token(refresh_token)
                if not new_token_info:
                    # Refresh failed, redirect to login
                    session.clear()
                    return redirect('/login')

# -------------------------------Authentication Helpers-----------------------------------------------

def refresh_user_token(refresh_token):
    """Refresh user's access token using refresh token."""
    try:
        auth_manager = create_auth_manager()
        new_token_info = auth_manager.refresh_access_token(refresh_token)
        
        if new_token_info and 'access_token' in new_token_info:
            # Update session with new token info
            session['access_token'] = new_token_info['access_token']
            if 'expires_at' in new_token_info:
                session['expires_at'] = new_token_info['expires_at']
            if 'refresh_token' in new_token_info:
                session['refresh_token'] = new_token_info['refresh_token']
            
            # Update cache file if user_id can be determined
            user_id = get_user_id_from_token(new_token_info['access_token'])
            if user_id:
                user_path = f'.data/{user_id}/'
                cache_handler = CacheFileHandler(cache_path=f"{user_path}.cache")
                cache_handler.save_token_to_cache(new_token_info)
            
            return new_token_info
    except Exception as e:
        # Log the error for debugging (in production, use proper logging)
        print(f"Token refresh failed: {str(e)}")
    
    return None

def handle_spotify_api_error(e, refresh_token):
    """Handle Spotify API errors and attempt token refresh if needed."""
    error_str = str(e).lower()
    
    # Check if error is related to expired/invalid token
    if any(keyword in error_str for keyword in ['expired', 'invalid', '401', 'unauthorized']):
        if refresh_token:
            new_token_info = refresh_user_token(refresh_token)
            if new_token_info:
                return new_token_info['access_token']
    
    return None

def build_redirect_uri():
    """Require explicit SPOTIPY_REDIRECT_URI for exact-match with Spotify settings.

    Set SPOTIPY_REDIRECT_URI to your exact callback URL, e.g.,
    https://<subdomain>.pythonanywhere.com/callback
    """
    load_dotenv(override=False)
    redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI', '').strip()
    if not redirect_uri:
        raise RuntimeError(
            'Missing SPOTIPY_REDIRECT_URI. Set it to your exact callback, e.g., '
            'https://spotifystatys.pythonanywhere.com/callback'
        )
    return redirect_uri

def create_auth_manager():
    """Create a new SpotifyOAuth instance using current env vars."""
    # Ensure latest env vars are loaded (useful with Flask reload)
    load_dotenv(override=False)
    client_id = os.getenv('SPOTIPY_CLIENT_ID', '').strip()
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET', '').strip()
    if not client_id or not client_secret:
        raise RuntimeError('Missing SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_SECRET. Set them in .env or environment.')
    return spotipy.oauth2.SpotifyOAuth(
        scope='user-read-currently-playing playlist-read-private user-top-read user-follow-read user-library-read',
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=build_redirect_uri(),
        show_dialog=False  # Don't show dialog on subsequent logins
    )

def get_user_id_from_token(access_token):
    """Get user ID from access token"""
    try:
        temp_spotify = spotipy.Spotify(auth=access_token)
        user_info = temp_spotify.me()
        return user_info['id']
    except:
        return None

def create_secure_cookie(value, max_age=86400):  # 24 hours
    """Create a secure signed cookie"""
    # Create signature
    signature = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()
    cookie_value = f"{value}.{signature}"
    
    # Encode to base64 for safe storage
    encoded = base64.b64encode(cookie_value.encode()).decode()
    
    response = make_response()
    response.set_cookie('spotify_token', encoded, max_age=max_age, httponly=True, secure=False)  # secure=False for HTTP
    return response

def verify_secure_cookie(cookie_value):
    """Verify and extract value from signed cookie"""
    try:
        # Decode from base64
        decoded = base64.b64decode(cookie_value.encode()).decode()
        
        # Split value and signature
        value, signature = decoded.rsplit('.', 1)
        
        # Verify signature
        expected_signature = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()
        
        if hmac.compare_digest(signature, expected_signature):
            return value
        return None
    except:
        return None

def get_current_user():
    """Get current user info from server-side session with automatic token refresh."""
    access_token = session.get('access_token')
    refresh_token = session.get('refresh_token')
    expires_at = session.get('expires_at')
    
    if not access_token:
        return None

    # Check if token is expired or will expire soon (within 5 minutes)
    current_time = datetime.datetime.utcnow().timestamp()
    if expires_at and current_time >= (expires_at - 300):  # 5 minutes buffer
        if refresh_token:
            new_token_info = refresh_user_token(refresh_token)
            if new_token_info:
                access_token = new_token_info['access_token']
            else:
                # Refresh failed, clear session
                session.clear()
                return None
        else:
            # No refresh token, clear session
            session.clear()
            return None

    try:
        spotify = spotipy.Spotify(auth=access_token)
        user_info = spotify.me()
        return {
            'id': user_info['id'],
            'display_name': user_info['display_name'],
            'access_token': access_token,
            'spotify_client': spotify
        }
    except Exception as e:
        # If API call fails, try to refresh token one more time
        if refresh_token:
            new_access_token = handle_spotify_api_error(e, refresh_token)
            if new_access_token:
                try:
                    # Try API call again with new token
                    spotify = spotipy.Spotify(auth=new_access_token)
                    user_info = spotify.me()
                    return {
                        'id': user_info['id'],
                        'display_name': user_info['display_name'],
                        'access_token': new_access_token,
                        'spotify_client': spotify
                    }
                except:
                    pass
        
        # All attempts failed, clear session
        session.clear()
        return None

# -------------------------------Web Page Routes-----------------------------------------------

@app.route('/login')
def login():
    """Handle login redirects and provide user-friendly authentication page."""
    # Clear any existing session data
    session.clear()
    
    try:
        auth_manager = create_auth_manager()
        auth_url = auth_manager.get_authorize_url()
        return render_template('login.html', auth_url=auth_url)
    except Exception as e:
        return f'<h3>Authentication setup error: {str(e)}</h3>'

# New polling-based setup routes for PythonAnywhere compatibility
@app.route('/start_setup_1', methods=['GET', 'POST'])
def start_setup_1():
    """Start setup 1 in background thread and return progress ID"""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Get user path
    user_path = f'.data/{user["id"]}/'
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    
    # Check if setup data exists (JSON)
    setup_json = f'{user_path}setup_data.json'
    if not os.path.exists(setup_json):
        return jsonify({'error': 'Setup data not found'}), 400
    
    # Load setup data
    with open(setup_json, 'r', encoding='utf-8') as f:
        setup_data = json.load(f)
    
    playlist_dict = setup_data['playlist_dict']
    
    progress_id = str(uuid.uuid4())
    progress_data[progress_id] = {
        'status': 'running',
        'messages': [],
        'current_step': 0,
        'total_steps': len(playlist_dict) + 1,  # +1 for liked songs
        'complete': False,
        'error': None
    }
    
    def run_setup_1():
        try:
            # Create a new SetupData instance for this thread
            temp_session = {'SPOTIFY': user['spotify_client']}
            temp_setup = SetupData(temp_session)
            
            ALL_SONGS_DF = pd.DataFrame()
            for i, (name, _id) in enumerate(playlist_dict.items()):
                if progress_data[progress_id]['status'] == 'cancelled':
                    break
                    
                progress_data[progress_id]['current_step'] = i + 1
                progress_data[progress_id]['messages'].append(f'{name} --> {i + 1}/{progress_data[progress_id]["total_steps"]}')
                
                try:
                    # Process playlist
                    df = temp_setup._get_playlist(name, _id)
                    if isinstance(df, str) and df.startswith('data:ERROR='):
                        progress_data[progress_id]['error'] = df
                        progress_data[progress_id]['status'] = 'error'
                        return
                    
                    # Add to main dataframe
                    if not df.empty:
                        ALL_SONGS_DF = pd.concat([ALL_SONGS_DF, df])
                    
                    # Check for max limit
                    if len(ALL_SONGS_DF) > 8888:
                        progress_data[progress_id]['messages'].append('***Total Song Count Reached Max Limit***')
                        break
                        
                except Exception as playlist_error:
                    progress_data[progress_id]['messages'].append(f'Error processing {name}: {str(playlist_error)}')
                    continue
            
            # Collect liked songs if we haven't hit the limit and haven't been cancelled
            if len(ALL_SONGS_DF) <= 8888 and progress_data[progress_id]['status'] != 'cancelled':
                progress_data[progress_id]['current_step'] = len(playlist_dict) + 1
                progress_data[progress_id]['messages'].append('Collecting Liked Songs...')
                
                try:
                    liked_df = temp_setup._get_liked_songs()
                    if isinstance(liked_df, str) and liked_df.startswith('data:ERROR='):
                        progress_data[progress_id]['error'] = liked_df
                        progress_data[progress_id]['status'] = 'error'
                        return
                    
                    # Add liked songs to main dataframe
                    if not liked_df.empty:
                        ALL_SONGS_DF = pd.concat([ALL_SONGS_DF, liked_df])
                        progress_data[progress_id]['messages'].append(f'Liked Songs --> {progress_data[progress_id]["current_step"]}/{progress_data[progress_id]["total_steps"]}')
                    else:
                        progress_data[progress_id]['messages'].append('No liked songs found')
                        
                except Exception as liked_error:
                    progress_data[progress_id]['messages'].append(f'Error processing liked songs: {str(liked_error)}')
                    # Don't return here - continue with what we have
            
            # Save the dataframe (NDJSON only)
            if not ALL_SONGS_DF.empty:
                if 'index' in ALL_SONGS_DF.columns:
                    ALL_SONGS_DF.drop(columns='index', inplace=True)
                ALL_SONGS_DF.to_json(f"{user_path}all_songs.ndjson.gz", orient='records', lines=True, compression='gzip')
                
                # Update status (JSON)
                status = {'SETUP1': True, 'SETUP2': False, 'SETUP3': False}
                with open(f'{user_path}setup_status.json', 'w', encoding='utf-8') as f:
                    json.dump(status, f)
                
                progress_data[progress_id]['status'] = 'complete'
                progress_data[progress_id]['complete'] = True
            else:
                progress_data[progress_id]['error'] = 'No songs were collected from playlists'
                progress_data[progress_id]['status'] = 'error'
            
        except Exception as e:
            progress_data[progress_id]['error'] = str(e)
            progress_data[progress_id]['status'] = 'error'
    
    thread = threading.Thread(target=run_setup_1)
    thread.daemon = True
    thread.start()
    
    return jsonify({'progress_id': progress_id})

@app.route('/start_setup_2', methods=['GET', 'POST'])
def start_setup_2():
    """Start setup 2 in background thread and return progress ID"""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_path = f'.data/{user["id"]}/'
    all_songs_df_file = f'{user_path}all_songs.ndjson.gz'
    
    if not os.path.exists(all_songs_df_file):
        return jsonify({'error': 'Setup 1 not completed'}), 400
    
    all_songs_df = pd.read_json(all_songs_df_file, lines=True, compression='gzip')
    
    progress_id = str(uuid.uuid4())
    progress_data[progress_id] = {
        'status': 'running',
        'messages': [],
        'current_step': 0,
        'total_steps': 11,
        'complete': False,
        'error': None
    }
    
    def run_setup_2():
        try:
            # Create a new SetupData instance for this thread
            temp_session = {'SPOTIFY': user['spotify_client']}
            temp_setup = SetupData(temp_session)
            
            steps = [
                ('Getting Unique Songs...', lambda: temp_setup._get_unique_songs_df()),
                ('Getting Top Artists...', lambda: temp_setup._get_top_artists()),
                ('Getting Top Songs...', lambda: temp_setup._get_top_songs()),
                ('Adding Top Artists Rank...', lambda: temp_setup._add_top_artists_rank()),
                ('Adding Top Songs Rank...', lambda: temp_setup._add_top_songs_rank()),
                ('Getting Artist Genres...', lambda: temp_setup._add_genres()),
                ('Setting Up Home Page...', lambda: None),  # Will be handled separately
                ('Setting Up About Me Page...', lambda: None),  # Will be handled separately
                ('Setting Up Top50 Page...', lambda: None),  # Will be handled separately
                ('Setting Up My Playlists Page...', lambda: None),  # Will be handled separately
                ('Finalizing Data Collection...', lambda: None)  # Will be handled separately
            ]
            
            for i, (message, func) in enumerate(steps):
                if progress_data[progress_id]['status'] == 'cancelled':
                    break
                    
                progress_data[progress_id]['current_step'] = i + 1
                progress_data[progress_id]['messages'].append(f'{message}{i + 1}/{len(steps)}')
                
                # Execute the function
                if func:
                    func()
                
                # Simulate some processing time
                import time
                time.sleep(0.2)
            
            # Page setup no longer writes pickled page objects; views compute on demand and cache via Flask-Caching
            # Update status (JSON)
            status = {'SETUP1': True, 'SETUP2': True, 'SETUP3': False}
            with open(f'{user_path}setup_status.json', 'w', encoding='utf-8') as f:
                json.dump(status, f)
            
            progress_data[progress_id]['status'] = 'complete'
            progress_data[progress_id]['complete'] = True
            
        except Exception as e:
            progress_data[progress_id]['error'] = str(e)
            progress_data[progress_id]['status'] = 'error'
    
    thread = threading.Thread(target=run_setup_2)
    thread.daemon = True
    thread.start()
    
    return jsonify({'progress_id': progress_id})

@app.route('/get_progress/<progress_id>')
def get_progress(progress_id):
    """Get progress for a specific setup operation"""
    if progress_id not in progress_data:
        return jsonify({'error': 'Progress ID not found'}), 404
    
    return jsonify(progress_data[progress_id])

# Login and Setup
@app.route('/')
def index():
	user = get_current_user()
	
	if not user:
		# Not authenticated - show sign in
		auth_manager = create_auth_manager()
		auth_url = auth_manager.get_authorize_url()
		return f'<h2><a href="{auth_url}">Sign in with Spotify</a></h2>'
	
	# User is authenticated - check setup status
	user_path = f'.data/{user["id"]}/'

	# Check if setup data exists; if not, run Setup 1 immediately and persist JSON
	setup_json = f'{user_path}setup_data.json'
	if not os.path.exists(setup_json):
		try:
			temp_session = {'SPOTIFY': user['spotify_client']}
			temp_setup = SetupData(temp_session)
			setup_data = {'user_id': user['id'], 'playlist_dict': temp_setup.PLAYLIST_DICT}
			if not os.path.exists(user_path):
				os.makedirs(user_path)
			with open(setup_json, 'w', encoding='utf-8') as f:
				json.dump(setup_data, f)
		except Exception as e:
			return f'<h3>Error setting up: {str(e)}</h3>'

	# Check collection status (JSON)
	collection_json = f'{user_path}setup_status.json'
	if not os.path.exists(collection_json):
		# If Setup 1 artifact exists (NDJSON), mark and kick off setup 2
		all_songs_path = f'{user_path}all_songs.ndjson.gz'
		if os.path.exists(all_songs_path):
			status = {'SETUP1': True, 'SETUP2': False, 'SETUP3': False}
			with open(collection_json, 'w', encoding='utf-8') as f:
				json.dump(status, f)
		else:
			return render_template('setup.html', setup2="false")

	with open(collection_json, 'r', encoding='utf-8') as f:
		status = json.load(f)

	# If SETUP1 done but SETUP2 not done, redirect to setup 2 page
	if status.get('SETUP1') and not status.get('SETUP2'):
		return render_template('setup.html', setup2="true")

	# If SETUP2 is complete and artifacts exist, go home; otherwise, keep user on setup
	all_songs_path = f'{user_path}all_songs.ndjson.gz'
	unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
	if status.get('SETUP2') and os.path.exists(all_songs_path) and os.path.exists(unique_songs_path):
		return redirect('/home')
	else:
		return render_template('setup.html', setup2="true" if status.get('SETUP1') else "false")

# Spotify OAuth callback
@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return '<h3>Authentication failed</h3>'
    
    try:
        auth_manager = create_auth_manager()
        token_info = auth_manager.get_access_token(code)
        
        if not token_info:
            return '<h3>Failed to get access token</h3>'
        
        access_token = token_info['access_token']

        # Validate token and bind user session
        user_id = get_user_id_from_token(access_token)
        if not user_id:
            return '<h3>Failed to get user info</h3>'

        # Persist Spotipy token cache under per-user .data directory
        user_path = f'.data/{user_id}/'
        if not os.path.exists(user_path):
            os.makedirs(user_path)
        cache_handler = CacheFileHandler(cache_path=f"{user_path}.cache")
        cache_handler.save_token_to_cache(token_info)
        
        # Clean up default root .cache if Spotipy created it
        try:
            default_cache_path = '.cache'
            if os.path.exists(default_cache_path):
                os.remove(default_cache_path)
        except Exception:
            pass

        # Store all token information in session
        session['access_token'] = access_token
        session['user_id'] = user_id
        
        # Store refresh token and expiry for automatic refresh
        if 'refresh_token' in token_info:
            session['refresh_token'] = token_info['refresh_token']
        if 'expires_at' in token_info:
            session['expires_at'] = token_info['expires_at']
        else:
            # If no expires_at provided, calculate it (Spotify tokens typically last 1 hour)
            session['expires_at'] = datetime.datetime.utcnow().timestamp() + 3600

        return redirect('/')
        
    except Exception as e:
        return f'<h3>Authentication error: {str(e)}</h3>'

# Sign out
@app.route('/logout')
def logout():
    """Sign out user and clear session data."""
    session.clear()
    return redirect('/')

# Home Page
def _cache_key_home():
	user = get_current_user()
	uid = user['id'] if user else 'anon'
	if user:
		user_path = f'.data/{uid}/'
		all_ready = os.path.exists(f'{user_path}all_songs.ndjson.gz')
		uniq_ready = os.path.exists(f'{user_path}unique_songs.ndjson.gz')
		ready = '1' if all_ready and uniq_ready else '0'
	else:
		ready = '0'
	return f"home:{uid}:{ready}"

@app.route('/home')
@cache.cached(timeout=300, key_prefix=_cache_key_home)
def home():
	user = get_current_user()
	if not user:
		return redirect('/')
	
	user_path = f'.data/{user["id"]}/'
	
	# Read data from filesystem (NDJSON, gzipped)
	all_songs_path = f'{user_path}all_songs.ndjson.gz'
	unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
	if not os.path.exists(all_songs_path) or not os.path.exists(unique_songs_path):
		return redirect('/')

	all_songs_df = pd.read_json(all_songs_path, lines=True, compression='gzip')
	unique_songs_df = pd.read_json(unique_songs_path, lines=True, compression='gzip')

	# Load setup data to get playlist_dict
	with open(f'{user_path}setup_data.json', 'r', encoding='utf-8') as f:
		setup_data = json.load(f)
	playlist_dict = setup_data['playlist_dict']

	# Build page artifacts in-memory (no pickles)
	from visualization import HomePage
	home_page = HomePage(user_path, all_songs_df, unique_songs_df, playlist_dict)

	on_this_date = home_page.load_on_this_date()
	full_timeline = home_page.load_timeline()
	last_added_playlist = home_page.load_last_added()
	overall_data = home_page.load_totals()
	first_times = home_page.load_first_times()

	return render_template('home.html', collection=False,
					   name=user['display_name'], today=str(
								   datetime.datetime.now().astimezone().date())[5:],
					   on_this_date=on_this_date,
					   full_timeline=full_timeline,
					   last_added_playlist=last_added_playlist,
					   overall_data=overall_data, first_times=first_times)

# Sign out
@app.route('/sign-out')
def sign_out():
    # Capture user before clearing session
    user = get_current_user()
    user_id = user['id'] if user else None

    # Prepare redirect response and clear session cookie
    response = make_response(redirect('/'))
    try:
        response.delete_cookie(app.config.get('SESSION_COOKIE_NAME', 'session'))
    except Exception:
        pass

    # Clear server-side session
    try:
        session.clear()
    except Exception:
        pass

    # Delete per-user data directory and Spotipy cache
    try:
        if user_id:
            user_path = f'.data/{user_id}/'
            if os.path.exists(user_path):
                shutil.rmtree(user_path, ignore_errors=True)
    except Exception:
        pass

    # Clear app cache (includes any user-specific cached pages)
    try:
        cache.clear()
    except Exception:
        pass

    return response

# Currently Playing Page
@app.route('/currently-playing')
def currently_playing():
    user = get_current_user()
    if not user:
        return redirect('/')
    
    track = user['spotify_client'].current_user_playing_track()

    if not track:
        return "<h3>No track currently playing</h3>"

    artist = ', '.join([listy['name'] for listy in track['item']['artists']])
    song = track['item']['name']
    song_id = track['item']['id']
    
    # Load user data (NDJSON presence indicates setup done)
    user_path = f'.data/{user["id"]}/'
    all_songs_path = f'{user_path}all_songs.ndjson.gz'
    unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
    if not os.path.exists(all_songs_path) or not os.path.exists(unique_songs_path):
        return "<h3>Please complete setup first</h3>"
    
    # Load required data (NDJSON, gzipped)
    all_songs_df = pd.read_json(f'{user_path}all_songs.ndjson.gz', lines=True, compression='gzip')
    unique_songs_df = pd.read_json(f'{user_path}unique_songs.ndjson.gz', lines=True, compression='gzip')
    
    # Get playlist info fresh (id -> name)
    id_to_name = {}
    try:
        results = user['spotify_client'].current_user_playlists(limit=50, offset=0)
        id_to_name.update({p['id']: p['name'] for p in results['items']})
        total = results.get('total', len(results['items']))
        offset = 50
        while offset < total:
            more = user['spotify_client'].current_user_playlists(limit=50, offset=offset)
            id_to_name.update({p['id']: p['name'] for p in more['items']})
            offset += 50
    except Exception:
        pass

    # See if User is playing one of their playlists
    playlist = None
    if track.get('context') and track['context'].get('uri'):
        playlist_id = track['context']['uri']
        playlist_id = playlist_id[playlist_id.rfind(':')+1:]
        if playlist_id in id_to_name:
            playlist = id_to_name[playlist_id]
    
    artists_url = artist.replace(', ', '/')

    # Check if song is in library
    song_in_library = True if len(unique_songs_df[(unique_songs_df['name'] == song) & (unique_songs_df['artist'] == artist)]) > 0 else False
    artist_in_library = True if len(unique_songs_df[unique_songs_df['artist'] == artist]) > 0 else False
    
    if not song_in_library:
        if artist_in_library:
            return f'<h3>You are playing the song {song} by <a href="/artists/{artists_url}">{artist}</a> - but it\'s not in any of your created playlists :(</h3>' \
                f'<h3>Start listening to any song from your playlists to see cool stats!</h3>'
        else:
            return f'<h3>You are playing the song {song} by {artist} - but it\'s not in any of your created playlists :(</h3>' \
                f'<h3>Start listening to any song from your playlists to see cool stats!</h3>' 
    else:
        page = CurrentlyPlayingPage(song, artist, playlist, all_songs_df, unique_songs_df)

        top_rank_table = page.graph_top_rank_table()
        song_features_radar = page.graph_song_features_vs_avg()
        song_features_percentiles_bar = page.graph_song_percentiles_vs_avg()
        playlist_date_gantt = page.graph_date_added_to_playlist()
        playlist_timeline = page.graph_count_timeline()
        artist_top_graphs = page.graph_all_artists()
        artist_genres = page.graph_artist_genres()
        most_common_artists_with_same_genres = page.graph_most_common_artists_with_same_genres() if playlist else None
        genres_playlist_percentiles = page.graph_song_genres_vs_avg(playlist=True) if playlist else None

        return render_template('currently_playing.html', song=song, artist=artist, playlist=playlist,
                                top_rank_table=top_rank_table,
                                song_features_radar=song_features_radar, song_features_percentiles_bar=song_features_percentiles_bar,
                                playlist_date_gantt=playlist_date_gantt, playlist_timeline=playlist_timeline,
                                artist_top_graphs=artist_top_graphs,
                                artist_genres=artist_genres, most_common_artists_with_same_genres=most_common_artists_with_same_genres, genres_playlist_percentiles=genres_playlist_percentiles,
                                song_id=song_id, artists_url=artists_url, playlist_id=playlist_id if playlist else None)

# About Me Page
@app.route('/about-me')
@cache.cached(timeout=300)
def about_me():
    user = get_current_user()
    if not user:
        return redirect('/')
    
    user_path = f'.data/{user["id"]}/'
    if not os.path.exists(user_path):
        return redirect('/')

    # Load runtime data from NDJSON
    all_songs_path = f'{user_path}all_songs.ndjson.gz'
    unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
    if not (os.path.exists(all_songs_path) and os.path.exists(unique_songs_path)):
        return redirect('/')
    all_songs_df = pd.read_json(all_songs_path, lines=True, compression='gzip')
    unique_songs_df = pd.read_json(unique_songs_path, lines=True, compression='gzip')

    # Fetch followed artists live (keeps data fresh)
    artists = user['spotify_client'].current_user_followed_artists()['artists']['items']
    from visualization import AboutPage
    page = AboutPage(user_path, all_songs_df, unique_songs_df, artists)

    top_genres_by_followed_artists_bar = page.load_followed_artists()
    top_songs_by_num_playlists = page.load_top_songs()
    top_artists_by_all_playlists = page.load_top_artists()
    top_albums_by_all_playlists = page.load_top_albums()

    return render_template('about_me.html', top_genres_by_followed_artists_bar=top_genres_by_followed_artists_bar,
                           top_songs_by_num_playlists=top_songs_by_num_playlists,
                           top_artists_by_all_playlists=top_artists_by_all_playlists,
                           top_albums_by_all_playlists=top_albums_by_all_playlists)

# Top 50 Page
def _cache_key_top50():
	user = get_current_user()
	uid = user['id'] if user else 'anon'
	if user:
		user_path = f'.data/{uid}/'
		unique_path = f'{user_path}unique_songs.ndjson.gz'
		ready = '1' if os.path.exists(unique_path) else '0'
		mtime = str(os.path.getmtime(unique_path)) if os.path.exists(unique_path) else '0'
	else:
		ready = '0'
		mtime = '0'
	return f"top50:{uid}:{ready}:{mtime}"

@app.route('/top-50')
@cache.cached(timeout=300, key_prefix=_cache_key_top50)
def top_50():
    user = get_current_user()
    if not user:
        return redirect('/')
    
    user_path = f'.data/{user["id"]}/'
    unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
    if not os.path.exists(unique_songs_path):
        return redirect('/')

    unique_songs_df = pd.read_json(unique_songs_path, lines=True, compression='gzip')

    # Load Top 50 inputs (prefer JSON)
    try:
        with open(f'{user_path}top_artists.json', 'r', encoding='utf-8') as f:
            top_artists = json.load(f)
    except Exception:
        top_artists = []
    try:
        with open(f'{user_path}top_artists_pop.json', 'r', encoding='utf-8') as f:
            top_artists_pop = json.load(f)
    except Exception:
        top_artists_pop = []

    from visualization import Top50Page
    page = Top50Page(user_path, unique_songs_df, top_artists, top_artists_pop)
    plots_by_time_range = page.load_dynamic_graph()

    return render_template('top_50.html', plots_by_time_range=plots_by_time_range)

# My Playlists Page
@app.route('/my-playlists')
@cache.cached(timeout=300)
def my_playlists():
    user = get_current_user()
    if not user:
        return redirect('/')
    
    user_path = f'.data/{user["id"]}/'
    all_songs_path = f'{user_path}all_songs.ndjson.gz'
    unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
    if not (os.path.exists(all_songs_path) and os.path.exists(unique_songs_path)):
        return redirect('/')

    all_songs_df = pd.read_json(all_songs_path, lines=True, compression='gzip')
    unique_songs_df = pd.read_json(unique_songs_path, lines=True, compression='gzip')

    # Load Top 50 lists (prefer JSON, fallback to pickle for backward compat)
    try:
        with open(f'{user_path}top_artists.json', 'r', encoding='utf-8') as f:
            top_artists = json.load(f)
    except Exception:
        top_artists = []
    try:
        with open(f'{user_path}top_songs.json', 'r', encoding='utf-8') as f:
            top_songs = json.load(f)
    except Exception:
        top_songs = []

    from visualization import MyPlaylistsPage
    page = MyPlaylistsPage(user_path, all_songs_df, top_artists, top_songs)

    playlists_by_length = page.load_playlists_by_length()
    playlists_by_explicit = page.load_playlists_by_explicit()
    avg_boxplot = page.load_avg_boxplot()
    first_last_added = page.load_first_last_added()
    top_playlists_by_songs = page.load_playlists_by_songs()
    top_playlists_by_artists = page.load_playlists_by_artists()

    return render_template('my_playlists.html', playlists_by_length=playlists_by_length,
                           playlists_by_explicit=playlists_by_explicit,
                           avg_boxplot=avg_boxplot,
                           first_last_added=first_last_added,
                           top_playlists_by_songs=top_playlists_by_songs,
                           top_playlists_by_artists=top_playlists_by_artists)

# Search Page - don't cache otherwise will just keep search bar
@app.route('/search', methods=['GET', 'POST'])
def search():
    user = get_current_user()
    if not user:
        return redirect('/')

    user_path = f'.data/{user["id"]}/'
    setup_json = f'{user_path}setup_data.json'
    collection_json = f'{user_path}setup_status.json'
    if not os.path.exists(setup_json) or not os.path.exists(collection_json):
        return redirect('/')
    
    with open(setup_json, 'r', encoding='utf-8') as f:
        setup_data = json.load(f)
    playlist_dict = setup_data['playlist_dict']  # name -> id
    lowercase_playlists = {name.lower(): pid for name, pid in playlist_dict.items()}

    # Data needed for ID/artist checks
    all_songs_df_path = f'{user_path}all_songs.ndjson.gz'
    all_songs_df = pd.read_json(all_songs_df_path, lines=True, compression='gzip') if os.path.exists(all_songs_df_path) else pd.DataFrame()
    
    # Add "Liked Songs" as a special playlist if it exists in the data
    if not all_songs_df.empty and 'Liked Songs' in all_songs_df['playlist'].values:
        lowercase_playlists['liked songs'] = 'liked_songs'  # Use special ID for liked songs
    unique_artist_names_path = f'{user_path}unique_artist_names.json'
    lowercase_artists = {}
    if os.path.exists(unique_artist_names_path):
        try:
            with open(unique_artist_names_path, 'r', encoding='utf-8') as f:
                unique_artist_names = json.load(f)
            lowercase_artists = {n.lower(): n for n in unique_artist_names}
        except Exception:
            lowercase_artists = {}

    if request.method == 'POST':
        q = request.form.get('query', '')
        query_parts = [i.strip() for i in q.split(',') if i.strip()]

        found_list = []
        found_playlist = False
        found_artist = False
        found_song = False

        all_song_ids = set(all_songs_df['id'].unique()) if not all_songs_df.empty else set()

        for part in query_parts:
            low = part.lower()
            if low in lowercase_playlists:
                found_list.append(lowercase_playlists[low])
                found_playlist = True
            elif low in lowercase_artists:
                found_list.append(lowercase_artists[low])
                found_artist = True
            elif part in all_song_ids:
                found_list.append(part)
                found_song = True
            else:
                return 'Query not found - make sure you spelled correctly!'

            # Disallow mixing different entity types
            if (found_playlist and found_artist) or (found_playlist and found_song) or (found_artist and found_song):
                return 'Query not found - input only playlists OR only artists OR only song IDs'

        if found_playlist:
            url = '/playlists/' + '/'.join(found_list)
        elif found_artist:
            url = '/artists/' + '/'.join(found_list)
        elif found_song:
            url = '/songs/' + '/'.join(found_list)
        else:
            return 'Query not found - make sure you spelled correctly!'

        return redirect(url)
    
    return render_template('search.html')


# Single / Multiple Playlists
@app.route('/playlists/<path:playlist_ids>')
@cache.cached(timeout=300)
def analyze_playlists(playlist_ids):
    user = get_current_user()
    if not user:
        return redirect('/')
    
    user_path = f'.data/{user["id"]}/'
    setup_json = f'{user_path}setup_data.json'
    if not os.path.exists(setup_json):
        return 'Setup not found'
    with open(setup_json, 'r', encoding='utf-8') as f:
        setup_data = json.load(f)
    playlist_dict = setup_data['playlist_dict']  # name -> id
    id_to_name = {pid: name for name, pid in playlist_dict.items()}
    
    # Add special handling for "Liked Songs"
    id_to_name['liked_songs'] = 'Liked Songs'

    all_songs_df = pd.read_json(f'{user_path}all_songs.ndjson.gz', lines=True, compression='gzip')
    unique_songs_df = pd.read_json(f'{user_path}unique_songs.ndjson.gz', lines=True, compression='gzip')

    ids = playlist_ids.split('/')
    if len(ids) == 1:
        pid = ids[0]
        if pid not in id_to_name:
            return 'Playlist ID Not Found'
        playlist_name = id_to_name[pid]
        
        # Get global playlist averages from home page for consistency
        from visualization import HomePage
        home_page = HomePage(user_path, all_songs_df, unique_songs_df, {})
        global_playlist_averages = home_page.get_global_playlist_averages()
        
        page = AnalyzePlaylistPage(playlist_name, all_songs_df, unique_songs_df, global_playlist_averages)

        timeline = page.graph_count_timeline()
        genres = page.graph_playlist_genres()
        top_artists = page.graph_top_artists()
        top_albums = page.graph_top_albums()
        features_boxplot = page.graph_song_features_boxplot()
        similar_playlists_current_in_others = page.graph_similar_playlists_by_current_in_others()
        similar_playlists_others_in_current = page.graph_similar_playlists_by_others_in_current()

        return render_template('analyze_playlist.html', playlist_name=playlist_name,
                            timeline=timeline, genres=genres,
                            top_artists=top_artists, top_albums=top_albums,
                            features_boxplot=features_boxplot,
                            similar_playlists_current_in_others=similar_playlists_current_in_others,
                            similar_playlists_others_in_current=similar_playlists_others_in_current)
    
    elif len(ids) > 1:
        try:
            names = [id_to_name[i] for i in ids]
        except KeyError:
            return 'Playlist IDs not found'
        # Get global playlist averages from home page for consistency
        from visualization import HomePage
        home_page = HomePage(user_path, all_songs_df, unique_songs_df, {})
        global_playlist_averages = home_page.get_global_playlist_averages()
        
        page = AnalyzePlaylistsPage(names, all_songs_df, unique_songs_df, global_playlist_averages)
        boxplots = page.graph_playlists_boxplots()
        boxplot = boxplots[0]
        length = boxplots[1]
        playlist_timelines = page.graph_playlist_timelines()
        genres = page.graph_genres_by_playlists()
        artists = page.graph_artists_by_playlists()

        return render_template('analyze_playlists.html', boxplot=boxplot, length=length, playlists=names,
                               num_playlists=len(names), playlist_ids=ids,
                               playlist_timelines=playlist_timelines, genres=genres, artists=artists)


# Single / Multiple Artists
@app.route('/artists/<path:artist_names>')
@cache.cached(timeout=300)
def analyze_artists(artist_names):
    user = get_current_user()
    if not user:
        return redirect('/')
    
    user_path = f'.data/{user["id"]}/'
    all_songs_df = pd.read_json(f'{user_path}all_songs.ndjson.gz', lines=True, compression='gzip')
    unique_songs_df = pd.read_json(f'{user_path}unique_songs.ndjson.gz', lines=True, compression='gzip')

    # Get global artist averages from home page for consistency
    from visualization import HomePage
    home_page = HomePage(user_path, all_songs_df, unique_songs_df, {})
    global_artist_averages = home_page.get_global_artist_averages()

    artists_list = [i for i in artist_names.split('/')]
    if len(artists_list) == 1:
        artist_name = artists_list[0]
        page = AnalyzeArtistPage(artist_name, all_songs_df, unique_songs_df, global_artist_averages)

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
                               playlists_genres=playlists_genres)

    elif len(artists_list) > 1:
        page = AnalyzeArtistsPage(artists_list, all_songs_df, unique_songs_df, global_artist_averages)
        artist_timelines = page.graph_artist_timelines()
        genres = page.graph_artist_genres()
        top_ranks = page.graph_top_rank_table()
        top_playlists = page.graph_playlists_by_artists()
        audio_features = page.graph_artists_boxplots()

        return render_template('analyze_artists.html', artists_list=artists_list,
                            artist_timelines=artist_timelines,
                            genres=genres, top_ranks=top_ranks, top_playlists=top_playlists,
                            audio_features=audio_features)
    

# Single / Multiple Songs
@app.route('/songs/<path:song_ids>')
@cache.cached(timeout=300)
def analyze_songs(song_ids):
    user = get_current_user()
    if not user:
        return redirect('/')
    
    user_path = f'.data/{user["id"]}/'
    all_songs_df = pd.read_json(f'{user_path}all_songs.ndjson.gz', lines=True, compression='gzip')
    unique_songs_df = pd.read_json(f'{user_path}unique_songs.ndjson.gz', lines=True, compression='gzip')

    ids = song_ids.split('/')
    if len(ids) == 1:
        song_id = ids[0]
        page = SingleSongPage(song_id, all_songs_df, unique_songs_df)
        song = page.get_song()
        artist = page.get_artist()
        artist_url = artist.replace(', ', '/')

        top_rank_table = page.graph_top_rank_table()
        song_features_radar = page.graph_song_features_vs_avg()
        song_features_percentiles_bar = page.graph_song_percentiles_vs_avg()
        playlist_date_gantt = page.graph_date_added_to_playlist()
        artist_top_graphs = page.graph_all_artists()
        artist_genres = page.graph_artist_genres()
        genres_overall_percentiles = page.graph_song_genres_vs_avg()

        return render_template('single_song.html', song=song, artist=artist, artist_url=artist_url, 
                                top_rank_table=top_rank_table,
                                song_features_radar=song_features_radar, song_features_percentiles_bar=song_features_percentiles_bar,
                                playlist_date_gantt=playlist_date_gantt,
                                artist_top_graphs=artist_top_graphs,
                               artist_genres=artist_genres, genres_overall_percentiles=genres_overall_percentiles)

    elif len(ids) > 1:
        return 'Multiple song analysis not implemented'

# -------------------------------Main Method-----------------------------------------------

if __name__ == '__main__':
    # threaded=True is default, meaning that multiple requests can be handled at once
    app.run()
