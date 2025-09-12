# This file contains the WSGI configuration required to serve up your
# web application at http://<your-username>.pythonanywhere.com/
# It works by setting the variable 'application' to a WSGI handler of some
# description.
#
# The below has been auto-generated for your Flask project

import sys
import os

# Add your project directory to the Python path
project_path = '/home/spotifystatys/mysite'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
sys.path.append('/home/spotifystatys/.local/lib/python3.10/site-packages')

# Set environment variables
os.environ['SPOTIPY_CLIENT_ID'] = 'fromSpotify'
os.environ['SPOTIPY_CLIENT_SECRET'] = 'fromSpotify'
os.environ['SPOTIPY_REDIRECT_URI'] = 'https://spotifystatys.pythonanywhere.com/callback'
os.environ['FLASK_SECRET_KEY'] = 'strongRandomString'
# Disable secure cookies for PythonAnywhere compatibility
os.environ['SESSION_COOKIE_SECURE'] = 'False'
os.environ['SESSION_COOKIE_HTTPONLY'] = 'True'
os.environ['SESSION_COOKIE_SAMESITE'] = 'Lax'
os.environ['SESSION_COOKIE_DOMAIN'] = 'None'
os.environ['PERMANENT_SESSION_LIFETIME'] = '3600'  # 1 hour

# Create necessary directories for cache and session storage
# This ensures PythonAnywhere has proper permissions
cache_dir = os.path.join(project_path, '.flask_cache')
session_dir = os.path.join(project_path, '.flask_session')
data_dir = os.path.join(project_path, '.data')

for directory in [cache_dir, session_dir, data_dir]:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, mode=0o755)
        except OSError:
            # If we can't create directories, continue anyway
            # The app will handle this gracefully
            pass

# Import your Flask app
from app import app as application

if __name__ == "__main__":
    app.run()