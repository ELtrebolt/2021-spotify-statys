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
os.environ['SESSION_COOKIE_SECURE'] = 'True'

# Import your Flask app
from app import app as application

if __name__ == "__main__":
    app.run()