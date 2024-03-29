# Spotify Statys

## Table of Contents
1. [Highlights](#highlights)
2. [Tech Stack](#tech-stack)
3. [Code Organization](#code-organization)
4. [Workflow](#workflow)
5. [Updates](#updates)
6. [Features](#features)

## Highlights 
- Built a web application using the Python-Flask framework integrated and hosted with Heroku
- Implemented dynamic querying through the Spotify API based on the user’s created playlists
- Visualized insights with over 30 unique Plotly figures including bar charts, timelines, and box plots
  
- **Demo 1:** Features 5 Pages = Home, About Me, Currently Playing, Top 50, Search (Artists/Playlists)

[https://github.com/ELtrebolt/2021-spotify-statys/assets/44250460/ed9e9a98-9688-417c-bbf8-f735e8062f7f](https://github.com/ELtrebolt/2021-spotify-statys/assets/44250460/ed9e9a98-9688-417c-bbf8-f735e8062f7f)

- **Demo 2:** Currently Playing, Temporary Window Caching, Individual User Session Data
    - See stats of any song that's in any of your playlists!
    - Best when you're jamming to one of your own playlists, but also works if you're adding songs to a queue or listening to someone else's playlist.
    - If you close the window without logging out, you will have some time to reload the page without having to sign in and collect your data again
    - Others cannot access your data even if you forget to sign out

[https://github.com/ELtrebolt/2021-spotify-statys/assets/44250460/00ebe70e-50d4-4933-a8a8-78be41a0c7e2](https://github.com/ELtrebolt/2021-spotify-statys/assets/44250460/00ebe70e-50d4-4933-a8a8-78be41a0c7e2)

## Tech Stack
- **Overview**
    - Backend = Python-Flask
    - Frontend = HTML / CSS
    - Hosted with Heroku
- **Streaming Data Collection**
    - Given Spotify API token, render_template setup.html in app.py
    - In JavaScript for setup.html, create EventSource for route setup_#
        - Use source.onmessage function to add streamed data to setup.html
    - In route setup_1, return Response with Generator Function and text/event-stream
    - In Generator Function, yield "data:" + streamed-data + "\n\n"
- **Dataframe Schema**
    - **ALL_SONGS_DF**
        - id
        - name
        - artist = String separated by commas
        - artist_ids = String separated by commas
        - album
        - explicit
        - popularity
        - playlist = User Playlist Name
        - added_at = String not Date
        - duration
        - genres = LIST of LISTS
    - **UNIQUE_SONGS_DF**
        - groups ALL_SONGS_DF by name and artist
        - playlist becomes playlists = separated by commas
        - added_at = separated by commas
        - songfeature_percentile = percentile across all songs in the dataframe
            - popularity, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration
        - num_playlists = integer
        - artists_short_rank = INT or "N/A" if single, STR separated by commas if multiple
            - med, long
        - songs_short_rank = INT or "N/A"
            - med, long
- **Frontload Data Collection & Querying**
    - Pickle Library = Dump and Load Python Objects in Local File Storage
    - Before letting the user explore the pages - make queries, draw graphs, and pickle those graphs so the pages can be quickly loaded without having to do the same work again
    - Sign_Out will delete all files, and Heroku does so automatically after the User Session

## Code Organization
- **static** --> contains CSS styling
- **templates** --> contains all HTML pages
- **app.py** --> main code logic
- **Procfile** --> required for Heroku
- **requirements.txt** --> list of package versions
- **runtime.txt** --> required for Heroku
- **SetupData** --> Class to build Pandas Dataframes from Spotify API
- **visualization.py** --> Class to build Pages and pickle Plotly graphs

The following folders are dynamically created while running the app:
- **.data** --> save .pkl graphs per Spotify User ID
- **.flask_session** --> save session ID
- **.sp_caches** --> save cache token for the SP object
- **.spotify_caches** --> save cache token for the SPOTIFY object

## Workflow
1. **Activate Python Virtual Environment**** - env\scripts\activate
2. **Test Locally** - flask run
3. **Troubleshooting**
    - Delete .data .flask_session .spotify_caches .cache before running
    - heroku logs --tail
4. **Heroku Setup**
    - pip install gunicorn
    - pip freeze > requirements.txt
    - Procfile / Runtime Files
5. **Uploading to Heroku**
    - Change REDIRECT_URI in app.py and SetupData.py to Heroku URL
    - heroku login
    - heroku git:remote -a spotify-statys
    - git add/commit
    - git push heroku master
    - heroku open
6. **Pulling from Heroku**
    - Install Heroku CLI
    - heroku login
    - heroku git:clone -a spotify-statys
    - env\scripts\activate
    - pip install -r requirements.txt

## Updates
- **Known Bugs**
    1. Top 50 - Last 4 Weeks - % of Top 50 Artists (bottom graph) will not always grab the full 50 artists = usually only the top 29 or 30. 
    2. Today's Date = a day ahead sometimes
- **Future Features**
    1. Collect Liked Songs and Include in All Graphs
    2. Search Autocomplete
    3. Styling
    4. Mobile Responsiveness

## Features
- **Home Page**
    - **On This Date**
        - What songs did you add to which playlists a year ago today?
        - What playlists did you create on this date?
        - What artists did you add to a playlist for the first time on this date?
    - Timeline of Adding Songs to Playlists
    - Most Recent Songs Added to Playlists
    - **Library Totals**
        - Across 59 playlists, I've added 5738 songs
        - My playlists have 3383 unique songs, 2359 unique albums, and 1870 unique artists
- **About Me Page**
    - Top Genres by Followed Artists
    - Top Playlists by Number of Top 50 Songs In Them
    - Top Playlists by Number of Top 10 Artists' Songs
    - Top 10 Most Common Songs / Albums / Artists by Count in All Playlists
- **Currently Playing Page**
    - Song / Artist Rank in Top 50
    - Song Features vs. AVG Song from Playlist, All Playlists, & Artist
    - Percentile of Song Audio Features by Playlist, All Playlists, & Artist
    - Timeline of When SONG Was Added to Playlists
    - Timeline of When Songs Were Added to Playlist
    - Most Common Playlists For Artist
    - Most Common Songs For Artist
    - Artist Genres
    - Percentage of Songs With Same Genres As Artist in Playlist
    - Percentage of Songs With Same Genres As Artist Across All Playlists
- **Top 50 Page - Across 3 Time Periods**
    - Audio Features & Boxplot of Top 50 Songs
    - AVG Audio Features & Boxplot of Top 50 Artists' Songs From Your Playlists
    - Most Common Genres & Top 5 Songs Per Genre Across Top 50 Songs
    - Most Common Genres & Top 5 Artists Per Genre Across Top 50 Artists
- **Search Page**
    - **Analyze Artist**
        - Timeline of When Songs Were Added to Playlists
        - Top 50 Rankings
        - Most Common Playlists For Artist
        - Most Common Songs For Artist
        - Audio Features of Artist's Songs
        - Playlists With Artist's Genres
    - **Analyze Playlist**
        - Timeline of When Songs Were Added to Playlist
        - Most Common Genres For Playlist
        - Most Common Artists For Playlist
        - Most Common Albums For Playlist
        - Audio Features & Boxplot of Songs in Playlist
        - Most Similar Playlists By Songs Shared
    - **Compare Artists**
        - Timeline of When Songs Were Added to Playlists
        - Genres Venn Diagram
    - **Compare Playlists**
        - Audio Features of Songs Which Intersect Playlists
        - Timeline of When Songs Were Added to Playlists
        - Most Shared Genres Between Playlists
