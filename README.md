# Spotify Statys

## Table of Contents
1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Code Organization](#code-organization)
4. [Workflow](#workflow)
5. [Known Bugs](#known-bugs)

## Overview 

**Highlights**
- Built a web application framework using Python-Flask integrated and hosted with Heroku
- Implemented dynamic querying through the Spotify API based on the user’s created playlists
- Visualized insights with over 30 unique Plotly figures including bar charts, time series graphs, and box plots

**Features**
- **Home Page**
    - On This Date
        - What songs did you add to which playlists a year ago today?
        - What playlists did you create on this date?
        - What artists did you add to a playlist for the first time on this date?
    - Timeline of Adding Songs to Playlists
    - Most Recent Songs Added to Playlists
    - Library Totals
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

## Tech Stack
- Backend = Python-Flask
    - Database = Local Pickle Files per User Session
- Frontend = HTML / CSS
- Hosted with Heroku

## Code Organization
- **static** --> contains CSS styling
- **templates** --> contains all HTML pages
- **app.py** --> main code logic
- **Procfile** --> required for Heroku
- **requirements.txt** --> list of package versions
- **runtime.txt** --> required for Heroku
- **SetupData** --> Class to build Pandas Dataframes from Spotify API
- **visualization.py** --> Class to build Pages and pickle Plotly graphs

## Workflow

1. Activate Python Virtual Environment - env\scripts\activate
2. Test Locally - flask run
3. Troubleshooting
    - Delete .data and .flask_session folders if cannot sign out
4. Uploading to Heroku
    - Change REDIRECT_URI in app.py and SetupData.py to Heroku URL


## Known Bugs
1. Top 50 %s may not be even if the API grabs less than the Top 50 songs (for example it has only grabbed 29 songs)
