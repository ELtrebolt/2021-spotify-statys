# visualization.py = contains Page Classes which have some unique and shared functions to make graphs

# Imports --------------------------------------------------------------------
from collections import defaultdict
import datetime
from re import A
import numpy as np
import pandas as pd
from markupsafe import Markup
from urllib.parse import quote
from matplotlib_venn import venn3
from matplotlib_venn import venn2
import base64
from matplotlib import pyplot as plt
from io import BytesIO
from plotly import subplots
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
import os
import tempfile
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name
import matplotlib
matplotlib.use('Agg')
import json
import gzip
from markupsafe import Markup
import traceback
import shutil
import time
import requests
from typing import Dict, Any, Optional, Tuple

# Constants -------------------------------------------------------------------

# PERCENTILE_COLS = ['popularity']
PERCENTILE_COLS = ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'duration', 'tempo', 'loudness']
# FEATURE_COLS = ['popularity']
FEATURE_COLS = ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

LABEL_CUTOFF_LENGTH = 25
MAX_HOVER_ROWS = 10
MAX_HBARS = 10
TIME_RANGE_DICT = {0: ['Last 4 Weeks', 'short_rank'], 1: [
    'Last 6 Months', 'med_rank'], 2: ['All Time', 'long_rank']}
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
          '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

# Timeline color constants for consistency across all timeline functions
MAIN_TIMELINE_COLOR = '#636EFA'      #1f77b4 Blue for main timeline traces
LIKED_TIMELINE_COLOR = '#00CC96'     # Green for liked songs traces
MAIN_AVG_COLOR = 'darkblue'         # Dark grey for main average lines
LIKED_AVG_COLOR = 'darkgreen'        # Dark grey for liked songs average lines
YEARLY_VLINE_COLOR = 'green'          # Color for yearly anniversary vertical lines

TOP_RANK_TABLE_MIN_HEIGHT = 300
MIN_ARTIST_SONG_COUNT_FOR_ON_THIS_DATE = 5

# Helper Functions -----------------------------------------------------------------------------------------

def _shorten_names(listy):
    return [i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in listy[:MAX_HOVER_ROWS]]

def _h_bar(series, title=None, xaxis=None, yaxis=None, percents=False,
           long_names=False, hovertext=None, to_html=True, name=None, color=None, markup=True, fixHeight=False):
    if percents:
        texty = []
        for i in series:
            try:
                if i is not None and isinstance(i, (int, float)):
                    texty.append(str(round(i, 2)) + '%')
                else:
                    texty.append(str(i) if i is not None else 'N/A')
            except (TypeError, ValueError):
                texty.append(str(i) if i is not None else 'N/A')
    else:
        texty = series

    if long_names:
        y_labels = _shorten_names(series.keys())
    else:
        y_labels = series.keys()


    fig = go.Bar(
        x=series,
        y=y_labels,
        orientation='h',
        text=texty,
        textposition='auto',
        hovertext=hovertext,
        name=name,
        marker_color=color
    )

    if to_html:
        fig = go.Figure(fig)

        if title:
            fig.update_layout(title_text=title)
        if xaxis:
            fig.update_layout(xaxis_title=xaxis)
        if yaxis:
            fig.update_layout(yaxis_title=yaxis)

        if fixHeight:
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=min(max((len(series)+1)*50, 300), 500))
        else:
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.update_yaxes(automargin=True)

        if markup:
            return Markup(fig.to_html(full_html=False))
        else:
            return fig
    else:
        return fig


def _boxplot(x_data, y_data, text, title=None, xaxis=None, yaxis=None, to_html=True, name=None, color=None, markup=True):
    if to_html:
        fig = go.Figure()

        i = 0
        for xd, yd in zip(x_data, y_data):

            fig.add_trace(go.Box(
                y=yd,
                name=xd,
                boxpoints='all',
                text=text,
                marker_color=COLORS[i]
            )
            )
            i += 1

        if title:
            fig.update_layout(title_text=title)
        if xaxis:
            fig.update_layout(xaxis_title=xaxis)
        if yaxis:
            fig.update_layout(yaxis_title=yaxis)

        if markup:
            return Markup(fig.to_html(full_html=False))
        else:
            return fig
    else:
        length = len(y_data[0])
        box = go.Box(
            y=[j for i in y_data for j in i],
            x=[k for j in [[i]*length for i in x_data] for k in j],
            name=name,
            boxpoints='all',
            hovertext=text*len(FEATURE_COLS),
            marker_color=color,
            offsetgroup=name
        )
        return box


def fix_genres_column_standalone(df):
    """Standalone function to fix genres column - can be used outside of classes"""
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(parse_genre_string_standalone)
    return df

def parse_genre_string_standalone(genre_value):
    """Standalone function to parse genre from string representation back to list"""
    # Handle pandas Series or arrays properly
    if hasattr(genre_value, '__iter__') and not isinstance(genre_value, (str, list)):
        # If it's a pandas Series or array, take the first value
        if len(genre_value) > 0:
            genre_value = genre_value.iloc[0] if hasattr(genre_value, 'iloc') else genre_value[0]
        else:
            return [["N/A"]]
    
    # Check for None or NaN values
    if genre_value is None:
        return [["N/A"]]
    
    # Handle pandas NaN values
    try:
        if pd.isna(genre_value):
            return [["N/A"]]
    except:
        pass
    
    if isinstance(genre_value, list):
        # Clean the list if it contains strings with quotes/brackets
        cleaned_list = []
        for item in genre_value:
            if isinstance(item, list):
                # Clean nested list
                cleaned_nested = []
                for nested_item in item:
                    if isinstance(nested_item, str):
                        clean_item = nested_item.strip().strip('"\'[]')
                        if clean_item:
                            cleaned_nested.append(clean_item)
                if cleaned_nested:
                    cleaned_list.append(cleaned_nested)
                else:
                    cleaned_list.append(["N/A"])
            elif isinstance(item, str):
                # Clean string item
                clean_item = item.strip().strip('"\'[]')
                if clean_item:
                    cleaned_list.append([clean_item])
                else:
                    cleaned_list.append(["N/A"])
            else:
                cleaned_list.append(["N/A"])
        return cleaned_list if cleaned_list else [["N/A"]]
    
    if isinstance(genre_value, str):
        # Handle JSON-like strings
        if genre_value.startswith('[') and genre_value.endswith(']'):
            try:
                import json
                parsed = json.loads(genre_value)
                if isinstance(parsed, list):
                    # Clean the parsed list
                    cleaned_list = []
                    for item in parsed:
                        if isinstance(item, list):
                            cleaned_nested = []
                            for nested_item in item:
                                if isinstance(nested_item, str):
                                    clean_item = nested_item.strip().strip('"\'[]')
                                    if clean_item:
                                        cleaned_nested.append(clean_item)
                            if cleaned_nested:
                                cleaned_list.append(cleaned_nested)
                            else:
                                cleaned_list.append(["N/A"])
                        elif isinstance(item, str):
                            clean_item = item.strip().strip('"\'[]')
                            if clean_item:
                                cleaned_list.append([clean_item])
                            else:
                                cleaned_list.append(["N/A"])
                    return cleaned_list if cleaned_list else [["N/A"]]
            except:
                pass
        
        # Handle comma-separated strings
        if ',' in genre_value:
            genres = [g.strip().strip('"\'[]') for g in genre_value.split(',') if g.strip()]
            return [genres] if genres else [["N/A"]]
        
        # Single genre
        if genre_value.strip():
            clean_genre = genre_value.strip().strip('"\'[]')
            if clean_genre:
                return [[clean_genre]]
    
    return [["N/A"]]

def _venn_diagram_artist_genres(artists, genres):
    # Safety check: ensure we have valid data
    if not artists or not genres or len(artists) != len(genres):
        print(f"Warning: Invalid data for Venn diagram - artists: {artists}, genres: {genres}")
        return None
        
    if len(genres) == 2:
        left = set(genres[0])
        right = set(genres[1])

        img = BytesIO()

        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 3.5))

        fig = venn2([left, right], tuple(artists))
        bubbles = ['10', '01', '11']
        text = [left-right, right-left, left & right]
        for i, j in zip(bubbles, text):
            try:
                fig.get_label_by_id(i).set_text('\n'.join(j))
            except:
                pass
        # Safety check: ensure we have labels before trying to access them
        if hasattr(fig, 'set_labels') and fig.set_labels:
            for text in fig.set_labels:
                text.set_fontsize(14)

        # Save it to a temporary buffer
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        # Embed the result in the html output
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    elif len(genres) >= 3:
        first = set(genres[0])
        second = set(genres[1])
        third = set(genres[2])

        img = BytesIO()

        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 6))

        fig = venn3([first, second, third],
                    set_labels=tuple(artists))

        bubbles = ['100', '010', '001', '110', '011', '101', '111']
        text = [first-second-third, second-first-third, third-first-second,
                first & second-third, second & third-first, first & third-second,
                first & second & third]

        for i, j in zip(bubbles, text):
            try:
                fig.get_label_by_id(i).set_text('\n'.join(j))
            except:
                pass

        # Save it to a temporary buffer
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        # Embed the result in the html output
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url


def _timeline_trace(df, continuous, playlists, bar, trace=False):
    df = df.groupby(['date_added'], as_index=False)[
        ['name', 'artist']].agg(lambda x: list(x))
    df = df.sort_values(by='date_added')

    total_count = []
    count = 0
    for i in df['name']:
        if continuous:
            count += len(i)
        else:
            count = len(i)
        total_count.append(count)
    df['total_count'] = total_count

    df['songs'] = ['<br>'.join(i) for i in df['name']]

    if playlists:
        title = playlists[0] if len(playlists) == 1 else 'Individual Playlists'
    else:
        title = 'Any Playlist'

    if trace:
        if bar:
            return go.Bar(name=trace, x=df['date_added'], y=df['total_count'], hovertext=df['songs'])
        else:
            return go.Scatter(name=trace, x=df['date_added'], y=df['total_count'], mode='lines', text=df['songs'])
    else:
        if bar:
            return px.bar(df, x='date_added', y='total_count',
                          title='Timeline of When Songs Were Added to ' + title, hover_name='songs')
        else:
            return px.line(df, x='date_added', y='total_count',
                           title='Timeline of When Songs Were Added to ' + title, hover_name='songs')


def _make_single_subplot(labels, figs, xaxis, yaxis, title, rows=1):
    if rows == 2:
        fig = subplots.make_subplots(rows=rows, cols=1, row_heights=[0.15, 0.85])
    else:
        fig = subplots.make_subplots(rows=1, cols=1)

    length = len(figs[0].data)
    for f in figs:
        if rows == 1:
            fig.add_traces(f.data, 1, 1)
        elif rows == 2:
            fig.add_trace(f.data[0], 2, 1)
            fig.add_trace(f.data[1], 1, 1)

    # Create buttons
    buttons = []
    for i, label in enumerate(labels):
        # True, False for first then False, True for second
        visibility = [i == j for j in range(len(labels))]
        # True, True, False, False for first then False, False, True, True for second
        visibility = [k for j in [[i]*length for i in visibility]
                        for k in j]
        button = dict(
            method='update',
            label=label,
            args=[{'visible': visibility}])
        buttons.append(button)

    # Put buttons right under title
    updatemenus = list([
        dict(type='buttons',
            direction='right',
            active=0,
            x=0.05,
            xanchor="left",
            y=1.15,
            yanchor="top",
            buttons=buttons
            )
    ])

    if rows == 1:
        fig.update_layout(updatemenus=updatemenus, showlegend=False, 
                          xaxis_title=xaxis, yaxis_title=yaxis, title=title)
    elif rows == 2:
        fig.update_layout(updatemenus=updatemenus, showlegend=False, title=title)
        fig['layout']['xaxis2']['title'] = xaxis
        fig['layout']['yaxis2']['title'] = yaxis
    
    # Only show 1st Graph on page-load
    Ld = len(fig.data)
    for k in range(length, Ld):
        fig.update_traces(visible=False, selector=k)
    return fig


# Shared Functions -----------------------------------------------------------------------------------------


def shared_graph_count_timeline(ALL_SONGS_DF, playlists=None, song_name=None, artists=None, continuous=False, bar=False, to_html=True):
    song_in_playlist = False
    if playlists and len(playlists) == 1:
        df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] ==
                          playlists[0]][['name', 'artist', 'date_added']]
        song_in_playlist = True if song_name in df['name'].values else False
        if song_name and artists and len(artists) == 1 and song_in_playlist:
            song_df = df[df['name'] == song_name]
            song_df = song_df[song_df['artist'] == artists[0]]
            song_dates = [song_df.iloc[0]['date_added']]
    else:
        df = ALL_SONGS_DF[['name', 'artist', 'date_added']]
        song_in_playlist = True if song_name in df['name'].values else False
        if song_name and artists and len(artists) == 1 and song_in_playlist:
            song_df = df[df['name'] == song_name]
            song_df = song_df[song_df['artist'] == artists[0]]
            song_dates = song_df['date_added'].unique()
        elif artists and len(artists) == 1:
            df = ALL_SONGS_DF[ALL_SONGS_DF['artist'].apply(
                lambda x: artists[0] in x.split(', '))]

    if playlists and len(playlists) > 1:
        fig = go.Figure()
        for p in playlists:
            df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] == p]
            trace = _timeline_trace(
                df, continuous, playlists, bar, trace=p)
            fig.add_trace(trace)
    elif artists and len(artists) > 1:
        fig = go.Figure()
        for a in artists:
            df = ALL_SONGS_DF[ALL_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))]
            trace = _timeline_trace(
                df, continuous, playlists, bar, trace=a)
            fig.add_trace(trace)
    elif playlists and len(playlists) == 1:
        # Single playlist case - pass the playlist name as trace to get go.* objects
        fig = _timeline_trace(df, continuous, playlists, bar, trace=playlists[0])
    else:
        fig = _timeline_trace(df, continuous, playlists, bar)

    if not to_html:
        return fig

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(
        yaxis_title='Total Songs' if continuous else 'Daily Added Songs')

    return Markup(fig.to_html(full_html=False))


def shared_graph_count_timelines(all_songs_df, title, playlists=None, artists=None, song_name=None, to_html=True, global_artist_averages=None):
    today = datetime.datetime.now().astimezone()
    
    # Unpack global artist averages if provided
    global_avg_liked_songs_per_artist = 0
    global_avg_total_songs_per_artist = 0
    if global_artist_averages:
        global_avg_liked_songs_per_artist, global_avg_total_songs_per_artist = global_artist_averages
    
    # Create timeline for main data (excluding liked songs if artists are specified)
    if artists:
        # For artist timelines, exclude liked songs from main trace
        main_df = all_songs_df[all_songs_df['playlist'] != 'Liked Songs']
        line_timeline = shared_graph_count_timeline(
            main_df, continuous=False, playlists=playlists, artists=artists, to_html=False)
        continuous_timeline = shared_graph_count_timeline(
            main_df, continuous=True, playlists=playlists, artists=artists, to_html=False)
        bar_timeline = shared_graph_count_timeline(
            main_df, bar=True, playlists=playlists, artists=artists, to_html=False)
        
        # Create timeline for liked songs only (filtered by artists)
        liked_songs_df = all_songs_df[all_songs_df['playlist'] == 'Liked Songs']
        if len(liked_songs_df.index) > 0:
            # Filter liked songs by the specified artists
            artist_filtered_liked = liked_songs_df[liked_songs_df['artist'].apply(
                lambda x: any(artist in x.split(', ') for artist in artists))]
            if len(artist_filtered_liked.index) > 0:
                line_timeline_liked = shared_graph_count_timeline(
                    artist_filtered_liked, playlists=['Liked Songs'], continuous=False, to_html=False)
                continuous_timeline_liked = shared_graph_count_timeline(
                    artist_filtered_liked, playlists=['Liked Songs'], continuous=True, to_html=False)
                bar_timeline_liked = shared_graph_count_timeline(
                    artist_filtered_liked, playlists=['Liked Songs'], bar=True, to_html=False)
            else:
                line_timeline_liked = None
                continuous_timeline_liked = None
                bar_timeline_liked = None
        else:
            line_timeline_liked = None
            continuous_timeline_liked = None
            bar_timeline_liked = None
    else:
        # Original behavior for non-artist timelines
        line_timeline = shared_graph_count_timeline(
            all_songs_df, continuous=False, playlists=playlists, artists=artists, to_html=False)
        continuous_timeline = shared_graph_count_timeline(
            all_songs_df, continuous=True, playlists=playlists, artists=artists, to_html=False)
        bar_timeline = shared_graph_count_timeline(
            all_songs_df, bar=True, playlists=playlists, artists=artists, to_html=False)
        
        # For playlist timelines, also create liked songs traces if we have playlists
        if playlists:
            # Create timeline for liked songs only (filtered by playlists)
            liked_songs_df = all_songs_df[all_songs_df['playlist'] == 'Liked Songs']
            if len(liked_songs_df.index) > 0:
                # Filter liked songs to only include those that are also in the specified playlists
                playlist_songs = all_songs_df[all_songs_df['playlist'].isin(playlists)]
                playlist_song_ids = set(playlist_songs['id'].tolist()) if 'id' in playlist_songs.columns else set()
                
                if playlist_song_ids:
                    # Get liked songs that are also in these playlists
                    liked_in_playlists = liked_songs_df[liked_songs_df['id'].isin(playlist_song_ids)]
                    
                    if len(liked_in_playlists.index) > 0:
                        line_timeline_liked = shared_graph_count_timeline(
                            liked_in_playlists, playlists=['Liked Songs'], continuous=False, to_html=False)
                        continuous_timeline_liked = shared_graph_count_timeline(
                            liked_in_playlists, playlists=['Liked Songs'], continuous=True, to_html=False)
                        bar_timeline_liked = shared_graph_count_timeline(
                            liked_in_playlists, playlists=['Liked Songs'], bar=True, to_html=False)
                    else:
                        line_timeline_liked = None
                        continuous_timeline_liked = None
                        bar_timeline_liked = None
                else:
                    line_timeline_liked = None
                    continuous_timeline_liked = None
                    bar_timeline_liked = None
            else:
                line_timeline_liked = None
                continuous_timeline_liked = None
                bar_timeline_liked = None
        else:
            line_timeline_liked = None
            continuous_timeline_liked = None
            bar_timeline_liked = None

    labels = ["Line", "Continuous", 'Bar']
    fig = subplots.make_subplots(rows=1, cols=1, vertical_spacing=.05,
                                 subplot_titles=(title))

    # Multi-colors for 1 Time Range = Short, Med, Long
    # Check if we have trace objects (from single playlist or artist timelines) or go.Figure objects
    if hasattr(line_timeline, 'data'):
        # These are go.Figure objects with .data attribute (multiple playlists)
        length = len(line_timeline.data)
        for i in range(length):
            fig.add_trace(line_timeline.data[i], 1, 1)
            fig.add_trace(continuous_timeline.data[i], 1, 1)
            fig.add_trace(bar_timeline.data[i], 1, 1)
    else:
        # These are trace objects directly (single playlist or artist timelines)
        fig.add_trace(line_timeline, 1, 1)
        fig.add_trace(continuous_timeline, 1, 1)
        fig.add_trace(bar_timeline, 1, 1)
    
    # Add traces for liked songs with green color (only for artist timelines)
    if artists and line_timeline_liked:
        # line_timeline_liked is now a go.Scatter object directly
        original_line = line_timeline_liked
        # For line traces, song names are in the 'text' property
        song_names = original_line.text if hasattr(original_line, 'text') and (original_line.text is not None and len(original_line.text) > 0) else []
        line_trace = go.Scatter(
            x=original_line.x,
            y=original_line.y,
            mode=original_line.mode,
            name='Liked Songs',
            line=dict(color='#00CC96'),  # Green color
            hovertext=song_names,  # Set hovertext to song names
            hovertemplate='<b>Liked Songs</b><br>Date: %{x}<br>Songs: %{hovertext}<extra></extra>',
            showlegend=True
        )
        
        # continuous_timeline_liked is now a go.Scatter object directly
        original_continuous = continuous_timeline_liked
        # For continuous traces, song names are also in the 'text' property
        song_names_continuous = original_continuous.text if hasattr(original_continuous, 'text') and (original_continuous.text is not None and len(original_continuous.text) > 0) else []
        continuous_trace = go.Scatter(
            x=original_continuous.x,
            y=original_continuous.y,
            mode=original_continuous.mode,
            name='Liked Songs',
            line=dict(color='#00CC96'),  # Green color
            hovertext=song_names_continuous,  # Set hovertext to song names
            hovertemplate='<b>Liked Songs</b><br>Date: %{x}<br>Songs: %{hovertext}<extra></extra>',
            showlegend=True
        )
        
        # bar_timeline_liked is now a go.Bar object directly
        original_bar = bar_timeline_liked
        # Ensure hovertext property contains song names
        song_names_bar = original_bar.hovertext if hasattr(original_bar, 'hovertext') and (original_bar.hovertext is not None and len(original_bar.hovertext) > 0) else []
        bar_trace = go.Bar(
            x=original_bar.x,
            y=original_bar.y,
            name='Liked Songs',
            marker=dict(color='#00CC96'),  # Green color
            hovertext=song_names_bar,  # This contains the song names
            showlegend=True
        )
        
        fig.add_trace(line_trace, 1, 1)
        fig.add_trace(continuous_trace, 1, 1)
        fig.add_trace(bar_trace, 1, 1)

    # hoverlabel_font_color='white'
    fig.update_layout(showlegend=True, title=title, yaxis_title='# Songs', xaxis_title='Date', barmode='stack')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Add vertical lines for today's anniversary dates
    first_date = all_songs_df['date_added'].sort_values(
        ascending=True).iloc[0]
    first_year = int(str(first_date)[:4])
    current_year = int(str(today.date())[:4])
    
    # Get today's month and day
    today_month = today.month
    today_day = today.day
    
    # Add vertical line for each year from first year to current year
    for year in range(first_year, current_year + 1):
        # Create the anniversary date for this year
        try:
            anniversary_date = datetime.datetime(year, today_month, today_day)
            
            fig.add_vline(
                x=anniversary_date, 
                line_width=3,
                line_dash="dash", 
                line_color="green"
            )
        except ValueError:
            # Skip invalid dates (e.g., February 29 in non-leap years)
            continue
    
    # Add song-specific vlines if requested (these come from shared_graph_count_timeline calls)
    # We need to track these separately for the visibility algorithm
    song_vline_count = 0
    if song_name and artists and len(artists) == 1:
        # Check if song exists in the data
        song_df = all_songs_df[all_songs_df['name'] == song_name]
        if len(song_df) > 0:
            song_df = song_df[song_df['artist'].apply(lambda x: artists[0] in x.split(', '))]
            if len(song_df) > 0:
                song_dates = song_df['date_added'].unique()
                for song_date in song_dates:
                    fig.add_vline(x=song_date, line_width=3,
                                  line_dash="dash", line_color="green")
                    song_vline_count += 1

    # Add horizontal lines for averages in the Continuous section
    # These lines will only be visible when the Continuous button is selected
    liked_songs_df = all_songs_df[all_songs_df['playlist'] == 'Liked Songs']
    
    # Initialize average variables to avoid UnboundLocalError
    avg_liked_songs = 0
    avg_total_songs = 0
    
    if len(liked_songs_df.index) > 0:
        if artists:
            # For artist timelines, calculate averages based on the specific artist(s) being analyzed
            # This gives more meaningful averages for single artist analysis
            artist_liked_counts = []
            for artist in artists:
                artist_liked = liked_songs_df[liked_songs_df['artist'].apply(
                    lambda x: artist in x.split(', '))]
                artist_liked_counts.append(len(artist_liked.index))
            avg_liked_songs = sum(artist_liked_counts) / len(artist_liked_counts) if artist_liked_counts else 0
            
            artist_total_counts = []
            for artist in artists:
                artist_songs = all_songs_df[all_songs_df['playlist'] != 'Liked Songs']
                artist_songs = artist_songs[artist_songs['artist'].apply(
                    lambda x: artist in x.split(', '))]
                artist_total_counts.append(len(artist_songs.index))
            avg_total_songs = sum(artist_total_counts) / len(artist_total_counts) if artist_total_counts else 0
        else:
            # For playlist timelines, calculate average liked songs per playlist
            playlist_liked_counts = []
            for playlist in playlists:
                playlist_songs = all_songs_df[all_songs_df['playlist'] == playlist]
                playlist_song_ids = set(playlist_songs['id'].tolist()) if 'id' in playlist_songs.columns else set()
                if playlist_song_ids:
                    playlist_liked = liked_songs_df[liked_songs_df['id'].isin(playlist_song_ids)]
                    playlist_liked_counts.append(len(playlist_liked.index))
            avg_liked_songs = sum(playlist_liked_counts) / len(playlist_liked_counts) if playlist_liked_counts else 0

            # Calculate average number of added-to-a-playlist songs per playlist
            playlist_total_counts = []
            for playlist in playlists:
                playlist_songs = all_songs_df[all_songs_df['playlist'] == playlist]
                playlist_total_counts.append(len(playlist_songs.index))
            avg_total_songs = sum(playlist_total_counts) / len(playlist_total_counts) if playlist_total_counts else 0

        # Add horizontal lines for the averages (initially hidden, only visible in Continuous view)
        if avg_liked_songs > 0:
            fig.add_hline(y=avg_liked_songs, line_width=2, line_dash="dash", 
                         line_color="darkgreen", visible=False)  # Initially hidden
            # Add annotation separately for better visibility control
            fig.add_annotation(
                x=0.02, y=avg_liked_songs, xref='paper', yref='y',
                text=f"Avg Liked Songs: {avg_liked_songs:.1f}",
                showarrow=False, font=dict(color="darkgreen", size=12),
                visible=False  # Initially hidden
            )
        
        if avg_total_songs > 0:
            fig.add_hline(y=avg_total_songs, line_width=2, line_dash="dash", 
                         line_color="darkblue", visible=False)  # Initially hidden
            # Add annotation separately for better visibility control
            fig.add_annotation(
                x=0.02, y=avg_total_songs, xref='paper', yref='y',
                text=f"Avg Total Songs: {avg_total_songs:.1f}",
                showarrow=False, font=dict(color="darkblue", size=12),
                visible=False  # Initially hidden
            )
        
        # Debug: Print the calculated averages to help troubleshoot
        print(f"DEBUG: avg_liked_songs = {avg_liked_songs}, avg_total_songs = {avg_total_songs}")

    # Create buttons for drop down menu AFTER all traces and horizontal lines are added
    buttons = []
    total_traces = len(fig.data)
    
    # Calculate how many horizontal lines and annotations we have
    num_horizontal_elements = 0
    if avg_liked_songs > 0:
        num_horizontal_elements += 2  # 1 line + 1 annotation
    if avg_total_songs > 0:
        num_horizontal_elements += 2  # 1 line + 1 annotation
    
    # Add song vlines to the count of non-trace elements
    num_horizontal_elements += song_vline_count
    
    # Debug: Print the calculated horizontal elements count
    print(f"DEBUG: num_horizontal_elements = {num_horizontal_elements}")
    
    for i, label in enumerate(labels):
        visibility = []
        
        # For timelines with liked songs, we have 6 traces (3 main + 3 liked songs)
        # For other timelines, we have 3 traces (3 main)
        if line_timeline_liked:
            # Timeline with liked songs: 6 traces total
            # Main traces: [0, 1, 2] = [line, continuous, bar]
            # Liked traces: [3, 4, 5] = [line, continuous, bar]
            for j in range(6):
                if j == i or j == i + 3:  # Show main trace (i) and liked songs trace (i+3)
                    visibility.append(True)
                else:
                    visibility.append(False)
        else:
            # Regular timeline: 3 traces total
            # Traces: [0, 1, 2] = [line, continuous, bar]
            for j in range(min(3, total_traces - num_horizontal_elements)):
                if j == i:  # Show trace corresponding to the selected button
                    visibility.append(True)
                else:
                    visibility.append(False)
        
        # Add visibility for song vlines: always visible
        if song_vline_count > 0:
            song_vline_visibility = [True] * song_vline_count
            visibility.extend(song_vline_visibility)
        
        # Add visibility for horizontal lines and annotations: only visible when Continuous (i=1)
        # Horizontal lines and annotations come after traces and song vlines
        remaining_horizontal_elements = num_horizontal_elements - song_vline_count
        if remaining_horizontal_elements > 0:
            horizontal_elements_visibility = [i == 1] * remaining_horizontal_elements
            visibility.extend(horizontal_elements_visibility)
            # Debug: Print button visibility for horizontal elements
            print(f"DEBUG: Button {i} ({label}): horizontal_elements_visibility = {horizontal_elements_visibility}")
        else:
            print(f"DEBUG: Button {i} ({label}): No horizontal elements to show")
        
        button = dict(
            method='update',
            label=label,
            args=[{'visible': visibility}])
        buttons.append(button)

    updatemenus = list([
        dict(type='buttons',
             direction='right',
             active=0,
             y=1.3,
             x=.6,
             buttons=buttons
             )
    ])

    # Update layout with updatemenus
    fig.update_layout(updatemenus=updatemenus)

    # Set initial visibility: show line traces on page load
    Ld = len(fig.data)
    num_traces = Ld - num_horizontal_elements
    
    for k in range(Ld):
        if k < num_traces:  # This is a trace
            # For timelines with liked songs: show indices 0 and 3 (main line and liked songs line)
            # For regular timelines: show index 0 (main line)
            if line_timeline_liked:
                # Timeline with liked songs: 6 traces [0,1,2,3,4,5]
                if k == 0 or k == 3:  # Show main line (0) and liked songs line (3)
                    fig.update_traces(visible=True, selector=k)
                else:  # Hide all other traces
                    fig.update_traces(visible=False, selector=k)
            else:
                # Regular timeline: 3 traces [0,1,2]
                if k == 0:  # Show only main line trace
                    fig.update_traces(visible=True, selector=k)
                else:  # Hide all other traces
                    fig.update_traces(visible=False, selector=k)
        else:  # This is a horizontal line, annotation, or song vline
            if k < num_traces + song_vline_count:  # This is a song vline - keep visible
                fig.update_traces(visible=True, selector=k)
            else:  # This is a horizontal line or annotation - hide initially
                fig.update_traces(visible=False, selector=k)

    if to_html:
        return Markup(fig.to_html(full_html=False))
    else:
        return fig


def shared_graph_top_playlists_by_artist(ALL_SONGS_DF, artist_name):
    mask = ALL_SONGS_DF['artist'].apply(lambda x: artist_name in x.split(', '))
    df = ALL_SONGS_DF[mask]

    series = df['playlist'].value_counts(ascending=False).head(10).sort_values(ascending=True)
    df = df.groupby(['playlist'], as_index=False)[['name']].agg(lambda x: '<br>'.join(x.head(MAX_HOVER_ROWS)))

    return _h_bar(series, title='Most Common Playlists For Artist: ' + artist_name,
                  xaxis='Number of Artist Songs in the Playlist', 
                  hovertext=df.sort_values(by='playlist', key=lambda x: x.map(series))['name'], fixHeight=True
                  )


def shared_graph_top_songs_by_artist(UNIQUE_SONGS_DF, artist_name):
    mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: artist_name in x.split(', '))
    df = UNIQUE_SONGS_DF[mask]

    # Sort by Num Playlists
    d = defaultdict(list)
    for i, j in zip(df['name'], df['num_playlists']):
        d[j].append(i)
    d = {k: ', '.join(v) for k, v in sorted(
        d.items(), key=lambda x: x[0], reverse=True)}

    # Limit the number of rows to prevent super long tables
    max_rows = 20  # Show maximum 20 rows to keep table manageable
    if len(d) > max_rows:
        # Keep only top rows and add note about truncation
        d_truncated = dict(list(d.items())[:max_rows])
        d_truncated[f"... and {len(d) - max_rows} more"] = [f"({len(d) - max_rows} additional songs with fewer playlists)"]
        d = d_truncated

    # Calculate responsive height - cap at reasonable maximum
    base_height = 200  # Base height for header and minimum content
    row_height = 35    # Height per row
    max_height = 600   # Maximum height to prevent super long tables
    calculated_height = base_height + (len(d) * row_height)
    table_height = min(calculated_height, max_height)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['# Playlists Song Is In', 'Artist Song(s)'],
            line_color='darkslategray',
            fill_color='royalblue',
            font=dict(color='white', size=16),
            height=40
        ),
        cells=dict(
            values=[list(d.keys()), list(d.values())],
            line_color='darkslategray',
            align='center',
            font_size=[14, 12],
            height=row_height
        ),
        columnwidth=[0.3, 0.7]  # 30% for playlist count, 70% for song names
    )])

    # Update layout with responsive sizing
    fig.update_layout(
        autosize=True, 
        title_text='Most Common Songs For Artist: ' + artist_name,
        xaxis_title="Number of Playlists The Artist's Song Is In",
        height=table_height,
        # Make table responsive
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return Markup(fig.to_html(full_html=False))


def shared_graph_top_rank_table(df):
    '''Top Rank Table for CurrentlyPlaying and SingleSong'''
    # Guard when rank columns are not yet computed (e.g., Setup 2 not completed)
    required_columns = {
        'artists_short_rank', 'songs_short_rank',
        'artists_med_rank', 'songs_med_rank',
        'artists_long_rank', 'songs_long_rank'
    }
    if df is None or len(df.index) == 0 or not required_columns.issubset(set(df.columns)):
        return None
    # Extract single-row scalar values
    try:
        artist_short = str(df['artists_short_rank'].iloc[0])
        song_short = str(df['songs_short_rank'].iloc[0])
        artist_med = str(df['artists_med_rank'].iloc[0])
        song_med = str(df['songs_med_rank'].iloc[0])
        artist_long = str(df['artists_long_rank'].iloc[0])
        song_long = str(df['songs_long_rank'].iloc[0])
    except Exception:
        return None

    # Two rows: Artists vs Song
    row_labels = ['Artists', 'Song']
    col_short = [artist_short, song_short]
    col_med = [artist_med, song_med]
    col_long = [artist_long, song_long]

    values = [row_labels, col_short, col_med, col_long]

    fig = go.Figure(data=[go.Table(
        # columnorder = [1, 2, 3, 4],
        # columnwidth = [80,400],
        header=dict(
            values=[['<b>Top 50 Rank</b>'],
                    ['<b>Last 4 Weeks</b>'],
                    ['<b>Last 6 Months</b>'],
                    ['<b>All Time</b>']],
            line_color='darkslategray',
            fill_color='royalblue',
            # align=['left','center'],
            font=dict(color='white', size=18),
            height=40
        ),
        cells=dict(
            values=values,
            line_color='darkslategray',
            # fill=dict(color=['paleturquoise', 'white']),
            align='center',
            font_size=18,
            height=40)
    )
    ])
    fig.update_layout(title_text=df['name'].iloc[0] + ' by [' + df['artist'].iloc[0] + '] Ranks in Top 50', 
                      height=TOP_RANK_TABLE_MIN_HEIGHT)

    return Markup(fig.to_html(full_html=False))


# Optimized Base Class for Progressive Loading and DataFrame Operations
class OptimizedPageBase:
    """
    Base class that provides optimized DataFrame operations and progressive loading capabilities.
    This class maintains backward compatibility while adding performance optimizations.
    """
    
    def __init__(self, path, all_songs_df, unique_songs_df, **kwargs):
        self._path = path
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)
        
        # Pre-compute common aggregations for faster access
        self._precompute_common_aggregations()
        
        # Progressive loading state
        self._loaded_components = set()
        self._component_cache = {}
    
    def _precompute_common_aggregations(self):
        """Pre-compute common aggregations to avoid repeated calculations"""
        # Optimize playlist operations
        if 'playlist' in self._all_songs_df.columns:
            # Create playlist counts lookup
            self._playlist_counts = self._all_songs_df['playlist'].value_counts()
            
            # Create playlist duration lookup
            if 'duration' in self._all_songs_df.columns:
                self._playlist_durations = self._all_songs_df.groupby('playlist')['duration'].sum()
            
            # Create playlist explicit content lookup
            if 'explicit' in self._all_songs_df.columns:
                self._playlist_explicit_counts = self._all_songs_df.groupby('playlist')['explicit'].sum()
                self._playlist_explicit_percentages = self._all_songs_df.groupby('playlist')['explicit'].mean()
        
        # Optimize artist operations
        if 'artist' in self._all_songs_df.columns:
            # Create artist lookup for faster searching
            self._artist_to_songs = {}
            for idx, row in self._all_songs_df.iterrows():
                artists = [a.strip() for a in str(row['artist']).split(', ')]
                for artist in artists:
                    if artist not in self._artist_to_songs:
                        self._artist_to_songs[artist] = []
                    self._artist_to_songs[artist].append(idx)
        
        # Optimize date operations
        if 'date_added' in self._all_songs_df.columns:
            # Convert dates once and store
            self._all_songs_df['date_added_parsed'] = pd.to_datetime(self._all_songs_df['date_added'], format='%Y-%m-%d', errors='coerce')
            
                    # Only process unique_songs_df if it has data and the required column
        if len(self._unique_songs_df) > 0 and 'date_added' in self._unique_songs_df.columns:
            self._unique_songs_df['date_added_parsed'] = pd.to_datetime(self._unique_songs_df['date_added'], format='%Y-%m-%d', errors='coerce')
        
        # Fix genres column - convert string representations back to lists
        self._all_songs_df = self._fix_genres_column(self._all_songs_df)
        self._unique_songs_df = self._fix_genres_column(self._unique_songs_df)
    
    def load_component_progressively(self, component_name, loader_func, *args, **kwargs):
        """
        Progressive loading: Load components on-demand and cache results.
        This allows pages to load faster by only computing what's needed.
        """
        if component_name not in self._loaded_components:
            # Load and cache the component
            result = loader_func(*args, **kwargs)
            self._component_cache[component_name] = result
            self._loaded_components.add(component_name)
            return result
        else:
            # Return cached result
            return self._component_cache[component_name]
    
    def get_optimized_playlist_data(self, playlist_name):
        """Optimized playlist data retrieval using pre-computed lookups"""
        if hasattr(self, '_playlist_counts') and playlist_name in self._playlist_counts:
            return {
                'count': self._playlist_counts[playlist_name],
                'duration': self._playlist_durations.get(playlist_name, 0) if hasattr(self, '_playlist_durations') else 0,
                'explicit_count': self._playlist_explicit_counts.get(playlist_name, 0) if hasattr(self, '_playlist_explicit_counts') else 0,
                'explicit_percentage': self._playlist_explicit_percentages.get(playlist_name, 0) if hasattr(self, '_playlist_explicit_percentages') else 0
            }
        return None
    
    def get_optimized_artist_data(self, artist_name):
        """Optimized artist data retrieval using pre-computed lookups"""
        if hasattr(self, '_artist_to_songs') and artist_name in self._artist_to_songs:
            song_indices = self._artist_to_songs[artist_name]
            return self._all_songs_df.iloc[song_indices]
        return pd.DataFrame()
    
    def vectorized_date_filter(self, date_column, target_date):
        """Vectorized date filtering for better performance"""
        if hasattr(self, '_all_songs_df') and f'{date_column}_parsed' in self._all_songs_df.columns:
            # Use pre-parsed dates for faster filtering
            parsed_column = f'{date_column}_parsed'
            try:
                # Handle date format like "08-27" (MM-DD)
                if '-' in target_date and len(target_date.split('-')) == 2:
                    month, day = target_date.split('-')
                    # Create a proper date string for parsing
                    target_date_str = f"2023-{month}-{day}"  # Use arbitrary year
                    target_date_parsed = pd.to_datetime(target_date_str)
                    return self._all_songs_df[self._all_songs_df[parsed_column].dt.strftime('%m-%d') == target_date_parsed.strftime('%m-%d')]
                else:
                    # Fallback to string matching for other formats
                    return self._all_songs_df[self._all_songs_df[date_column].str.contains(target_date, na=False)]
            except Exception:
                # Fallback to string matching if date parsing fails
                return self._all_songs_df[self._all_songs_df[date_column].str.contains(target_date, na=False)]
        else:
            # Fallback to string matching
            return self._all_songs_df[self._all_songs_df[date_column].str.contains(target_date, na=False)]
    
    def _fix_genres_column(self, df):
        """Convert genre strings back to lists after loading from NDJSON"""
        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(self._parse_genre_string)
        return df

    def _parse_genre_string(self, genre_value):
        """Parse genre from string representation back to list"""
        # Handle pandas Series or arrays properly
        if hasattr(genre_value, '__iter__') and not isinstance(genre_value, (str, list)):
            # If it's a pandas Series or array, take the first value
            if len(genre_value) > 0:
                genre_value = genre_value.iloc[0] if hasattr(genre_value, 'iloc') else genre_value[0]
            else:
                return [["N/A"]]
        
        # Check for None or NaN values
        if genre_value is None:
            return [["N/A"]]
        
        # Handle pandas NaN values
        try:
            if pd.isna(genre_value):
                return [["N/A"]]
        except:
            pass
        
        if isinstance(genre_value, list):
            # Check if this is already a proper list of genres (not nested)
            # If all items are strings, return as-is (already properly formatted)
            if all(isinstance(item, str) for item in genre_value):
                # Clean string items
                cleaned_genres = []
                for item in genre_value:
                    clean_item = item.strip().strip('"\'[]')
                    if clean_item and clean_item != 'N/A':
                        cleaned_genres.append(clean_item)
                return cleaned_genres if cleaned_genres else ['N/A']
            
            # Handle nested list structure (legacy format)
            cleaned_list = []
            for item in genre_value:
                if isinstance(item, list):
                    # Clean nested list
                    cleaned_nested = []
                    for nested_item in item:
                        if isinstance(nested_item, str):
                            clean_item = nested_item.strip().strip('"\'[]')
                            if clean_item:
                                cleaned_nested.append(clean_item)
                    if cleaned_nested:
                        cleaned_list.append(cleaned_nested)
                    else:
                        cleaned_list.append(["N/A"])
                elif isinstance(item, str):
                    # Clean string item
                    clean_item = item.strip().strip('"\'[]')
                    if clean_item:
                        cleaned_list.append([clean_item])
                    else:
                        cleaned_list.append(["N/A"])
                else:
                    cleaned_list.append(["N/A"])
            return cleaned_list if cleaned_list else [["N/A"]]
        
        if isinstance(genre_value, str):
            # Handle JSON-like strings
            if genre_value.startswith('[') and genre_value.endswith(']'):
                try:
                    import json
                    parsed = json.loads(genre_value)
                    if isinstance(parsed, list):
                        # Clean the parsed list
                        cleaned_list = []
                        for item in parsed:
                            if isinstance(item, list):
                                cleaned_nested = []
                                for nested_item in item:
                                    if isinstance(nested_item, str):
                                        clean_item = nested_item.strip().strip('"\'[]')
                                        if clean_item:
                                            cleaned_nested.append(clean_item)
                                if cleaned_nested:
                                    cleaned_list.append(cleaned_nested)
                                else:
                                    cleaned_list.append(["N/A"])
                            elif isinstance(item, str):
                                clean_item = item.strip().strip('"\'[]')
                                if clean_item:
                                    cleaned_list.append([clean_item])
                                else:
                                    cleaned_list.append(["N/A"])
                        return cleaned_list if cleaned_list else [["N/A"]]
                except:
                    pass
            
            # Handle comma-separated strings
            if ',' in genre_value:
                genres = [g.strip().strip('"\'[]') for g in genre_value.split(',') if g.strip()]
                return [genres] if genres else [["N/A"]]
            
            # Single genre
            if genre_value.strip():
                clean_genre = genre_value.strip().strip('"\'[]')
                if clean_genre:
                    return [[clean_genre]]
        
        return [["N/A"]]
    
    def _fix_playlist_column(self, df):
        """Convert playlist strings back to lists after loading from NDJSON"""
        if 'playlist' in df.columns:
            df['playlist'] = df['playlist'].apply(self._parse_playlist_string)
        return df
    
    def _parse_playlist_string(self, playlist_value):
        """Parse playlist from string representation back to list"""
        # Handle pandas Series or arrays properly
        if hasattr(playlist_value, '__iter__') and not isinstance(playlist_value, (str, list)):
            # If it's a pandas Series or array, take the first value
            if len(playlist_value) > 0:
                playlist_value = playlist_value.iloc[0] if hasattr(playlist_value, 'iloc') else playlist_value[0]
            else:
                return []
        
        # Check for None or NaN values
        if playlist_value is None or (hasattr(playlist_value, '__len__') and len(playlist_value) == 0):
            return []
        
        # If it's already a list, return it
        if isinstance(playlist_value, list):
            return playlist_value
        
        # If it's a string, try to parse it
        if isinstance(playlist_value, str):
            import ast
            try:
                return ast.literal_eval(playlist_value)
            except:
                # If parsing fails, treat as a single playlist name
                return [playlist_value]
        
        # For any other type, try to convert to string and parse
        try:
            import ast
            return ast.literal_eval(str(playlist_value))
        except:
            return [str(playlist_value)]

    def optimized_groupby_operation(self, df, group_column, agg_columns, agg_functions):
        """
        Optimized groupby operations with vectorized aggregations.
        This replaces slow apply() operations with faster vectorized alternatives.
        """
        # Create aggregation dictionary
        agg_dict = {}
        for col, func in zip(agg_columns, agg_functions):
            if col in df.columns:
                agg_dict[col] = func
        
        if not agg_dict:
            return pd.DataFrame()
        
        # Perform optimized groupby
        result = df.groupby(group_column).agg(agg_dict).reset_index()
        return result


# Dynamic Track Fetching Classes ----------------------------------------------------------------------------------------------

class DynamicTrackFetcher:
    """
    Fetches missing track data from Spotify and ReccoBeats APIs.
    Implements retry logic, error handling, and data validation.
    """
    
    def __init__(self, spotify_client, user_path: str):
        self.spotify_client = spotify_client
        self.user_path = user_path
        
        # API configuration
        self.spotify_base_url = "https://api.spotify.com/v1"
        self.reccobeats_base_url = "https://api.reccobeats.com/v1"
        
        # Rate limiting configuration
        self.spotify_rate_limit = 100  # requests per second
        self.reccobeats_rate_limit = 50  # requests per minute
        self.max_retries = 3
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Spotify-Stats-App/1.0'
        })
    
    def fetch_track_data(self, spotify_track_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch complete track data from both Spotify and ReccoBeats APIs.
        
        Args:
            spotify_track_id: Spotify track ID
            
        Returns:
            Tuple of (spotify_data, reccobeats_data)
        """
        spotify_data = self._fetch_spotify_track(spotify_track_id)
        reccobeats_data = self._fetch_reccobeats_features(spotify_track_id)
        
        return spotify_data, reccobeats_data
    
    def _fetch_spotify_track(self, track_id: str) -> Dict[str, Any]:
        """Fetch track metadata from Spotify API with retry logic."""
        endpoint = f"{self.spotify_base_url}/tracks/{track_id}"
        
        for attempt in range(self.max_retries):
            try:
                # Use Spotify client if available, otherwise make direct HTTP request
                if hasattr(self.spotify_client, 'track'):
                    response = self.spotify_client.track(track_id)
                    return response
                else:
                    # Fallback to direct HTTP request
                    response = self.session.get(endpoint, timeout=10)
                    response.raise_for_status()
                    return response.json()
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to fetch Spotify track data after {self.max_retries} attempts: {e}")
                    return {}
    
    def _fetch_reccobeats_features(self, spotify_track_id: str) -> Dict[str, Any]:
        """Fetch audio features from ReccoBeats API with retry logic."""
        # The ReccoBeats API works differently - we need to:
        # 1. First get all available audio features
        # 2. Then get track details to map ReccoBeats IDs to Spotify IDs
        # 3. Find the matching track and extract its features
        
        try:
            # Step 1: Get audio features for the specific Spotify track ID
            features_endpoint = f"{self.reccobeats_base_url}/audio-features"
            features_params = {"ids": spotify_track_id}
            
            response = self.session.get(features_endpoint, params=features_params, timeout=15)
            response.raise_for_status()
            features_data = response.json()
            
            if not features_data or 'content' not in features_data:
                return {}
            
            content_data = features_data['content']
            if not content_data:
                return {}
            
            # Step 2: Get track details to map ReccoBeats IDs to Spotify IDs
            track_endpoint = f"{self.reccobeats_base_url}/track"
            
            # Collect all ReccoBeats IDs from the features data
            reccobeats_ids = []
            audio_features_map = {}
            
            for track_data in content_data:
                if 'id' in track_data:
                    reccobeats_id = track_data['id']
                    reccobeats_ids.append(reccobeats_id)
                    
                    # Store audio features for this ReccoBeats ID
                    audio_features_map[reccobeats_id] = {}
                    for col in ['danceability', 'energy', 'instrumentalness', 'key', 'liveness', 
                               'loudness', 'mode', 'speechiness', 'tempo', 'valence']:
                        if col in track_data:
                            audio_features_map[reccobeats_id][col] = track_data[col]
            
            if not reccobeats_ids:
                return {}
            
            # Get track details for all ReccoBeats IDs
            track_ids_param = ",".join(reccobeats_ids)
            track_params = {"ids": track_ids_param}
            
            response = self.session.get(track_endpoint, params=track_params, timeout=15)
            response.raise_for_status()
            track_data = response.json()
            
            if not track_data or 'content' not in track_data:
                return {}
            
            track_content = track_data['content']
            
            # Step 3: Find the track that matches our Spotify ID
            for track in track_content:
                if 'href' in track and 'spotify.com/track/' in track['href']:
                    # Extract Spotify ID from href
                    spotify_id = track['href'].split('/track/')[-1].split('?')[0]
                    if spotify_id == spotify_track_id:
                        reccobeats_id = track['id']
                        
                        # Return the audio features for this track
                        if reccobeats_id in audio_features_map:
                            features = audio_features_map[reccobeats_id].copy()
                            features['reccobeats_id'] = reccobeats_id
                            return features
            
            return {}
            
        except Exception as e:
            print(f"Error fetching ReccoBeats features: {e}")
            return {}
    
    def validate_track_data(self, spotify_data: Dict[str, Any], reccobeats_data: Dict[str, Any]) -> bool:
        """Validate that fetched data meets requirements for database insertion."""
        if not spotify_data:
            return False
        
        # Check required Spotify fields
        required_spotify_fields = ['id', 'name', 'artists', 'album', 'duration_ms', 'popularity']
        for field in required_spotify_fields:
            if field not in spotify_data:
                print(f"Missing required Spotify field: {field}")
                return False
        
        # Check that we have at least some audio features
        if not reccobeats_data:
            # Don't return False here - we can still proceed with basic track info
            pass
        
        # Check required audio feature fields (only if we have ReccoBeats data)
        if reccobeats_data:
            required_audio_fields = ['danceability', 'energy', 'valence']
            audio_features_available = any(field in reccobeats_data for field in required_audio_fields)
            if not audio_features_available:
                # Still proceed - we have Spotify data
                pass
        
        return True
    
    def process_track_data(self, spotify_data: Dict[str, Any], reccobeats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format fetched data for database insertion."""
        processed_data = {
            'id': spotify_data['id'],
            'name': spotify_data['name'],
            'artist': ', '.join([artist['name'] for artist in spotify_data['artists']]),
            'album': spotify_data['album']['name'],
            'duration': spotify_data['duration_ms'] / 1000,  # Convert to seconds
            'popularity': spotify_data['popularity'],
            'explicit': spotify_data.get('explicit', False),
            'date_added': datetime.datetime.now().strftime('%Y-%m-%d'),
            'playlist': ['Currently Playing'],  # Default playlist
            'genres': []  # Will be populated later if needed
        }
        
        # Add audio features from ReccoBeats (if available)
        if reccobeats_data:
            audio_feature_fields = [
                'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                'loudness', 'mode', 'speechiness', 'tempo', 'valence'
            ]
            
            for field in audio_feature_fields:
                if field in reccobeats_data:
                    processed_data[field] = reccobeats_data[field]
            
            # Add ReccoBeats ID if available
            if 'reccobeats_id' in reccobeats_data:
                processed_data['reccobeats_id'] = reccobeats_data['reccobeats_id']
        else:
            # If no ReccoBeats data, we'll still have basic track info
            pass
        
        return processed_data


class TrackDataEnricher:
    """
    Enriches track data with percentile calculations and updates the database.
    """
    
    def __init__(self, user_path: str):
        self.user_path = user_path
        self.unique_songs_path = f'{user_path}unique_songs.ndjson.gz'
    
    def add_track_and_recalculate_percentiles(self, new_track_data: Dict[str, Any]) -> bool:
        """
        Add new track to database and recalculate percentiles for all tracks.
        
        Args:
            new_track_data: Processed track data from DynamicTrackFetcher
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing data
            if os.path.exists(self.unique_songs_path):
                try:
                    existing_df = pd.read_json(self.unique_songs_path, lines=True, compression='gzip')
                except Exception:
                    # If reading fails, start with empty DataFrame
                    existing_df = pd.DataFrame()
            else:
                existing_df = pd.DataFrame()
            
            # Add new track
            new_track_df = pd.DataFrame([new_track_data])
            updated_df = pd.concat([existing_df, new_track_df], ignore_index=True)
            
            # Recalculate percentiles for all tracks
            updated_df = self._calculate_percentiles(updated_df)
            
            # Save updated data
            self._save_updated_data(updated_df)
            
            return True
            
        except Exception as e:
            print(f"Error adding track and recalculating percentiles: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentiles for all numeric columns."""
        # Define columns that should have percentiles calculated
        percentile_cols = ['popularity', 'danceability', 'energy', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 
                          'duration', 'tempo', 'loudness']
        
        # Calculate percentiles for each column
        for col in percentile_cols:
            if col in df.columns and not df[col].isna().all():
                # Vectorized percentile calculation
                ranks = df[col].rank(method='max')
                df[f'{col}_percentile'] = 100.0 * (ranks - 1) / (len(df) - 1)
        
        return df
    
    def _save_updated_data(self, df: pd.DataFrame) -> None:
        """Save updated data using atomic write operations."""
        try:
            from SetupData import SetupData  # Import here to avoid circular imports
            
            setup_data = SetupData(self.user_path)
            setup_data._atomic_gzip_ndjson_write(self.unique_songs_path, df)
        except Exception as e:
            # Fallback to direct pandas save if SetupData import fails
            try:
                # Ensure the DataFrame is properly formatted for JSON export
                df_clean = df.copy()
                # Convert any problematic data types
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':
                        df_clean[col] = df_clean[col].astype(str)
                
                df_clean.to_json(self.unique_songs_path, orient='records', lines=True, compression='gzip')
            except Exception as fallback_error:
                print(f"Fallback save also failed: {fallback_error}")
                raise


# Currently Playing Page ----------------------------------------------------------------------------------------------
class CurrentlyPlayingPage():
    def __init__(self, song, artist, playlist, all_songs_df, unique_songs_df, user_path=None, spotify_client=None, song_id=None):
        self._song = song
        self._artist = artist
        self._playlist = playlist
        self._user_path = user_path
        self._spotify_client = spotify_client
        self._song_id = song_id  # Store the Spotify track ID

        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)
        
        # Fix genres column - convert string representations back to lists
        self._all_songs_df = self._fix_genres_column(self._all_songs_df)
        self._unique_songs_df = self._fix_genres_column(self._unique_songs_df)

        # Check if song exists in database
        # Use flexible matching for artists to handle multiple artists properly
        def artist_match(row_artist, target_artist):
            """Check if the target artist matches the row artist, handling multiple artists"""
            if row_artist == target_artist:
                return True
            
            # Split both into individual artists and compare sets
            row_artists = set([a.strip() for a in row_artist.split(',')])
            target_artists = set([a.strip() for a in target_artist.split(',')])
            
            # Check if all target artists are present in row artists
            return target_artists.issubset(row_artists)
        
        song_df = self._unique_songs_df[self._unique_songs_df['artist'].apply(lambda x: artist_match(x, artist))]
        self._song_df = song_df[song_df['name'] == song]
        
        # Track if we need to fetch data dynamically
        self._needs_dynamic_fetch = len(self._song_df.index) == 0
        self._dynamic_fetch_completed = False
        self._dynamic_fetch_error = None
        

        # Process genres for each individual artist
        artist_names = [name.strip() for name in self._artist.split(',')]
        self._artist_genres = []
        all_genres = []
        
        for artist_name in artist_names:
            # Look up this specific artist in the database
            artist_songs = self._unique_songs_df[self._unique_songs_df['artist'].apply(lambda x: artist_name in x.split(', '))]
            
            if len(artist_songs.index) > 0 and 'genres' in artist_songs.columns:
                # Get the first song for this artist
                row = artist_songs.iloc[0]
                raw_genres = row['genres']
                
                # Find the index of this artist in the song's artist list
                song_artists = [a.strip() for a in row['artist'].split(',')]
                try:
                    artist_index = song_artists.index(artist_name)
                except ValueError:
                    artist_index = 0  # Fallback to first artist
                
                # Get genres for this specific artist
                if isinstance(raw_genres, list) and len(raw_genres) > artist_index:
                    artist_genres = raw_genres[artist_index]
                    if isinstance(artist_genres, list) and artist_genres != ["N/A"]:
                        # Clean each genre string
                        cleaned_artist_genres = []
                        for genre in artist_genres:
                            if isinstance(genre, str):
                                clean_genre = genre.strip().strip('"\'[]')
                                if clean_genre:
                                    cleaned_artist_genres.append(clean_genre)
                        if cleaned_artist_genres:
                            self._artist_genres.append(cleaned_artist_genres)
                            all_genres.extend(cleaned_artist_genres)
                        else:
                            self._artist_genres.append(["N/A"])
                    else:
                        self._artist_genres.append(["N/A"])
                else:
                    self._artist_genres.append(["N/A"])
            else:
                self._artist_genres.append(["N/A"])
        
        # Flatten all genres for song_genres (used in other methods)
        if all_genres:
            self._song_genres = list(set(all_genres))  # Remove duplicates
        else:
            self._song_genres = ["N/A"]
        
        # Note: _clean_rank_columns() will be called after track data is available
    
    def _clean_rank_columns(self):
        """Clean rank columns by replacing 'nan' values with 'N/A' for consistency."""
        try:
            # Check if we have rank columns in the song data
            rank_columns = ['songs_short_rank', 'songs_med_rank', 'songs_long_rank']
            for col in rank_columns:
                if col in self._song_df.columns:
                    # Only replace actual NaN/None values, not valid rank numbers
                    # Check if the value is actually NaN/None before replacing
                    mask = (self._song_df[col].isna() | 
                           (self._song_df[col] == 'nan') | 
                           (self._song_df[col] == 'None') | 
                           (self._song_df[col] == None))
                    
                    # Only replace the NaN values, keep valid rank numbers
                    self._song_df.loc[mask, col] = 'N/A'
        except Exception as e:
            print(f"Error cleaning rank columns: {e}")
    
    def ensure_track_data_available(self) -> bool:
        """
        Ensure track data is available, fetching it dynamically if needed.
        Returns True if data is available, False otherwise.
        """
        if not self._needs_dynamic_fetch:
            return True
        
        if self._dynamic_fetch_completed:
            return True
        
        # Check each required parameter individually
        missing_params = []
        if not self._user_path:
            missing_params.append("user_path")
        if not self._spotify_client:
            missing_params.append("spotify_client")
        
        if missing_params:
            self._dynamic_fetch_error = f"Missing configuration for dynamic track fetching: {', '.join(missing_params)}"
            return False
        
        try:
            # Use the stored Spotify track ID
            track_id = self._song_id
            
            if not track_id:
                self._dynamic_fetch_error = "Could not determine Spotify track ID"
                return False
            
            # Initialize dynamic track fetcher
            fetcher = DynamicTrackFetcher(self._spotify_client, self._user_path)
            
            # Fetch track data
            spotify_data, reccobeats_data = fetcher.fetch_track_data(track_id)
            
            # Validate data
            if not fetcher.validate_track_data(spotify_data, reccobeats_data):
                self._dynamic_fetch_error = "Fetched track data validation failed"
                return False
            
            # Process data
            processed_data = fetcher.process_track_data(spotify_data, reccobeats_data)
            
            # Add to database and recalculate percentiles
            enricher = TrackDataEnricher(self._user_path)
            if enricher.add_track_and_recalculate_percentiles(processed_data):
                # Reload the song data from the updated database
                self._reload_song_data(processed_data)
                self._dynamic_fetch_completed = True
                return True
            else:
                self._dynamic_fetch_error = "Failed to add track to database"
                return False
                
        except Exception as e:
            self._dynamic_fetch_error = f"Dynamic track fetching failed: {str(e)}"
            return False
    
    def _reload_song_data(self, new_track_data: Dict[str, Any]) -> None:
        """Reload song data after dynamic fetching."""
        # Create a new song DataFrame with the fetched data
        new_song_df = pd.DataFrame([new_track_data])
        
        # Update the song_df reference
        self._song_df = new_song_df
        
        # Update the unique_songs_df to include the new track
        self._unique_songs_df = pd.concat([self._unique_songs_df, new_song_df], ignore_index=True)
        
        # Update genres if available
        if 'genres' in new_track_data and new_track_data['genres']:
            self._artist_genres = new_track_data['genres']
            self._song_genres = list({j for i in self._artist_genres for j in i})
        else:
            self._artist_genres = []
            self._song_genres = None

    def _calculate_playlist_similarity_knn(self, song_features, k=5):
        """
        Calculate the top k most similar playlists using KNN based on audio features.
        
        Args:
            song_features: Dictionary of audio features for the current song
            k: Number of most similar playlists to return
            
        Returns:
            List of tuples (playlist_name, similarity_score, playlist_features)
        """
        try:
            # Get all unique playlists (excluding the current song's playlist if it exists)
            all_playlists = self._all_songs_df['playlist'].unique()
            
            # Calculate median audio features for each playlist
            playlist_features = {}
            available_features = []
            
            # Determine which features are available in the song data
            for feature in FEATURE_COLS:
                if feature in song_features and not pd.isna(song_features[feature]):
                    available_features.append(feature)
            
            if not available_features:
                return []
            
            # Calculate median features for each playlist
            for playlist in all_playlists:
                playlist_df = self._all_songs_df[self._all_songs_df['playlist'] == playlist]
                
                # Only include playlists that have audio features data
                if len(playlist_df) > 0:
                    playlist_median = {}
                    has_features = False
                    
                    for feature in available_features:
                        if feature in playlist_df.columns and not playlist_df[feature].isna().all():
                            # Calculate median for this feature across all songs in the playlist
                            median_value = playlist_df[feature].median()
                            playlist_median[feature] = median_value
                            has_features = True
                        else:
                            playlist_median[feature] = median_value = 0.0
                    
                    if has_features:
                        playlist_features[playlist] = playlist_median
            
            if not playlist_features:
                return []
            
            # Calculate Euclidean distance between song features and each playlist
            similarities = []
            
            for playlist_name, playlist_feature_dict in playlist_features.items():
                # Calculate Euclidean distance
                distance = 0
                for feature in available_features:
                    song_val = song_features[feature]
                    playlist_val = playlist_feature_dict[feature]
                    
                    # Normalize popularity to 0-1 scale for fair comparison
                    if feature == 'popularity':
                        song_val = song_val / 100.0
                        playlist_val = playlist_val / 100.0
                    
                    # Calculate squared difference
                    diff = song_val - playlist_val
                    distance += diff * diff
                
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + distance)
                similarities.append((playlist_name, similarity_score, playlist_feature_dict))
            
            # Sort by similarity score (descending) and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            print(f"Error calculating playlist similarity: {e}")
            return []
    
    def _create_closest_playlists_chart(self, song_features, similar_playlists):
        """
        Create a vertical bar chart showing audio features comparison between
        the current song and the most similar playlists.
        
        Args:
            song_features: Dictionary of audio features for the current song
            similar_playlists: List of similar playlists from KNN calculation
            
        Returns:
            Plotly figure as HTML markup
        """
        try:
            # Determine which features are available
            available_features = []
            for feature in FEATURE_COLS:
                if feature in song_features and not pd.isna(song_features[feature]):
                    available_features.append(feature)
            
            if not available_features:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No audio features available for comparison</div>')
            
            # Create the bar chart
            fig = go.Figure()
            
            # Add trace for the current song
            song_values = []
            for feature in available_features:
                val = song_features[feature]
                # Normalize popularity to 0-1 scale
                if feature == 'popularity':
                    val = val / 100.0
                song_values.append(round(val, 3))
            
            fig.add_trace(go.Bar(
                name=self._song,
                x=available_features,
                y=song_values,
                text=song_values,
                textposition='auto',
                marker_color='#1DB954',  # Spotify green
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Feature: %{x}<br>' +
                            'Value: %{y}<br>' +
                            '<extra></extra>'
            ))
            
            # Add traces for similar playlists
            for i, (playlist_name, similarity_score, playlist_features) in enumerate(similar_playlists):
                playlist_values = []
                for feature in available_features:
                    val = playlist_features[feature]
                    # Normalize popularity to 0-1 scale
                    if feature == 'popularity':
                        val = val / 100.0
                    playlist_values.append(round(val, 3))
                
                color = COLORS[i % len(COLORS)]
                fig.add_trace(go.Bar(
                    name=f"{playlist_name} (Similarity: {similarity_score:.3f})",
                    x=available_features,
                    y=playlist_values,
                    text=playlist_values,
                    textposition='auto',
                    marker_color=color,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Feature: %{x}<br>' +
                                'Value: %{y}<br>' +
                                '<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Closest Fitting Playlists - Audio Features Comparison',
                xaxis_title='Audio Features',
                yaxis_title='Feature Values (Normalized)',
                barmode='group',
                showlegend=True,
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add a special marker to help the template detect this is the Closest Fitting Playlists chart
            html_content = fig.to_html(full_html=False)
            # Add a hidden marker that the template can use to detect this chart type
            marker = '<!-- CLOSEST_FITTING_PLAYLISTS_CHART -->'
            return Markup(marker + html_content)
            
        except Exception as e:
            print(f"Error creating closest playlists chart: {e}")
            return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Error creating chart: {str(e)}</div>')

    def _show_closest_fitting_playlists(self):
        """
        Show the "Closest Fitting Playlists" section when the song doesn't exist
        in the database or doesn't have rank data.
        """
        try:
            # Get the current song's audio features from the dynamic fetch or create default values
            song_features = {}
            
            if hasattr(self, '_song_df') and len(self._song_df.index) > 0:
                # Try to get features from the song data
                for feature in FEATURE_COLS:
                    if feature in self._song_df.columns:
                        song_features[feature] = self._song_df.iloc[0][feature]
            else:
                # Create default features based on what we know about the song
                # These would be placeholder values - in a real implementation,
                # you'd want to fetch these from Spotify API
                song_features = {
                    'popularity': 50,  # Default middle popularity
                    'danceability': 0.5,
                    'energy': 0.5,
                    'speechiness': 0.1,
                    'acousticness': 0.5,
                    'instrumentalness': 0.1,
                    'liveness': 0.1,
                    'valence': 0.5
                }
            
            # Calculate similar playlists using KNN
            similar_playlists = self._calculate_playlist_similarity_knn(song_features, k=5)
            
            if not similar_playlists:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Unable to calculate playlist similarities</div>')
            
            # Create the chart
            return self._create_closest_playlists_chart(song_features, similar_playlists)
            
        except Exception as e:
            print(f"Error showing closest fitting playlists: {e}")
            return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Error creating closest playlists view: {str(e)}</div>')

    def graph_top_rank_table(self):
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # Clean any 'nan' values in rank columns for consistency
        self._clean_rank_columns()
        
        # Check if song exists in database and has rank data
        if len(self._song_df.index) == 0:
            return self._show_closest_fitting_playlists()
        
        # Check if the song was added to any playlist
        song_in_playlists = self._all_songs_df[
            (self._all_songs_df['name'] == self._song) & 
            (self._all_songs_df['artist'] == self._artist)
        ]
        
        # If song was not added to any playlist, show Closest Fitting Playlists
        if len(song_in_playlists.index) == 0:
            return self._show_closest_fitting_playlists()
        
        # Check if we have rank columns and if they contain meaningful data
        rank_columns = ['songs_short_rank', 'songs_med_rank', 'songs_long_rank']
        has_ranks = all(col in self._song_df.columns for col in rank_columns)
        
        if has_ranks:
            # Song exists in database and has rank columns - show Top 50 Rankings table
            # Even if ranks are 'N/A', we should show the table structure since the song exists
            rank_table = shared_graph_top_rank_table(self._song_df)
            if rank_table is not None:
                return rank_table
            # If shared function fails, fall back to Closest Fitting Playlists
            return self._show_closest_fitting_playlists()
        
        # If we get here, either:
        # 1. Song doesn't exist in database, or
        # 2. Song exists but missing rank columns entirely
        # In both cases, show Closest Fitting Playlists
        return self._show_closest_fitting_playlists()
        


    def graph_song_percentiles_vs_avg(self):
        """Create a bar chart comparing song percentiles vs averages."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # This method would need to be implemented based on your existing logic
        # For now, return a placeholder
        return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song percentiles chart - implementation needed</div>')

    def graph_date_added_to_playlist(self):
        """Create a Gantt chart showing when the song was added to playlists."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # This method would need to be implemented based on your existing logic
        # For now, return a placeholder
        return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Date added to playlist chart - implementation needed</div>')

    def graph_count_timeline(self):
        """Create a timeline chart showing song count over time."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Count timeline chart - implementation needed</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Count timeline chart - implementation needed</div>')
        
        # This method would need to be implemented based on your existing logic
        # For now, return a placeholder
        return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Count timeline chart - implementation needed</div>')

    def graph_all_artists(self):
        """Create charts showing all artists data."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # This method would need to be implemented based on your existing logic
        # For now, return a placeholder
        return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">All artists charts - implementation needed</div>')

    def graph_artist_genres(self):
        """Create charts showing artist genres data."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # Check if we have genre data
        if not self._artist_genres or len(self._artist_genres) == 0:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No genre data available for this artist</div>')
        
        # Check if we have multiple artists (more than one artist in the string)
        num_artists = len(self._artist.split(','))
        
        if num_artists == 1:
            return [', '.join(self._song_genres), False]
        else:
            # Create Venn diagram for multiple artists
            try:
                fig = _venn_diagram_artist_genres(
                    self._artist.split(', '), self._artist_genres)
                
                # Handle case where Venn diagram creation fails
                if fig is None:
                    return [Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Unable to create artist genres comparison chart</div>'), True]
                return [fig, True]
            except Exception as e:
                print(f"Error creating artist genres chart: {e}")
                return [Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Error creating artist genres chart: {str(e)}</div>'), True]

    def graph_most_common_artists_with_same_genres(self):
        """Create charts showing most common artists with same genres."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # Check if we have a playlist and genre data
        if not self._playlist:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No playlist specified for this analysis</div>')
        
        if not self._song_genres or len(self._song_genres) == 0:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No genre data available for this artist</div>')
        
        try:
            # Get all songs in the current playlist
            playlist_mask = self._all_songs_df['playlist'] == self._playlist
            playlist_df = self._all_songs_df[playlist_mask]
            
            if len(playlist_df.index) == 0:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No songs found in this playlist</div>')
            
            # Find artists in the playlist that share genres with the current artist
            shared_genre_artists = {}
            
            # Check if current artist has genres other than "N/A"
            current_has_non_na_genres = any(genre != "N/A" for genre in self._song_genres)
            
            for _, row in playlist_df.iterrows():
                # Split the artist field on commas to handle multiple artists per song
                song_artists = [artist.strip() for artist in row['artist'].split(',')]
                
                # Check if this song has genre data
                if 'genres' in row and row['genres']:
                    song_genres = set()
                    for genre_list in row['genres']:
                        if isinstance(genre_list, list):
                            song_genres.update(genre_list)
                        else:
                            song_genres.add(genre_list)
                    
                    # Check for genre overlap (including N/A genres)
                    if song_genres & set(self._song_genres):
                        for artist in song_artists:
                            if artist in shared_genre_artists:
                                shared_genre_artists[artist] += 1
                            else:
                                shared_genre_artists[artist] = 1
            
            if not shared_genre_artists:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No artists with shared genres found in this playlist</div>')
            
            # Sort by count and take top 10
            top_artists = sorted(shared_genre_artists.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create the bar chart
            artist_names = [artist for artist, count in top_artists]
            counts = [count for artist, count in top_artists]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=counts,
                    y=artist_names,
                    orientation='h',
                    marker_color='#1DB954',
                    text=counts,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f'Most Common Artists with Shared Genres in {self._playlist}',
                xaxis_title='Number of Songs',
                yaxis_title='Artist',
                height=max(400, len(artist_names) * 40)
            )
            
            return Markup(fig.to_html(full_html=False))
            
        except Exception as e:
            print(f"Error creating most common artists chart: {e}")
            return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Error creating most common artists chart: {str(e)}</div>')

    def graph_song_genres_vs_avg(self, playlist=False):
        """Create charts showing song genres vs averages."""
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')
        
        # Check if we have genre data
        if not self._song_genres or len(self._song_genres) == 0:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No genre data available for this artist</div>')
        
        try:
            if playlist:
                # New functionality: Percentage of Songs with Same Genres as Artist in Current Playlist
                if not self._playlist:
                    # If no playlist specified, show genre percentages across all playlists
                    playlist_df = self._all_songs_df
                    title = 'Percentage of Songs With Same Genres As ' + self._artist + ' Across All Playlists'
                else:
                    # Get all songs in the current playlist
                    playlist_mask = self._all_songs_df['playlist'] == self._playlist
                    playlist_df = self._all_songs_df[playlist_mask]
                    
                    if len(playlist_df.index) == 0:
                        return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No songs found in this playlist</div>')
                    
                    title = 'Percentage of Songs With Same Genres As ' + self._artist + ' in ' + self._playlist
                xaxis = '% of Songs with Genres'

                percents = []
                song_counts = []
                length = len(playlist_df.index) if len(playlist_df.index) > 0 else 1
                
                for g in self._song_genres:
                    try:
                        mask = playlist_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                        count = len(playlist_df[mask].index)
                        percents.append(count/length*100)
                        song_counts.append(count)
                    except Exception:
                        percents.append(0.0)
                        song_counts.append(0)

                # Create hovertext with song counts
                hovertext = [f"{count} songs ({percent:.1f}%)" for count, percent in zip(song_counts, percents)]
                
                series = pd.Series(dict(zip(self._song_genres, percents)))
                return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True, hovertext=hovertext)
            else:
                # Original functionality for overall playlists
                avg_df = self._all_songs_df
                title = 'Percentage of Songs With Same Genres As ' + self._artist + ' Across All Playlists'
                xaxis = '% of Songs with Genres'

                percents = []
                length = len(avg_df.index) if len(avg_df.index) > 0 else 1
                for g in self._song_genres:
                    try:
                        mask = avg_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                        percents.append(len(avg_df[mask].index)/length*100)
                    except Exception:
                        percents.append(0.0)

                series = pd.Series(dict(zip(self._song_genres, percents)))
                return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True)
                
        except Exception as e:
            print(f"Error creating song genres chart: {e}")
            return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Error creating song genres chart: {str(e)}</div>')

    def graph_song_features_vs_avg(self):
        """
        Create a vertical bar graph comparing audio features across different data sources.
        Features: playlist median, all playlists median, artist median, and song's actual values.
        """
        # Ensure track data is available
        if not self.ensure_track_data_available():
            if self._needs_dynamic_fetch:
                if self._dynamic_fetch_error:
                    return Markup(f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Track data not available: {self._dynamic_fetch_error}</div>')
                else:
                    return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #5f5c6b; border-radius: 5px;">Fetching track data...</div>')
            else:
                return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')

        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        # Get the specific song data
        song_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
        song_df = song_df[song_df['artist'] == self._artist]

        if len(song_df) == 0:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')

        # Filter for available feature columns
        available_feature_cols = []
        for col in FEATURE_COLS:
            if col in song_df.columns and not song_df[col].isna().all():
                available_feature_cols.append(col)
        
        if not available_feature_cols:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No audio features data available for this song</div>')

        # Get song's actual values
        song_values = song_df[available_feature_cols].iloc[0]

        # Get playlist median (if playlist exists)
        playlist_median = None
        if self._playlist:
            playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] == self._playlist]
            if len(playlist_df) > 0:
                playlist_median = playlist_df[available_feature_cols].median()

        # Get all playlists median
        all_playlists_median = UNIQUE_SONGS_DF[available_feature_cols].median()

        # Get artist median for each artist in the comma-separated list
        artist_medians = []
        for artist_name in self._artist.split(', '):
            artist_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(lambda x: artist_name in x)]
            if len(artist_df) > 0:
                artist_median = artist_df[available_feature_cols].median()
                artist_medians.append((artist_name, artist_median))

        # Prepare data for plotting
        data = []
        names = []
        visibility = []
        
        # Add playlist median if it exists
        if playlist_median is not None:
            data.append(playlist_median)
            names.append(self._playlist)
            # Hide playlist median when playing from playlist, but keep in legend
            visibility.append('legendonly' if self._playlist else True)
        
        # Add all playlists median
        data.append(all_playlists_median)
        names.append('All Playlists')
        # Hide all playlists median when playing from playlist, but keep in legend
        visibility.append('legendonly' if self._playlist else True)
        
        # Add artist medians
        for artist_name, artist_median in artist_medians:
            data.append(artist_median)
            names.append(artist_name)
            # Hide artist medians when playing from playlist, but keep in legend
            visibility.append('legendonly' if self._playlist else True)
        
        # Add song's actual values
        data.append(song_values)
        names.append(self._song)
        # Always show song trace
        visibility.append(True)

        # Normalize popularity to 0-1 scale and round all values to 2 decimal places
        for i, df in enumerate(data):
            if 'popularity' in df.index:
                data[i]['popularity'] = round(df['popularity'] / 100, 2)
            # Round all other values to 2 decimal places
            for col in df.index:
                if col != 'popularity' and not pd.isna(df[col]):
                    data[i][col] = round(df[col], 2)

        # Create the bar chart
        fig = go.Figure()
        
        for i, (name, values, is_visible) in enumerate(zip(names, data, visibility)):
            # Convert to list and handle any NaN values
            y_values = []
            text_values = []
            for col in available_feature_cols:
                val = values[col]
                if pd.isna(val):
                    y_values.append(0)
                    text_values.append('N/A')
                else:
                    y_values.append(val)  # Already rounded above
                    text_values.append(val)  # Already rounded above
            
            fig.add_trace(go.Bar(
                name=name,
                x=available_feature_cols,
                y=y_values,
                text=text_values,
                textposition='auto',
                visible=is_visible,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Feature: %{x}<br>' +
                            'Value: %{y}<br>' +
                            '<extra></extra>'
            ))

        # Update layout
        if self._playlist:
            title = 'Song Audio Features vs. Medians in Playlist, All Playlists, & Artist'
        else:
            title = 'Song Audio Features vs. Medians in All Playlists & Artist'
            
        
        fig.update_layout(
            title=title,
            xaxis_title='Audio Features',
            yaxis_title='Feature Values',
            barmode='group',
            showlegend=True,
            height=500
        )

        return Markup(fig.to_html(full_html=False))

    def graph_song_percentiles_vs_avg(self):
        """
        Create percentile comparison graph for song audio features.
        
        Improvements made:
        - NaN values are filtered out before percentile calculation
        - Playlist percentiles use method='max' for consistent ranking
        - Artist percentiles use method='min' to avoid 100th percentile for single songs
        - Single song cases are set to 0 (since 0% of songs are below them)
        - Consistent data filtering and validation across all percentile calculations
        """
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        # Since playlist and artist df length will vary, make new percentile cols for them and then isolate song
        playlist_df = pd.DataFrame()  # Initialize empty DataFrame
        if self._playlist:
            playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] == self._playlist].copy()

            # Check if current song already exists in playlist
            exists_df = playlist_df[playlist_df['name'] == self._song]
            exists_df = exists_df[exists_df['artist'] == self._artist]
            
            # If song doesn't exist in playlist, add it for percentile calculation
            if len(exists_df) == 0:
                # Get the current song data from unique_songs_df
                current_song_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
                current_song_df = current_song_df[current_song_df['artist'] == self._artist]
                
                if len(current_song_df) > 0:
                    # Add current song to playlist_df for percentile calculation
                    playlist_df = pd.concat([playlist_df, current_song_df], ignore_index=True)

            # Calculate percentiles for the playlist (including current song if it wasn't there)
            for col in PERCENTILE_COLS:
                if col in playlist_df.columns and not playlist_df[col].isna().all():
                    # Filter out NaN values before calculating percentiles
                    valid_data = playlist_df[col].dropna()
                    if len(valid_data) > 0:
                        sz = valid_data.size - 1
                        if sz > 0:
                            # Use method='max' for consistent ranking
                            playlist_df[col + '_percentile'] = valid_data.rank(
                                method='max').apply(lambda x: round(100.0*(x-1)/sz, 2))
                        else:
                            # Single song case - set to 0 (since 0% of songs are below them)
                            playlist_df[col + '_percentile'] = 0.0
                    else:
                        playlist_df[col + '_percentile'] = 0.0
                else:
                    playlist_df[col + '_percentile'] = 0.0
            
            # Now get just the current song's data from the playlist
            playlist_df = playlist_df[(playlist_df['name'] == self._song) & (playlist_df['artist'] == self._artist)]

        artist_dfs = []
        for a in self._artist.split(', '):
            mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))
            artist_df = UNIQUE_SONGS_DF[mask]
            for col in PERCENTILE_COLS:
                if col in artist_df.columns and not artist_df[col].isna().all():
                    # Filter out NaN values before calculating percentiles
                    valid_data = artist_df[col].dropna()
                    if len(valid_data) > 0:
                        sz = valid_data.size - 1
                        if sz > 0:
                            # Use method='min' for artist percentiles to avoid 100th percentile for single songs
                            artist_df[col + '_percentile'] = valid_data.rank(
                                method='min').apply(lambda x: round(100.0*(x-1)/sz, 2))
                        else:
                            # Single song case - set to 0 (since 0% of songs are below them)
                            artist_df[col + '_percentile'] = 0.0
                    else:
                        artist_df[col + '_percentile'] = 0.0
                else:
                    artist_df[col + '_percentile'] = 0.0
            artist_df = artist_df[artist_df['name'] == self._song]
            if len(artist_df.index) == 0:
                return None
            artist_dfs.append(artist_df)

        # UNIQUE_SONGS_DF already has _percentile columns, so get the matching song by song and artist name
        avg_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
        avg_df = avg_df[avg_df['artist'] == self._artist]

        # Filter for available percentile columns (only show features that have data)
        available_percentile_cols = []
        for col in PERCENTILE_COLS:
            percentile_col = col + '_percentile'
            if percentile_col in avg_df.columns and not avg_df[percentile_col].isna().all():
                available_percentile_cols.append(percentile_col)
        
        if not available_percentile_cols:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No audio features data available for this song</div>')
        
        cols = available_percentile_cols
        
        # Replace NaN values with "N/A" for better display
        if len(playlist_df) > 0:
            for col in cols:
                playlist_df[col] = playlist_df[col].fillna('N/A')
        
        if len(avg_df) > 0:
            for col in cols:
                avg_df[col] = avg_df[col].fillna('N/A')
        
        for artist_df in artist_dfs:
            if len(artist_df) > 0:
                for col in cols:
                    artist_df[col] = artist_df[col].fillna('N/A')

        dfs = {}
        visibility = {}
        
        if self._playlist:
            # When playing from a playlist, show playlist trace by default
            dfs[self._playlist] = playlist_df[cols]
            visibility[self._playlist] = True
            
            # Add other traces but hide them initially, but keep in legend
            dfs['All Playlists'] = avg_df[cols]
            visibility['All Playlists'] = 'legendonly'
            
            for i, j in zip(self._artist.split(', '), [adf[cols] for adf in artist_dfs]):
                dfs[i] = j
                visibility[i] = 'legendonly'
        else:
            # When not playing from a playlist, show All Playlists trace by default
            dfs['All Playlists'] = avg_df[cols]
            visibility['All Playlists'] = True
            
            # Hide artist traces initially but keep them in legend
            for i, j in zip(self._artist.split(', '), [adf[cols] for adf in artist_dfs]):
                dfs[i] = j
                visibility[i] = 'legendonly'

        data = []
        # Create x-axis labels from the available percentile columns (remove '_percentile' suffix)
        x_labels = [col.replace('_percentile', '') for col in available_percentile_cols]
        
        for name, df in dfs.items():
            # Safety check: ensure DataFrame has data
            if len(df.index) == 0:
                print(f"Warning: DataFrame for {name} is empty, skipping...")
                continue
                
            # Handle NaN values and "N/A" values safely when converting to text
            y_values = df.iloc[0]
            text_values = []
            hover_values = []
            for val in y_values:
                if pd.isna(val) or val == 'N/A':
                    text_values.append('N/A')
                    hover_values.append('N/A')
                else:
                    # Convert to integer for bar display (no decimals)
                    bar_val = int(round(val, 0))
                    text_values.append(bar_val)
                    # Keep 2 decimal places for hover text (precision)
                    hover_val = round(val, 2)
                    hover_values.append(hover_val)
            
            # Filter out "N/A" values for the y-axis (but keep them in text and hover)
            y_values_filtered = []
            x_labels_filtered = []
            text_values_filtered = []
            hover_values_filtered = []
            
            for i, (y_val, x_label, text_val, hover_val) in enumerate(zip(y_values, x_labels, text_values, hover_values)):
                if y_val != 'N/A':
                    y_values_filtered.append(y_val)
                    x_labels_filtered.append(x_label)
                    text_values_filtered.append(text_val)
                    hover_values_filtered.append(hover_val)
            
            # Only add the bar if we have valid data
            if y_values_filtered:
                data.append(go.Bar(
                    name=name, 
                    x=x_labels_filtered, 
                    y=y_values_filtered, 
                    text=text_values_filtered, 
                    textposition='auto',
                    visible=visibility[name],
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Feature: %{x}<br>' +
                                 'Percentile: %{customdata}<extra></extra>',
                    customdata=hover_values_filtered
                ))

        # Check if we have any data to plot
        if not data:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No data available for percentile comparison</div>')

        fig = go.Figure(data=data)

        if self._playlist:
            title = 'Percentile of Song Audio Features by Playlist, All Playlists, & Artist'
        else:
            title = 'Percentile of Audio Features by All Playlists & Artist'

        fig.update_layout(barmode='group', title_text=title)

        return Markup(fig.to_html(full_html=False))

    def graph_date_added_to_playlist(self):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF[ALL_SONGS_DF['name'] == self._song]
        df = df[df['artist'] == self._artist]

        # today = datetime.datetime.now().astimezone().date()
        today = datetime.datetime.utcnow().date()
        new = pd.DataFrame({'Task': df['playlist'], 'Start': df['date_added'], 'Finish': [
                           today]*len(df['playlist'])})

        fig = ff.create_gantt(new)
        fig.update_layout(title_text='Timeline Of When ' + self._song + ' Was Added To Playlists',
                          height=min(max((len(df)+1)*50, 400), 500))

        return Markup(fig.to_html(full_html=False))

    def graph_count_timeline(self):
        if not self._playlist:
            # If no playlist specified, show the song's timeline across all playlists
            playlist_df = self._all_songs_df[
                (self._all_songs_df['name'] == self._song) & 
                (self._all_songs_df['artist'] == self._artist)
            ]
            title_suffix = " Across All Playlists"
        else:
            # Filter data for the current playlist
            playlist_df = self._all_songs_df[self._all_songs_df['playlist'] == self._playlist]
            title_suffix = f" in {self._playlist}"
        
        if len(playlist_df.index) == 0:
            return None
        
        # Group by date and count songs, create hovertext
        timeline_data = playlist_df.groupby('date_added').agg({
            'name': lambda x: list(x),
            'artist': lambda x: list(x)
        }).reset_index()
        
        # Create hovertext with song names separated by <br>
        hovertext = []
        for _, row in timeline_data.iterrows():
            song_names = [f"{name} by {artist}" for name, artist in zip(row['name'], row['artist'])]
            hovertext.append('<br>'.join(song_names))
        
        # Create the line graph
        fig = go.Figure()
        
        # Add main timeline trace
        trace_name = f'Songs Added{title_suffix}' if self._playlist else f'{self._song} Added to Playlists'
        fig.add_trace(go.Scatter(
            x=timeline_data['date_added'],
            y=timeline_data['name'].apply(len),
            mode='lines+markers',
            name=trace_name,
            line=dict(color=MAIN_TIMELINE_COLOR),
            marker=dict(size=8),
            hovertext=hovertext,
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Date: %{x}<br>' +
                         'Songs Added: %{y}<extra></extra>'
        ))
        
        # Find the date when the current song was added to the current playlist
        current_song_df = playlist_df[
            (playlist_df['name'] == self._song) & 
            (playlist_df['artist'] == self._artist)
        ]
        
        if len(current_song_df.index) > 0:
            current_song_date = current_song_df.iloc[0]['date_added']
            
            # Convert string date to datetime object for proper vline positioning
            if isinstance(current_song_date, str):
                current_song_date = datetime.datetime.fromisoformat(current_song_date.replace('Z', '+00:00'))
            
            # Add vertical line at the date the current song was added
            fig.add_vline(
                x=current_song_date,
                line_dash="dash",
                line_color=YEARLY_VLINE_COLOR,
                line_width=2
            )
        
        # Update layout
        title_text = f"Timeline of Songs Added{title_suffix}" if self._playlist else f"Timeline of {self._song} Added to Playlists"
        fig.update_layout(
            title_text=title_text,
            xaxis_title="Date",
            yaxis_title="Number of Songs Added",
            height=500,
            showlegend=False
        )
        
        # Update x-axis to show dates properly
        fig.update_xaxes(
            # don't need this
            # rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        return Markup(fig.to_html(full_html=False))

    def graph_top_playlists_by_artist(self, artist_name):
        return shared_graph_top_playlists_by_artist(self._all_songs_df, artist_name)

    def graph_top_songs_by_artist(self, artist_name):
        return shared_graph_top_songs_by_artist(self._unique_songs_df, artist_name)

    def graph_artist_genres(self):
        if len(self._artist_genres) == 1:
            return [', '.join(self._song_genres), False]
        else:
            fig = _venn_diagram_artist_genres(
                self._artist.split(', '), self._artist_genres)
            return [fig, True]

    # What percentage of songs in a playlist or among all playlists have these genres?
    def graph_song_genres_vs_avg(self, playlist=False):
        """Graph percentage of songs with same genres as current artist"""
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df
        # Require genres column to compute percentages
        if 'genres' not in UNIQUE_SONGS_DF.columns or 'genres' not in ALL_SONGS_DF.columns:
            return None
        
        if playlist:
            # New functionality: Percentage of Songs with Same Genres as Artist in Current Playlist
            if not self._playlist:
                return None
            
            # Get all songs in the current playlist
            playlist_mask = ALL_SONGS_DF['playlist'] == self._playlist
            playlist_df = ALL_SONGS_DF[playlist_mask]
            
            if len(playlist_df.index) == 0:
                return None
            
            title = 'Percentage of Songs With Same Genres As ' + self._artist + ' in ' + self._playlist
            xaxis = '% of Songs with Genres'

            percents = []
            song_counts = []
            length = len(playlist_df.index) if len(playlist_df.index) > 0 else 1
            for g in self._song_genres:
                try:
                    mask = playlist_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                    count = len(playlist_df[mask].index)
                    percents.append(count/length*100)
                    song_counts.append(count)
                except Exception:
                    percents.append(0.0)
                    song_counts.append(0)

            # Create hovertext with song counts
            hovertext = [f"{count} songs ({percent:.1f}%)" for count, percent in zip(song_counts, percents)]
            
            series = pd.Series(dict(zip(self._song_genres, percents)))
            return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True, hovertext=hovertext)
        else:
            # Original functionality for overall playlists
            avg_df = ALL_SONGS_DF
            title = 'Percentage of Songs With Same Genres As ' + self._artist + ' Across All Playlists'
            xaxis = '% of Songs with Genres'

            percents = []
            length = len(avg_df.index) if len(avg_df.index) > 0 else 1
            for g in self._song_genres:
                try:
                    mask = avg_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                    percents.append(len(avg_df[mask].index)/length*100)
                except Exception:
                    percents.append(0.0)

            series = pd.Series(dict(zip(self._song_genres, percents)))
            return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True)

    def graph_most_common_artists_with_same_genres(self):
        """Graph most common artists in playlist with same genres as current artist"""
        ALL_SONGS_DF = self._all_songs_df
        # Require genres column to compute percentages
        if 'genres' not in ALL_SONGS_DF.columns:
            return None
        
        if not self._playlist:
            return None
        
        # Get all songs in the current playlist
        playlist_mask = ALL_SONGS_DF['playlist'] == self._playlist
        playlist_df = ALL_SONGS_DF[playlist_mask]
        
        if len(playlist_df.index) == 0:
            return None
        
        # Find artists in the playlist that share genres with the current artist
        shared_genre_artists = {}
        
        # Check if current artist has genres other than "N/A"
        current_has_non_na_genres = any(genre != "N/A" for genre in self._song_genres)
        
        for _, row in playlist_df.iterrows():
            # Split the artist field on commas to handle multiple artists per song
            song_artists = [artist.strip() for artist in row['artist'].split(',')]
            
            # Check if this artist has any shared genres
            try:
                artist_genres = row['genres']
                if artist_genres and isinstance(artist_genres, list):
                    # Flatten the nested genre structure
                    flat_artist_genres = {genre for genre_list in artist_genres for genre in genre_list}
                    
                    # Check for shared genres
                    shared_genres = set(self._song_genres) & flat_artist_genres
                    
                    if shared_genres:
                        # Process each artist in the song separately
                        for artist_name in song_artists:
                            # Skip the current artist
                            if artist_name == self._artist:
                                continue
                            
                            # Include all artists that share any genres, including N/A
                            
                            # This artist shares genres with the current artist
                            if artist_name not in shared_genre_artists:
                                shared_genre_artists[artist_name] = {
                                    'count': 0,
                                    'songs': [],
                                    'shared_genres': list(shared_genres)
                                }
                            
                            shared_genre_artists[artist_name]['count'] += 1
                            shared_genre_artists[artist_name]['songs'].append(row['name'])
            except Exception:
                continue
        
        if not shared_genre_artists:
            # No artists with shared genres found
            fig = go.Figure()
            fig.update_layout(
                title_text=f'Most Common Artists in {self._playlist} with Same Genres as {self._artist}',
                xaxis_title='Number of Songs',
                yaxis_title='Artist',
                annotations=[{
                    'text': f'No artists in {self._playlist} share genres with {self._artist}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                height=300
            )
            return Markup(fig.to_html(full_html=False))
        
        # Create data for the horizontal bar chart
        artists = []
        counts = []
        hovertexts = []
        
        # Sort by count (descending) and take top 10 artists
        sorted_artists = sorted(shared_genre_artists.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        
        for artist_name, data in sorted_artists:
            artists.append(artist_name)
            counts.append(data['count'])
            
            # Create hovertext with shared genres first, then songs below
            shared_genres_text = ', '.join(data['shared_genres'])
            # Cap the songs to MAX_HOVER_ROWS for hovertext
            songs_text = '<br>'.join(data['songs'][:MAX_HOVER_ROWS])
            hovertext = f"<b>Shared Genres:</b> {shared_genres_text}<br><br><b>Songs:</b><br>{songs_text}"
            hovertexts.append(hovertext)
        
        # Create the horizontal bar chart
        fig = go.Figure(data=[go.Bar(
            x=counts,
            y=artists,
            orientation='h',
            text=counts,
            textposition='auto',
            hovertext=hovertexts,
            hovertemplate='<b>%{hovertext}</b><extra></extra>',
            marker_color=COLORS[0]  # Use first color from COLORS
        )])
        
        # Update layout
        fig.update_layout(
            title_text=f'Most Common Artists in {self._playlist} with Same Genres as {self._artist}',
            xaxis_title='Number of Songs',
            yaxis_title='Artist',
            yaxis={'categoryorder': 'total ascending'},
            height=min(max((len(artists)+1)*40, 300), 600),  # Responsive height
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        fig.update_yaxes(automargin=True)
        
        return Markup(fig.to_html(full_html=False))

    def graph_all_artists(self):
        artist_top_graphs = []
        for i in self._artist.split(', '):
            artist_top_playlists_bar = self.graph_top_playlists_by_artist(i)
            artist_top_songs_table = self.graph_top_songs_by_artist(i)
            artist_top_graphs.append(
                (artist_top_playlists_bar, artist_top_songs_table))
        return artist_top_graphs

    def _fix_genres_column(self, df):
        """Convert genre strings back to lists after loading from NDJSON"""
        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(self._parse_genre_string)
        return df

    def _parse_genre_string(self, genre_value):
        """Parse genre from string representation back to list"""
        # Handle pandas Series or arrays properly
        if hasattr(genre_value, '__iter__') and not isinstance(genre_value, (str, list)):
            # If it's a pandas Series or array, take the first value
            if len(genre_value) > 0:
                genre_value = genre_value.iloc[0] if hasattr(genre_value, 'iloc') else genre_value[0]
            else:
                return [["N/A"]]
        
        # Check for None or NaN values
        if genre_value is None:
            return [["N/A"]]
        
        # Handle pandas NaN values
        try:
            if pd.isna(genre_value):
                return [["N/A"]]
        except:
            pass
        
        if isinstance(genre_value, list):
            # Check if this is already a proper list of genres (not nested)
            # If all items are strings, return as-is (already properly formatted)
            if all(isinstance(item, str) for item in genre_value):
                # Clean string items
                cleaned_genres = []
                for item in genre_value:
                    clean_item = item.strip().strip('"\'[]')
                    if clean_item and clean_item != 'N/A':
                        cleaned_genres.append(clean_item)
                return cleaned_genres if cleaned_genres else ['N/A']
            
            # Handle nested list structure (legacy format)
            cleaned_list = []
            for item in genre_value:
                if isinstance(item, list):
                    # Clean nested list
                    cleaned_nested = []
                    for nested_item in item:
                        if isinstance(nested_item, str):
                            clean_item = nested_item.strip().strip('"\'[]')
                            if clean_item:
                                cleaned_nested.append(clean_item)
                    if cleaned_nested:
                        cleaned_list.append(cleaned_nested)
                    else:
                        cleaned_list.append(["N/A"])
                elif isinstance(item, str):
                    # Clean string item
                    clean_item = item.strip().strip('"\'[]')
                    if clean_item:
                        cleaned_list.append([clean_item])
                    else:
                        cleaned_list.append(["N/A"])
                else:
                    cleaned_list.append(["N/A"])
            return cleaned_list if cleaned_list else [["N/A"]]
        
        if isinstance(genre_value, str):
            # Handle JSON-like strings
            if genre_value.startswith('[') and genre_value.endswith(']'):
                try:
                    import json
                    parsed = json.loads(genre_value)
                    if isinstance(parsed, list):
                        # Clean the parsed list
                        cleaned_list = []
                        for item in parsed:
                            if isinstance(item, list):
                                cleaned_nested = []
                                for nested_item in item:
                                    if isinstance(nested_item, str):
                                        clean_item = nested_item.strip().strip('"\'[]')
                                        if clean_item:
                                            cleaned_nested.append(clean_item)
                                if cleaned_nested:
                                    cleaned_list.append(cleaned_nested)
                                else:
                                    cleaned_list.append(["N/A"])
                            elif isinstance(item, str):
                                clean_item = item.strip().strip('"\'[]')
                                if clean_item:
                                    cleaned_list.append([clean_item])
                                else:
                                    cleaned_list.append(["N/A"])
                        return cleaned_list if cleaned_list else [["N/A"]]
                except:
                    pass
            
            # Handle comma-separated strings
            if ',' in genre_value:
                genres = [g.strip().strip('"\'[]') for g in genre_value.split(',') if g.strip()]
                return [genres] if genres else [["N/A"]]
            
            # Single genre
            if genre_value.strip():
                clean_genre = genre_value.strip().strip('"\'[]')
                if clean_genre:
                    return [[clean_genre]]
        
        return [["N/A"]]


# Home Page ----------------------------------------------------------------------------------------------

class HomePage(OptimizedPageBase):
    def __init__(self, path, all_songs_df, unique_songs_df, playlist_dict=None):
        # Initialize parent class with optimizations
        super().__init__(path, all_songs_df, unique_songs_df)
        
        self._today = datetime.datetime.now().astimezone()
        self._playlist_dict = playlist_dict or {}

        # Calculate global artist averages once for consistency across all artist timelines
        self._calculate_global_artist_averages()

        # Progressive loading: Don't generate all components upfront
        # Components will be loaded on-demand when requested

    def _calculate_global_artist_averages(self):
        """Calculate global averages for liked songs and total songs per artist across all playlists"""
        liked_songs_df = self._all_songs_df[self._all_songs_df['playlist'] == 'Liked Songs']
        
        if len(liked_songs_df.index) > 0:
            # Get all unique artists from the entire dataset
            all_artists = set()
            for artist_list in self._all_songs_df['artist']:
                artists = [a.strip() for a in artist_list.split(', ')]
                all_artists.update(artists)
            
            # Calculate average liked songs per artist (only for artists who have liked songs)
            artist_liked_counts = []
            for artist in all_artists:
                artist_liked = liked_songs_df[liked_songs_df['artist'].apply(
                    lambda x: artist in x.split(', '))]
                if len(artist_liked.index) > 0:  # Only include artists who have liked songs
                    artist_liked_counts.append(len(artist_liked.index))
            
            self._global_avg_liked_songs_per_artist = sum(artist_liked_counts) / len(artist_liked_counts) if artist_liked_counts else 0
            
            # Calculate average total songs per artist across all playlists (excluding liked songs)
            artist_total_counts = []
            non_liked_df = self._all_songs_df[self._all_songs_df['playlist'] != 'Liked Songs']
            for artist in all_artists:
                artist_songs = non_liked_df[non_liked_df['artist'].apply(
                    lambda x: artist in x.split(', '))]
                artist_total_counts.append(len(artist_songs.index))
            
            self._global_avg_total_songs_per_artist = sum(artist_total_counts) / len(artist_total_counts) if artist_total_counts else 0

            # Calculate global playlist averages
            # Get all unique playlists (excluding Liked Songs)
            all_playlists = self._all_songs_df[self._all_songs_df['playlist'] != 'Liked Songs']['playlist'].unique()
            
            # Calculate average songs added per playlist
            playlist_total_counts = []
            for playlist in all_playlists:
                playlist_songs = self._all_songs_df[self._all_songs_df['playlist'] == playlist]
                playlist_total_counts.append(len(playlist_songs.index))
            
            self._global_avg_total_songs_per_playlist = sum(playlist_total_counts) / len(playlist_total_counts) if playlist_total_counts else 0
            
            # Calculate average liked songs per playlist
            # Only include playlists that have at least one liked song
            playlist_liked_counts = []
            liked_songs_df = self._all_songs_df[self._all_songs_df['playlist'] == 'Liked Songs']
            
            for playlist in all_playlists:
                playlist_songs = self._all_songs_df[self._all_songs_df['playlist'] == playlist]
                
                # Find songs that exist in both the playlist and Liked Songs
                # Merge on name and artist to find matches
                playlist_liked_songs = pd.merge(playlist_songs[['name', 'artist']], 
                                               liked_songs_df[['name', 'artist']], 
                                               on=['name', 'artist'])
                
                if len(playlist_liked_songs.index) > 0:  # Only include playlists that have liked songs
                    playlist_liked_counts.append(len(playlist_liked_songs.index))
            
            self._global_avg_liked_songs_per_playlist = sum(playlist_liked_counts) / len(playlist_liked_counts) if playlist_liked_counts else 0
        else:
            self._global_avg_liked_songs_per_artist = 0
            self._global_avg_total_songs_per_artist = 0
            self._global_avg_liked_songs_per_playlist = 0
            self._global_avg_total_songs_per_playlist = 0

    def get_global_artist_averages(self):
        """Return the pre-calculated global artist averages"""
        return self._global_avg_liked_songs_per_artist, self._global_avg_total_songs_per_artist

    def get_global_playlist_averages(self):
        """Return the pre-calculated global playlist averages"""
        return self._global_avg_liked_songs_per_playlist, self._global_avg_total_songs_per_playlist

    def load_on_this_date(self):
        return self.load_component_progressively('on_this_date', self._graph_on_this_date)

    def load_timeline(self):
        return self.load_component_progressively('timeline', self._graph_count_timeline)

    def load_last_added(self):
        return self.load_component_progressively('last_added', self._graph_last_added)

    def load_totals(self):
        return self.load_component_progressively('totals', self._get_library_totals)

    def load_first_times(self):
        # Always return a list so templates that iterate do not split characters
        if hasattr(self, '_first_times') and isinstance(self._first_times, list) and len(self._first_times) > 0:
            return self._first_times
        return ['No notable anniversaries found']

    def _graph_on_this_date(self):
        '''Output1 = Table = Year | Playlist | Songs Added
        Output2 = List = You created playlist / added song from artist for the first time'''
        today = str(self._today.date())[5:]
        
        # Use optimized date filtering
        df = self.vectorized_date_filter('date_added', today)
        first_times = []

        # Vectorized year extraction
        df['year'] = df['date_added'].str[:4]
        df = df[['name', 'artist', 'playlist', 'year']]

        if len(df.index) == 0:
            first_times.append('No notable anniversaries found')
        else:
            # Vectorized artist extraction and filtering
            unique_artists = set()
            for artist_list in df['artist']:
                unique_artists.update(artist_list.split(', '))
            
            for artist in unique_artists:
                # Use optimized artist data retrieval
                artist_df = self.get_optimized_artist_data(artist)
                if len(artist_df) > 0:
                    first_date = artist_df['date_added'].sort_values(ascending=True).iloc[0]
                    if str(first_date)[5:] == today:
                        # Check if artist meets criteria: 5+ songs OR at least 1 liked song
                        artist_song_count = len(artist_df.index)
                        has_liked_songs = 'Liked Songs' in artist_df['playlist'].values
                        
                        if artist_song_count >= MIN_ARTIST_SONG_COUNT_FOR_ON_THIS_DATE or has_liked_songs:
                            years_ago = int(str(self._today.date())[:4]) - int(str(first_date)[:4])
                            first_times.append(
                                (years_ago, 'You first added a song from the artist <a href="/artists/' + quote(artist) + '">' + artist + '</a> ' + str(years_ago) + ' years ago today!'))
            
            first_times = [i[1] for i in sorted(first_times, key=lambda x: x[0], reverse=True)]
        
            # Vectorized playlist processing
            for playlist_name in df['playlist'].unique():
                # Use optimized playlist data retrieval
                playlist_data = self.get_optimized_playlist_data(playlist_name)
                if playlist_data:
                    # Get first date for this playlist
                    playlist_df = self._all_songs_df[self._all_songs_df['playlist'] == playlist_name]
                    if len(playlist_df) > 0:
                        first_date = playlist_df['date_added'].sort_values(ascending=True).iloc[0]
                        if str(first_date)[5:] == today:
                            years_ago = int(str(self._today.date())[:4]) - int(str(first_date)[:4])
                            
                            # Create playlist link using ID if available
                            if playlist_name == 'Liked Songs':
                                # Special case for Liked Songs
                                playlist_link = '<a href="/playlists/liked_songs">' + playlist_name + '</a>'
                            elif playlist_name in self._playlist_dict:
                                # Use playlist ID for regular playlists
                                playlist_id = self._playlist_dict[playlist_name]
                                playlist_link = f'<a href="/playlists/{playlist_id}">{playlist_name}</a>'
                            else:
                                # Fallback to name if no ID found
                                playlist_link = playlist_name
                            
                            first_times.append(
                                'You created the playlist ' + playlist_link + ' ' + str(years_ago) + ' years ago today!')

        self._first_times = first_times
        
        df = df.groupby(['playlist', 'year'], as_index=False)[
            ['name', 'artist']].agg(lambda x: ', '.join(x))
        df = df.sort_values(by='year')
        try:
            df.columns=df.columns.str.strip()
            years = df['year'].to_list()
            
            # Create clickable playlist links for the table
            playlist_links = []
            for playlist_name in df['playlist']:
                if playlist_name == 'Liked Songs':
                    # Special case for Liked Songs
                    playlist_links.append('<a href="/playlists/liked_songs">Liked Songs</a>')
                elif playlist_name in self._playlist_dict:
                    # Use playlist ID for regular playlists
                    playlist_id = self._playlist_dict[playlist_name]
                    playlist_links.append(f'<a href="/playlists/{playlist_id}">{playlist_name}</a>')
                else:
                    # Fallback to name if no ID found
                    playlist_links.append(playlist_name)
            
            names = df['name'].to_list()
            
            # Create clickable artist links for the table
            artist_links = []
            for artist_list in df['artist']:
                # Split the comma-separated artists and create links for each
                individual_artists = [a.strip() for a in artist_list.split(', ')]
                artist_links_for_row = []
                for artist_name in individual_artists:
                    # Remove duplicates while preserving order
                    if artist_name not in [a['name'] for a in artist_links_for_row]:
                        artist_links_for_row.append({
                            'name': artist_name,
                            'link': f'<a href="/artists/{quote(artist_name)}">{artist_name}</a>'
                        })
                # Join the artist links with commas
                artist_links.append(', '.join([a['link'] for a in artist_links_for_row]))
        except:
            return

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Year', 'Playlist', 'Songs Added', 'Artists'],
                line_color='darkslategray',
                fill_color='royalblue',
                font=dict(color='white', size=18),
                height=40
            ),
            cells=dict(
                values=[years, playlist_links, names, artist_links],
                line_color='darkslategray',
                align='center',
                font_size=[18, 14],
                height=30)
        )
        ])

        return Markup(fig.to_html(full_html=False))


    def _graph_count_timeline(self):
        """Create a custom timeline with liked songs as a separate green trace"""
        # Create timeline for all playlists (excluding liked songs)
        all_playlists_df = self._all_songs_df[self._all_songs_df['playlist'] != 'Liked Songs']
        liked_songs_df = self._all_songs_df[self._all_songs_df['playlist'] == 'Liked Songs']
        
        # Group by date for main timeline (excluding liked songs)
        main_timeline = all_playlists_df.groupby('date_added').size().reset_index(name='count')
        main_timeline = main_timeline.sort_values('date_added')
        
        # Group by date for liked songs timeline
        liked_timeline = liked_songs_df.groupby('date_added').size().reset_index(name='count')
        liked_timeline = liked_timeline.sort_values('date_added')
        
        # Create hovertext mapping for main timeline
        main_hovertext = []
        for date in main_timeline['date_added']:
            songs_on_date = all_playlists_df[all_playlists_df['date_added'] == date]['name'].tolist()
            hovertext = '<br>'.join(songs_on_date)
            main_hovertext.append(hovertext)
        
        # Create hovertext mapping for liked songs timeline
        liked_hovertext = []
        for date in liked_timeline['date_added']:
            songs_on_date = liked_songs_df[liked_songs_df['date_added'] == date]['name'].tolist()
            hovertext = '<br>'.join(songs_on_date)
            liked_hovertext.append(hovertext)
        
        # Create figure with subplots
        fig = subplots.make_subplots(
            rows=1, cols=1,
            vertical_spacing=0.05,
            subplot_titles=('<b>Timeline of Adding Songs</b>')
        )
        
        # Add main timeline traces (Line, Continuous, Bar)
        # Line trace for main timeline
        main_line_trace = go.Scatter(
            x=main_timeline['date_added'],
            y=main_timeline['count'],
            mode='lines+markers',
            name='All Playlists',
            line=dict(color=MAIN_TIMELINE_COLOR),
            visible=True,
            hovertext=main_hovertext,
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
        )
        
        # Continuous trace for main timeline (cumulative)
        main_continuous_trace = go.Scatter(
            x=main_timeline['date_added'],
            y=main_timeline['count'].cumsum(),
            mode='lines+markers',
            name='All Playlists (Cumulative)',
            line=dict(color=MAIN_TIMELINE_COLOR),
            visible=False,
            hovertext=main_hovertext,
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative Count: %{y}<br>%{hovertext}<extra></extra>'
        )
        
        # Bar trace for main timeline
        main_bar_trace = go.Bar(
            x=main_timeline['date_added'],
            y=main_timeline['count'],
            name='All Playlists',
            marker=dict(color=MAIN_TIMELINE_COLOR),
            visible=False,
            hovertext=main_hovertext,
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
        )
        
        # Add liked songs traces (Line, Continuous, Bar)
        if len(liked_timeline.index) > 0:
            # Line trace for liked songs
            liked_line_trace = go.Scatter(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'],
                mode='lines+markers',
                name='Liked Songs',
                line=dict(color=LIKED_TIMELINE_COLOR),
                visible=True,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Continuous trace for liked songs (cumulative)
            liked_continuous_trace = go.Scatter(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'].cumsum(),
                mode='lines+markers',
                name='Liked Songs (Cumulative)',
                line=dict(color=LIKED_TIMELINE_COLOR),
                visible=False,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Bar trace for liked songs
            liked_bar_trace = go.Bar(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'],
                name='Liked Songs',
                marker=dict(color=LIKED_TIMELINE_COLOR),
                visible=False,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Add all traces to figure
            fig.add_trace(main_line_trace)
            fig.add_trace(main_continuous_trace)
            fig.add_trace(main_bar_trace)
            fig.add_trace(liked_line_trace)
            fig.add_trace(liked_continuous_trace)
            fig.add_trace(liked_bar_trace)
        else:
            # Add only main traces if no liked songs
            fig.add_trace(main_line_trace)
            fig.add_trace(main_continuous_trace)
            fig.add_trace(main_bar_trace)
        
        # Add anniversary vertical lines
        if len(all_playlists_df.index) > 0:
            first_date = all_playlists_df['date_added'].sort_values(ascending=True).iloc[0]
            first_year = int(str(first_date)[:4])
            current_year = int(str(self._today.date())[:4])
            
            # Get month and day from today's date (current date)
            today_month = self._today.month
            today_day = self._today.day
            
            for year in range(first_year, current_year + 1):
                try:
                    anniversary_date = datetime.datetime(year, today_month, today_day)
                    fig.add_vline(
                        x=anniversary_date,
                        line_width=3,
                        line_dash="dash",
                        line_color="green"
                    )
                except ValueError:
                    # Skip invalid dates (e.g., February 29 in non-leap years)
                    continue
        
        # Create buttons for Line, Continuous, Bar views
        buttons = []
        
        # Line button (shows line traces)
        line_visibility = [True, False, False, True, False, False] if len(liked_timeline.index) > 0 else [True, False, False]
        buttons.append(dict(
            method='update',
            label='Line',
            args=[{'visible': line_visibility}]
        ))
        
        # Continuous button (shows continuous traces)
        continuous_visibility = [False, True, False, False, True, False] if len(liked_timeline.index) > 0 else [False, True, False]
        buttons.append(dict(
            method='update',
            label='Continuous',
            args=[{'visible': continuous_visibility}]
        ))
        
        # Bar button (shows bar traces)
        bar_visibility = [False, False, True, False, False, True] if len(liked_timeline.index) > 0 else [False, False, True]
        buttons.append(dict(
            method='update',
            label='Bar',
            args=[{'visible': bar_visibility}]
        ))
        
        # Add button menu
        updatemenus = [dict(
            type='buttons',
            direction='right',
            active=0,
            y=1.3,
            x=0.6,
            buttons=buttons
        )]
        
        # Update layout
        fig.update_layout(
            updatemenus=updatemenus,
            showlegend=True,
            title='<b>Timeline of Adding Songs</b>',
            yaxis_title='# Songs',
            xaxis_title='Date',
            barmode='stack'
        )
        
        # Update x-axis with range selector
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        return Markup(fig.to_html(full_html=False))

    def _graph_last_added(self):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF.sort_values(by='date_added', ascending=False)
        last_date = df['date_added'].to_list()[0]
        distance = (self._today.date() - datetime.date(*
                    map(int, last_date.split('-')))).days+1

        df = ALL_SONGS_DF[ALL_SONGS_DF['date_added'] == last_date]
        num_songs = len(df.index)
        df = df[['name', 'playlist']]
        df = df.groupby(['playlist'], as_index=False)[
            ['name']].agg(lambda x: list(x))

        # Create clickable playlist links using IDs
        playlist_links = []
        for playlist_name in df['playlist']:
            if playlist_name == 'Liked Songs':
                # Special case for Liked Songs
                playlist_links.append('<a href="/playlists/liked_songs">Liked Songs</a>')
            elif playlist_name in self._playlist_dict:
                # Use playlist ID for regular playlists
                playlist_id = self._playlist_dict[playlist_name]
                playlist_links.append(f'<a href="/playlists/{playlist_id}">{playlist_name}</a>')
            else:
                # Fallback to name if no ID found
                playlist_links.append(playlist_name)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Playlist', 'Songs Added'],
                line_color='darkslategray',
                fill_color='royalblue',
                font=dict(color='white', size=18),
                height=40
            ),
            cells=dict(
                values=[playlist_links, [', '.join(i)
                                               for i in df['name']]],
                line_color='darkslategray',
                align='center',
                font_size=[18, 14],
                height=30)
        )
        ])

        final = [distance, Markup(fig.to_html(
            full_html=False)), num_songs, len(df['playlist'])]
        return final

    def _get_library_totals(self):
        ALL_SONGS_DF = self._all_songs_df
        UNIQUE_SONGS_DF = self._unique_songs_df
        
        # Get liked songs count
        liked_songs_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] == 'Liked Songs']
        liked_songs_count = len(liked_songs_df.index)

        overall_data = [len(ALL_SONGS_DF.index), len(UNIQUE_SONGS_DF),
                        len(ALL_SONGS_DF['playlist'].unique()), len(
                            UNIQUE_SONGS_DF['artist'].unique()),
                        len(UNIQUE_SONGS_DF['album'].unique()), liked_songs_count]

        return overall_data


# Overall Stats = About Me Page ----------------------------------------------------------------------------------------------

class AboutPage():
    def __init__(self, path, all_songs_df, unique_songs_df, artists):
        self._path = path
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        self._artists = artists

        # Precompute page fragments in-memory
        self._followed_artists = self._graph_top_genres_by_followed_artists()
        self._overall_songs = self._graph_top_songs_by_num_playlists()
        self._overall_artists = self._graph_top_artists_and_albums_by_num_playlists()
        self._overall_albums = self._graph_top_artists_and_albums_by_num_playlists(albums=True)

    def load_followed_artists(self):
        return self._followed_artists

    def load_top_songs(self):
        return self._overall_songs

    def load_top_artists(self):
        return self._overall_artists

    def load_top_albums(self):
        return self._overall_albums

    def _graph_top_genres_by_followed_artists(self):
        d = defaultdict(int)
        d2 = defaultdict(list)

        for a in self._artists:
            for g in a['genres']:
                d[g] += 1
                d2[g].append(a['name'])

        top_n = 10
        data = {i[0]: i[1] for i in sorted(
            d.items(), key=lambda x: x[1], reverse=True)[:top_n][::-1]}

        series = pd.Series(data)
        final = _h_bar(series, title='Most Common Genres by Followed Artists', xaxis='# of Followed Artists',
                       hovertext=[', '.join(d2[i]) for i in data.keys()], yaxis='Genre', long_names=True)
        return final
    

    def _graph_top_songs_by_num_playlists(self, buttons=5):
        
        df = self._unique_songs_df.sort_values(by='num_playlists', ascending=False)
        df = df.head(buttons*10)

        graphs = []
        for i in range(buttons):
            first = i*10
            last = (i+1)*10
            subset = df.iloc[first:last].sort_values(by='num_playlists')
            subset['playlists'] = ['<br>'.join(i) for i in subset['playlist']]
            subset = subset[['name', 'num_playlists', 'playlists']]

            title = 'Top ' + str(first+1)  + '-' + str(last) + ' Most Common Songs Across ' + \
                str(len(self._all_songs_df['playlist'].unique())) + ' Playlists'
            
            # Create series for _h_bar function
            series = pd.Series(dict(zip(subset['name'], subset['num_playlists'])))
            bar_trace = _h_bar(series, title=title, xaxis='Number of Playlists', yaxis='Song', 
                              hovertext=subset['playlists'], to_html=False)
            
            # Convert Bar trace to Figure
            bar = go.Figure(data=[bar_trace])
            bar.update_layout(title=title, xaxis_title='Number of Playlists', yaxis_title='Song')
            graphs.append(bar)

        labels = []
        for i in range(buttons):
            start = i * 10 + 1
            end = (i + 1) * 10
            labels.append(f"{start}-{end}")

        final = _make_single_subplot(labels, graphs,
                                   'Number of Playlists Song Is In',
                                   'Song',
                                   'Top ' + str(buttons*10) + ' Most Common Songs Across ' + \
                str(len(self._all_songs_df['playlist'].unique())) + ' Playlists')

        return Markup(final.to_html(full_html=False))


    def _graph_top_artists_and_albums_by_num_playlists(self, buttons=5, albums=False):
        ALL_SONGS_DF = self._all_songs_df
        xaxis = 'Number of Album Songs in All Playlists' if albums else 'Number of Artist Songs in All Playlists'
        yaxis = 'Album' if albums else 'Artist'

        dicty = dict()
        if albums:
            df = ALL_SONGS_DF['album'].value_counts().reset_index()
            df.columns = ['Album', 'Count'] 
            df['Artist'] = [ALL_SONGS_DF[ALL_SONGS_DF['album'] == i].iloc[0]['artist'] for i in df['Album']]
        else:
            artists, counts = [], []
            for a in {j for i in self._unique_songs_df['artist'].unique() for j in i.split(', ')}:
                mask = ALL_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))
                a_df = ALL_SONGS_DF[mask]
                artists.append(a)
                counts.append(len(a_df.index))
            df = pd.DataFrame({'Artist':artists, 'Count':counts})
        df = df.sort_values(by='Count', ascending=False).iloc[:buttons*10]
        
        # Validate the main DataFrame before processing
        if df.empty:
            print(f"Warning: No data available for {yaxis} visualization")
            return Markup("<p>No data available for this visualization.</p>")
        
        # Ensure Count column is numeric
        df['Count'] = pd.to_numeric(df['Count'], errors='coerce')
        df = df.dropna(subset=['Count'])
        
        if df.empty:
            print(f"Warning: No valid numeric data for {yaxis} visualization")
            return Markup("<p>No valid data available for this visualization.</p>")
        
        # df['link'] = [REDIRECT_URI + yaxis.lower() + 's/' + i for i in df[yaxis.lower()]]

        graphs = []
        for i in range(buttons):
            first = i*10
            last = (i+1)*10
            subset = df.iloc[first:last].sort_values(by='Count')

            # Skip if subset is empty or has no valid data
            if subset.empty or len(subset) == 0:
                continue
                
            # Ensure Count column has valid numeric values
            subset = subset.dropna(subset=['Count'])
            subset['Count'] = pd.to_numeric(subset['Count'], errors='coerce')
            subset = subset.dropna(subset=['Count'])
            
            if subset.empty:
                continue

            title = 'Top ' + str(first+1)  + '-' + str(last) + f' Most Common {yaxis}s Across ' + \
                str(len(self._all_songs_df['playlist'].unique())) + ' Playlists'
            
            try:
                if albums:
                    bar = px.bar(subset, x='Count', y=yaxis, orientation='h', title=title, text='Count', hover_data='Artist')
                else:
                    bar = px.bar(subset, x='Count', y=yaxis, orientation='h', title=title, text='Count')
            except Exception as e:
                print(f"Error creating bar chart for {yaxis}: {e}")
                continue
            ## Prob doesn't work with subplots
            # for r in subset.iterrows():
            #     bar.add_annotation(
            #         {
            #             "y": r[1][yaxis.lower()],
            #             "x": r[1]["Count"],
            #             "text": f"""<a href="{r[1]["link"]}" target="_blank">{r[1][yaxis.lower()]}</a>""",
            #         }
            #     )
            graphs.append(bar)

        # Check if we have any valid graphs
        if not graphs:
            print(f"Warning: No valid graphs created for {yaxis} visualization")
            return Markup("<p>No valid data available for this visualization.</p>")

        labels = []
        for i in range(len(graphs)):  # Use actual number of graphs created
            start = i * 10 + 1
            end = (i + 1) * 10
            labels.append(f"{start}-{end}")

        final = _make_single_subplot(labels, graphs, xaxis, yaxis, 
                'Top ' + str(len(graphs)*10) + f' Most Common {yaxis}s Across ' + \
                str(len(ALL_SONGS_DF['playlist'].unique())) + ' Playlists')

        if albums:
            return Markup(final.to_html(full_html=False))
        else:
            return Markup(final.to_html(full_html=False))


# Analyze Multiple Playlists Page ----------------------------------------------------------------------------------------------

class AnalyzePlaylistsPage():
    def __init__(self, playlists, all_songs_df, unique_songs_df, global_playlist_averages=None):
        self._playlists = playlists
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)
        self._global_playlist_averages = global_playlist_averages

    def _get_intersection_of_playlists(self, color):
        mask = self._unique_songs_df['playlist'].apply(
            lambda x: set(self._playlists).issubset(set(x)))
        df = self._unique_songs_df[mask]

        if len(df.index) < 1:
            return [None, 0]
        else:
            x_data = FEATURE_COLS
            y_data = []

            df['popularity'] = df['popularity']/100
            y_data = [df[col] for col in FEATURE_COLS if col]

            title = 'Audio Features of Songs Which Intersect Playlists'
            text = [n + '<br>' + a for n, a in zip(df['name'], df['artist'])]
            xaxis = 'Song Feature'
            yaxis = 'Level (Low -> High)'
            return [_boxplot(x_data, y_data, text, title, xaxis, yaxis, to_html=False, name='Intersection', color=color), len(df.index)]

    def graph_playlist_timelines(self):
        """Create timeline graphs showing when songs were added to playlists vs when they were added to Liked Songs"""
        if not self._playlists or len(self._playlists) == 0:
            return None
        
        # Get global playlist averages
        global_avg_liked_songs_per_playlist = 0
        global_avg_total_songs_per_playlist = 0
        if self._global_playlist_averages:
            global_avg_liked_songs_per_playlist, global_avg_total_songs_per_playlist = self._global_playlist_averages
        
        # Create the figure
        fig = go.Figure()
        
        # Track all traces for visibility management
        all_traces = []
        
        # Add traces for each playlist
        for i, playlist in enumerate(self._playlists):
            color = COLORS[i % len(COLORS)]
            
            # Filter data for this playlist (excluding Liked Songs playlist)
            playlist_df = self._all_songs_df[self._all_songs_df['playlist'] == playlist]
            
            if len(playlist_df.index) > 0:
                # Group by date and count songs, create hovertext for added songs
                timeline_data = playlist_df.groupby('date_added').agg({
                    'name': lambda x: list(x),
                    'artist': lambda x: list(x)
                }).reset_index()
                
                # Create hovertext with song names separated by <br>
                hovertext = []
                for _, row in timeline_data.iterrows():
                    song_names = [f"{name} by {artist}" for name, artist in zip(row['name'], row['artist'])]
                    hovertext.append('<br>'.join(song_names))
                
                # Add Added Songs Line trace (visible by default)
                line_trace = go.Scatter(
                    x=timeline_data['date_added'],
                    y=timeline_data['name'].apply(len),
                    mode='lines+markers',
                    name=playlist,
                    line=dict(color=color),
                    marker=dict(size=6),
                    hovertext=hovertext,
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Date: %{x}<br>' +
                                'Songs Added: %{y}<extra></extra>',
                    visible=True
                )
                fig.add_trace(line_trace)
                all_traces.append(line_trace)
                
                # Add Added Songs Continuous trace (cumulative)
                continuous_trace = go.Scatter(
                    x=timeline_data['date_added'],
                    y=timeline_data['name'].apply(len).cumsum(),
                    mode='lines+markers',
                    name=f"{playlist}",
                    line=dict(color=color, dash='dash'),
                    marker=dict(size=6),
                    hovertext=hovertext,
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Date: %{x}<br>' +
                                'Cumulative Songs Added: %{y}<extra></extra>',
                    visible=False
                )
                fig.add_trace(continuous_trace)
                all_traces.append(continuous_trace)
                
                # Add Added Songs Bar trace
                bar_trace = go.Bar(
                    x=timeline_data['date_added'],
                    y=timeline_data['name'].apply(len),
                    name=f"{playlist}",
                    marker_color=color,
                    hovertext=hovertext,
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Date: %{x}<br>' +
                                'Songs Added: %{y}<extra></extra>',
                    visible=False
                )
                fig.add_trace(bar_trace)
                all_traces.append(bar_trace)
                
                # Add Liked Songs traces for this playlist
                # Find songs that exist in both the playlist and Liked Songs
                liked_songs_df = self._all_songs_df[self._all_songs_df['playlist'] == 'Liked Songs']
                if len(liked_songs_df.index) > 0:
                    # Merge on name and artist to find matches
                    playlist_liked_songs = pd.merge(playlist_df[['name', 'artist']], 
                                                   liked_songs_df[['name', 'artist', 'date_added']], 
                                                   on=['name', 'artist'])
                    
                    if len(playlist_liked_songs.index) > 0:
                        # Group by date when songs were added to Liked Songs (not when added to playlist)
                        liked_timeline_data = playlist_liked_songs.groupby('date_added').agg({
                            'name': lambda x: list(x),
                            'artist': lambda x: list(x)
                        }).reset_index()
                        
                        # Create hovertext with song names separated by <br>
                        liked_hovertext = []
                        for _, row in liked_timeline_data.iterrows():
                            song_names = [f"{name} by {artist}" for name, artist in zip(row['name'], row['artist'])]
                            liked_hovertext.append('<br>'.join(song_names))
                        
                        # Add Liked Songs Line trace (not visible by default)
                        liked_line_trace = go.Scatter(
                            x=liked_timeline_data['date_added'],
                            y=liked_timeline_data['name'].apply(len),
                            mode='lines+markers',
                            name=f"{playlist} Liked Songs",
                            line=dict(color=color),
                            marker=dict(size=6, symbol='diamond'),
                            hovertext=liked_hovertext,
                            hovertemplate='<b>%{hovertext}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Liked Songs Added: %{y}<extra></extra>',
                            visible=False
                        )
                        fig.add_trace(liked_line_trace)
                        all_traces.append(liked_line_trace)
                        
                        # Add Liked Songs Continuous trace (cumulative)
                        liked_continuous_trace = go.Scatter(
                            x=liked_timeline_data['date_added'],
                            y=liked_timeline_data['name'].apply(len).cumsum(),
                            mode='lines+markers',
                            name=f"{playlist} Liked Songs",
                            line=dict(color=color, dash='dash'),
                            marker=dict(size=6, symbol='diamond'),
                            hovertext=liked_hovertext,
                            hovertemplate='<b>%{hovertext}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Cumulative Liked Songs: %{y}<extra></extra>',
                            visible=False
                        )
                        fig.add_trace(liked_continuous_trace)
                        all_traces.append(liked_continuous_trace)
                        
                        # Add Liked Songs Bar trace
                        liked_bar_trace = go.Bar(
                            x=liked_timeline_data['date_added'],
                            y=liked_timeline_data['name'].apply(len),
                            name=f"{playlist} Liked Songs",
                            marker_color=color,
                            hovertext=liked_hovertext,
                            hovertemplate='<b>%{hovertext}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Liked Songs Added: %{y}<extra></extra>',
                            visible=False
                        )
                        fig.add_trace(liked_bar_trace)
                        all_traces.append(liked_bar_trace)
                    else:
                        # No liked songs for this playlist, add empty traces to maintain structure
                        for _ in range(3):  # Line, Continuous, Bar
                            empty_trace = go.Scatter(
                                x=[], y=[],
                                mode='lines+markers',
                                name=f"{playlist} Liked Songs",
                                line=dict(color=color),
                                visible=False
                            )
                            fig.add_trace(empty_trace)
                            all_traces.append(empty_trace)
                else:
                    # No liked songs playlist, add empty traces to maintain structure
                    for _ in range(3):  # Line, Continuous, Bar
                        empty_trace = go.Scatter(
                            x=[], y=[],
                            mode='lines+markers',
                            name=f"{playlist} Liked Songs",
                            line=dict(color=color),
                            visible=False
                        )
                        fig.add_trace(empty_trace)
                        all_traces.append(empty_trace)
        
        # Add horizontal average lines as traces (only visible when Continuous button is selected)
        if global_avg_total_songs_per_playlist > 0:
            # Get x range from the data
            if len(fig.data) > 0 and len(fig.data[0].x) > 0:
                x_min = min([trace.x[0] for trace in fig.data if len(trace.x) > 0])
                x_max = max([trace.x[-1] for trace in fig.data if len(trace.x) > 0])
            else:
                x_min = '2020-01-01'
                x_max = '2024-12-31'
            
            # Create a trace for the main average line
            avg_total_line = go.Scatter(
                x=[x_min, x_max],
                y=[global_avg_total_songs_per_playlist, global_avg_total_songs_per_playlist],
                mode='lines',
                name=f"Your Avg # of Songs Added per Playlist = {global_avg_total_songs_per_playlist:.2f}",
                line=dict(color=MAIN_AVG_COLOR, dash='dash', width=2),
                visible=False
            )
            fig.add_trace(avg_total_line)
            all_traces.append(avg_total_line)

        if global_avg_liked_songs_per_playlist > 0:
            # Get x range from the data
            if len(fig.data) > 0 and len(fig.data[0].x) > 0:
                x_min = min([trace.x[0] for trace in fig.data if len(trace.x) > 0])
                x_max = max([trace.x[-1] for trace in fig.data if len(trace.x) > 0])
            else:
                x_min = '2020-01-01'
                x_max = '2024-12-31'
            
            # Create a trace for the liked songs average line
            avg_liked_line = go.Scatter(
                x=[x_min, x_max],
                y=[global_avg_liked_songs_per_playlist, global_avg_liked_songs_per_playlist],
                mode='lines',
                name=f"Your Avg # of Liked Songs per Playlist = {global_avg_liked_songs_per_playlist:.2f}",
                line=dict(color=LIKED_AVG_COLOR, dash='dash', width=2),
                visible=False
            )
            fig.add_trace(avg_liked_line)
            all_traces.append(avg_liked_line)

        # Create buttons for Line, Continuous, Bar views
        buttons = []
        
        # Create visibility arrays based on actual figure data
        # Count how many average lines we have
        num_avg_lines = 0
        if global_avg_total_songs_per_playlist > 0:
            num_avg_lines += 1
        if global_avg_liked_songs_per_playlist > 0:
            num_avg_lines += 1
        
        # Calculate total playlist traces (excluding average lines)
        total_playlist_traces = len(fig.data) - num_avg_lines
        
        # Line button (shows only playlist line traces)
        line_visibility = []
        for i in range(total_playlist_traces):
            if i % 6 == 0:  # First trace for each playlist (Line)
                line_visibility.append(True)
            else:
                line_visibility.append(False)
        # Add average lines visibility (False for Line view)
        for _ in range(num_avg_lines):
            line_visibility.append(False)
        buttons.append(dict(
            method='update',
            label='Line',
            args=[{'visible': line_visibility}]
        ))

        # Continuous button (shows continuous traces and horizontal lines)
        continuous_visibility = []
        for i in range(total_playlist_traces):
            if i % 6 == 1 or i % 6 == 4:  # Continuous traces (index 1 and 4 for each playlist)
                continuous_visibility.append(True)
            else:
                continuous_visibility.append(False)
        # Add horizontal lines visibility for continuous (True for average lines)
        for _ in range(num_avg_lines):
            continuous_visibility.append(True)
        buttons.append(dict(
            method='update',
            label='Continuous',
            args=[{'visible': continuous_visibility}]
        ))

        # Bar button (shows only playlist bar traces)
        bar_visibility = []
        for i in range(total_playlist_traces):
            if i % 6 == 2:  # Third trace for each playlist (Bar)
                bar_visibility.append(True)
            else:
                bar_visibility.append(False)
        # Add average lines visibility (False for Bar view)
        for _ in range(num_avg_lines):
            bar_visibility.append(False)
        buttons.append(dict(
            method='update',
            label='Bar',
            args=[{'visible': bar_visibility}]
        ))

        # Update layout
        fig.update_layout(
            title_text="Timeline of When Songs Were Added to Playlists",
            xaxis_title="Date",
            yaxis_title="Number of Songs Added",
            height=600,
            showlegend=True,
            updatemenus=[dict(
                type="buttons",
                direction="right",
                y=1.3,
                x=0.6,
                buttons=buttons
            )]
        )

        # Update x-axis to show dates properly
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        return Markup(fig.to_html(full_html=False))

    def graph_genres_by_playlists(self):
        '''Return a stacked horizontal bar graph, with each bar being a genre, 
        and each stack being % of artist genre in a different playlist'''

        # Create dataframe - each row a genre, each column a playlist
        percents = defaultdict(list)
        data = []
        for p in self._playlists:
            df = self._all_songs_df[self._all_songs_df['playlist'] == p]
            playlist_genres = {y for i in df['genres'] for x in i for y in x}
            for g in playlist_genres:
                mask = df['genres'].apply(
                    lambda x: g in {z for y in x for z in y})
                percents[g].append(len(df[mask])/len(df)*100)

        for i in percents:
            percents[i].append(sum(percents[i]))

        df = pd.DataFrame.from_dict(
            percents, orient='index')
        df.columns = self._playlists + ['sum']
        df = df.sort_values(by='sum', ascending=False).head(10)

        for p in self._playlists:
            series = pd.Series(
                dict(zip(df.index, df[p])))
            go_bar = _h_bar(series, name=p,
                            percents=True, to_html=False)
            data.append(go_bar)

        fig = go.Figure(data=data)
        fig.update_layout(barmode='stack', yaxis={
            'categoryorder': 'total ascending'}, yaxis_title='Genre', xaxis_title='% of Songs In Playlist', title='Most Common Genres Among Playlists')
        return Markup(fig.to_html(full_html=False))
    

    def _get_playlist_boxplot(self, playlist, color):
        df = self._all_songs_df[self._all_songs_df['playlist'] == playlist]

        x_data = FEATURE_COLS
        y_data = []

        df['popularity'] = df['popularity']/100
        y_data = [df[col] for col in FEATURE_COLS if col]

        title = 'Audio Features of ' + playlist + ' Songs'
        text = [n + '<br>' + a for n, a in zip(df['name'], df['artist'])]
        xaxis = 'Song Feature'
        yaxis = 'Level (Low -> High)'
        return _boxplot(x_data, y_data, text, title, xaxis, yaxis, to_html=False, name=playlist, color=color)


    def graph_playlists_boxplots(self):
        '''See Boxplots of Intersection vs Sad vs Slow vs No Lyrics'''
        fig = subplots.make_subplots(rows=1, cols=1)

        # Features have same colors for Same Playlist
        colorInd = 0
        intersection = self._get_intersection_of_playlists(COLORS[colorInd])
        if intersection[0] != None:
            fig.add_trace(intersection[0])
            colorInd = 1

        for i in self._playlists:
            playlist_boxplot = self._get_playlist_boxplot(i, COLORS[colorInd])
            fig.add_trace(playlist_boxplot, 1, 1)
            colorInd += 1

        fig.update_layout(title_text="Intersection and Playlists\' Audio Features",
                  xaxis=dict(title="Audio Features"),
                  yaxis=dict(title="Level (Low-High)"),
                  boxmode='group',
                  showlegend=True)

        return [Markup(fig.to_html(full_html=False)), intersection[1]]
    

    def graph_artists_by_playlists(self):
        '''Return a stacked horizontal bar graph
        full bar = sum of songs from unique artist from given playlists
        each color = # artist songs per playlist'''

        # Create dataframe - each row an artist, each column a playlist
        counts = defaultdict(list)
        data = []
        songs = {}
        num = 0
        for p in self._playlists:
            df = self._all_songs_df[self._all_songs_df['playlist'] == p]
            unique_artists = {j for i in df['artist'] for j in i.split(', ')}
            for a in unique_artists:
                adf = df[df['artist'].apply(lambda x: a in x.split(', '))]
                if len(counts[a]) < num:
                    counts[a].append(0)
                counts[a].append(len(adf))
                songs[(p,a)] = '<br>'.join(adf['name'][:MAX_HOVER_ROWS])
            for i in counts:
                if len(counts[i]) < num:
                    counts[i].append(0)
            num += 1

        for i in counts:
            counts[i].append(sum(counts[i]))

        df = pd.DataFrame.from_dict(counts, orient='index')
        df.columns = self._playlists + ['sum']
        df = df.sort_values(by='sum', ascending=False).head(10)

        for p in self._playlists:
            hovertext = [songs[(p,a)] if (p,a) in songs else '' for a in df.index]
            series = pd.Series(dict(zip(df.index, df[p])))
            go_bar = _h_bar(series, name=p, hovertext=hovertext, to_html=False)
            data.append(go_bar)

        fig = go.Figure(data=data)
        fig.update_layout(barmode='stack', yaxis={
            'categoryorder': 'total ascending'}, yaxis_title='Artist', xaxis_title='# of Songs In Playlist', title='Most Common Artists Among Playlists')
        return Markup(fig.to_html(full_html=False))


# Analyze Single Playlist Page ----------------------------------------------------------------------------------------------

class AnalyzePlaylistPage(OptimizedPageBase):
    def __init__(self, playlist, all_songs_df, unique_songs_df, global_playlist_averages=None):
        super().__init__(None, all_songs_df, unique_songs_df)
        self._playlist = playlist
        self._global_playlist_averages = global_playlist_averages

        self._playlist_df = self._all_songs_df[self._all_songs_df['playlist']
                                               == self._playlist]

    def graph_count_timeline(self):
        """Create a custom timeline for the specific playlist with liked songs as a separate trace"""
        # Check if we're analyzing the Liked Songs playlist itself
        is_liked_songs_playlist = (self._playlist == 'Liked Songs')
        
        # Create timeline for the specific playlist
        playlist_df = self._all_songs_df[self._all_songs_df['playlist'] == self._playlist]
        
        # Only get liked songs intersection if we're NOT analyzing the Liked Songs playlist
        if not is_liked_songs_playlist:
            # Get liked songs that are within this specific playlist
            # We need to find songs that exist in both the playlist and Liked Songs
            playlist_songs = playlist_df[['name', 'artist', 'date_added']].copy()
            liked_songs_df = self._all_songs_df[self._all_songs_df['playlist'] == 'Liked Songs'][['name', 'artist', 'date_added']].copy()
            
            # Find intersection: songs that are in both the playlist and Liked Songs
            # Merge on name and artist to find matches
            playlist_liked_songs = pd.merge(playlist_songs, liked_songs_df, 
                                           on=['name', 'artist'], 
                                           suffixes=('_playlist', '_liked'))
        else:
            # If analyzing Liked Songs playlist, create empty intersection
            playlist_liked_songs = pd.DataFrame(columns=['name', 'artist', 'date_added_playlist'])
        
        # Group by date for playlist timeline
        playlist_timeline = playlist_df.groupby('date_added').size().reset_index(name='count')
        playlist_timeline = playlist_timeline.sort_values('date_added')
        
        # Create continuous (cumulative) timeline
        playlist_continuous_timeline = playlist_timeline.copy()
        playlist_continuous_timeline['count'] = playlist_continuous_timeline['count'].cumsum()
        
        # Group by date for liked songs within this playlist timeline
        if not is_liked_songs_playlist and len(playlist_liked_songs.index) > 0:
            liked_timeline = playlist_liked_songs.groupby('date_added_liked').size().reset_index(name='count')
            liked_timeline = liked_timeline.rename(columns={'date_added_liked': 'date_added'})
            liked_timeline = liked_timeline.sort_values('date_added')
        else:
            # Create empty timeline if no liked songs in this playlist or if analyzing Liked Songs playlist
            liked_timeline = pd.DataFrame(columns=['date_added', 'count'])
        
        # Create continuous (cumulative) timeline for liked songs
        liked_continuous_timeline = liked_timeline.copy()
        if len(liked_continuous_timeline.index) > 0:
            liked_continuous_timeline['count'] = liked_continuous_timeline['count'].cumsum()
        
        # Create hovertext for playlist songs
        playlist_hovertext = []
        for date in playlist_timeline['date_added']:
            songs_on_date = playlist_df[playlist_df['date_added'] == date]
            song_names = songs_on_date['name'].tolist()
            playlist_hovertext.append('<br>'.join(song_names))
        
        # Create hovertext for liked songs within this playlist
        liked_hovertext = []
        if not is_liked_songs_playlist:
            for date in liked_timeline['date_added']:
                songs_on_date = playlist_liked_songs[playlist_liked_songs['date_added_liked'] == date]
                song_names = songs_on_date['name'].tolist()
                liked_hovertext.append('<br>'.join(song_names))
        
        # Create the figure
        fig = go.Figure()
        
        # Add playlist line trace
        playlist_line_trace = go.Scatter(
            x=playlist_timeline['date_added'],
            y=playlist_timeline['count'],
            mode='lines+markers',
            name=f'Songs Added to {self._playlist}',
            line=dict(color=MAIN_TIMELINE_COLOR, width=2),
            marker=dict(size=6),
            visible=True,
            hovertext=playlist_hovertext,
            hovertemplate='<b>%{x}</b><br>%{y} songs<br>%{hovertext}<extra></extra>'
        )
        fig.add_trace(playlist_line_trace)
        
        # Add playlist continuous trace
        playlist_continuous_trace = go.Scatter(
            x=playlist_continuous_timeline['date_added'],
            y=playlist_continuous_timeline['count'],
            mode='lines+markers',
            name=f'Total Songs in {self._playlist}',
            line=dict(color=MAIN_TIMELINE_COLOR, width=2, dash='dash'),
            marker=dict(size=6),
            visible=False,
            hovertext=playlist_hovertext,
            hovertemplate='<b>%{x}</b><br>%{y} total songs<br>%{hovertext}<extra></extra>'
        )
        fig.add_trace(playlist_continuous_trace)
        
        # Add playlist bar trace
        playlist_bar_trace = go.Bar(
            x=playlist_timeline['date_added'],
            y=playlist_timeline['count'],
            name=f'Songs Added to {self._playlist}',
            marker_color=MAIN_TIMELINE_COLOR,
            visible=False,
            hovertext=playlist_hovertext,
            hovertemplate='<b>%{x}</b><br>%{y} songs<br>%{hovertext}<extra></extra>'
        )
        fig.add_trace(playlist_bar_trace)
        
        # Add liked songs traces only if NOT analyzing the Liked Songs playlist
        if not is_liked_songs_playlist:
            # Add liked songs line trace
            liked_line_trace = go.Scatter(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'],
                mode='lines+markers',
                name=f'Liked Songs in {self._playlist}',
                line=dict(color=LIKED_TIMELINE_COLOR, width=2),
                marker=dict(size=6),
                visible=True,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{x}</b><br>%{y} songs<br>%{hovertext}<extra></extra>'
            )
            fig.add_trace(liked_line_trace)
            
            # Add liked songs continuous trace
            liked_continuous_trace = go.Scatter(
                x=liked_continuous_timeline['date_added'],
                y=liked_continuous_timeline['count'],
                mode='lines+markers',
                name=f'Total Liked Songs in {self._playlist}',
                line=dict(color=LIKED_TIMELINE_COLOR, width=2, dash='dash'),
                marker=dict(size=6),
                visible=False,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{x}</b><br>%{y} total songs<br>%{hovertext}<extra></extra>'
            )
            fig.add_trace(liked_continuous_trace)
            
            # Add liked songs bar trace
            liked_bar_trace = go.Bar(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'],
                name=f'Liked Songs in {self._playlist}',
                marker_color=LIKED_TIMELINE_COLOR,
                visible=False,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{x}</b><br>%{y} songs<br>%{hovertext}<extra></extra>'
            )
            fig.add_trace(liked_bar_trace)
        
        # Add vertical dashed lines for each year's anniversary
        if len(playlist_df.index) > 0:
            first_date = playlist_df['date_added'].min()
            # Convert to datetime if it's a string
            if isinstance(first_date, str):
                first_date = datetime.datetime.fromisoformat(first_date.replace('Z', '+00:00'))
            
            first_year = first_date.year
            current_year = datetime.datetime.now().year
            
            # Get month and day from today's date (current date)
            today = datetime.datetime.now()
            today_month = today.month
            today_day = today.day
            
            for year in range(first_year, current_year + 1):
                try:
                    anniversary_date = datetime.datetime(year, today_month, today_day)
                    
                    fig.add_vline(
                        x=anniversary_date,
                        line_width=3,
                        line_dash="dash",
                        line_color="green"
                    )
                except ValueError:
                    # Skip invalid dates (e.g., February 29 in non-leap years)
                    continue
        
        # Add horizontal lines for global averages (only visible in Continuous view)
        if self._global_playlist_averages:
            avg_liked_songs, avg_total_songs = self._global_playlist_averages
        else:
            avg_liked_songs = 0
            avg_total_songs = 0
        
        # Add horizontal line for average total songs per playlist
        if avg_total_songs > 0:
            # Get x range from the data
            if len(playlist_df.index) > 0:
                x_min = playlist_df['date_added'].min()
                x_max = playlist_df['date_added'].max()
                # Convert to datetime if they're strings
                if isinstance(x_min, str):
                    x_min = datetime.datetime.fromisoformat(x_min.replace('Z', '+00:00'))
                if isinstance(x_max, str):
                    x_max = datetime.datetime.fromisoformat(x_max.replace('Z', '+00:00'))
            else:
                x_min = datetime.datetime(2020, 1, 1)
                x_max = datetime.datetime(2024, 12, 31)
            
            avg_total_line = go.Scatter(
                x=[x_min, x_max],  # Span the full x range
                y=[avg_total_songs, avg_total_songs],
                mode='lines',
                name=f"Your Avg # of Songs Added per Playlist = {avg_total_songs:.2f}",
                line=dict(color=MAIN_AVG_COLOR, dash='dash', width=2),
                visible=False,
                showlegend=True,
                hoverinfo='skip'
            )
            fig.add_trace(avg_total_line)
        
        # Add horizontal line for average liked songs per playlist
        if avg_liked_songs > 0:
            # Get x range from the data
            if len(playlist_df.index) > 0:
                x_min = playlist_df['date_added'].min()
                x_max = playlist_df['date_added'].max()
                # Convert to datetime if they're strings
                if isinstance(x_min, str):
                    x_min = datetime.datetime.fromisoformat(x_min.replace('Z', '+00:00'))
                if isinstance(x_max, str):
                    x_max = datetime.datetime.fromisoformat(x_max.replace('Z', '+00:00'))
            else:
                x_min = datetime.datetime(2020, 1, 1)
                x_max = datetime.datetime(2024, 12, 31)
            
            avg_liked_line = go.Scatter(
                x=[x_min, x_max],  # Span the full x range
                y=[avg_liked_songs, avg_liked_songs],
                mode='lines',
                name=f"Your Avg # of Liked Songs per Playlist = {avg_liked_songs:.2f}",
                line=dict(color=LIKED_AVG_COLOR, dash='dash', width=2),
                visible=False,
                showlegend=True,
                hoverinfo='skip'
            )
            fig.add_trace(avg_liked_line)
        
        # Create buttons for Line, Continuous, and Bar views
        # Calculate visibility arrays based on whether horizontal lines exist
        has_avg_total = avg_total_songs > 0
        has_avg_liked = avg_liked_songs > 0
        
        # Determine which traces exist
        has_liked_traces = not is_liked_songs_playlist and len(liked_timeline.index) > 0
        
        # Base visibility: [playlist_line, playlist_continuous, playlist_bar, liked_line, liked_continuous, liked_bar, avg_total_line, avg_liked_line]
        if has_liked_traces:
            # All traces exist (including liked songs)
            line_visible = [True, False, False, True, False, False, False, False]
            continuous_visible = [False, True, False, False, True, False, has_avg_total, has_avg_liked]
            bar_visible = [False, False, True, False, False, True, False, False]
        else:
            # Only playlist traces exist (no liked songs)
            line_visible = [True, False, False, False, False, False, False, False]
            continuous_visible = [False, True, False, False, False, False, has_avg_total, has_avg_liked]
            bar_visible = [False, False, True, False, False, False, False, False]
        
        updatemenus = [
            dict(
                type='buttons',
                direction='right',
                y=1.3,
                x=0.6,
                buttons=[
                    dict(
                        label="Line",
                        method="update",
                        args=[{"visible": line_visible}]
                    ),
                    dict(
                        label="Continuous",
                        method="update",
                        args=[{"visible": continuous_visible}]
                    ),
                    dict(
                        label="Bar",
                        method="update",
                        args=[{"visible": bar_visible}]
                    )
                ]
            )
        ]
        
        # Update layout
        fig.update_layout(
            title=f'Timeline of Adding {self._playlist} Songs to Playlists',
            xaxis_title='Date',
            yaxis_title='Daily Added Songs',
            barmode='stack',
            updatemenus=updatemenus
        )
        
        # Update x-axis to include range slider and selector
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Update y-axis
        fig.update_yaxes(title_text='Daily Added Songs')
        
        return Markup(fig.to_html(full_html=False))

    def graph_playlist_genres(self):
        df = self._playlist_df
        
        # Check if we have any data
        if len(df.index) == 0:
            # Return empty graph if no data
            fig = go.Figure()
            fig.update_layout(
                title='Most Common Genres For Playlist: ' + self._playlist,
                xaxis_title='% of Songs',
                yaxis_title='Artist Genre',
                annotations=[{
                    'text': 'No genre data available for this playlist',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return Markup(fig.to_html(full_html=False))
        
        genres = defaultdict(int)
        genre_songs = defaultdict(list)  # Store song names for each genre
        
        for idx, row in df.iterrows():
            song_name = row['name']
            i = row['genres']
            
            if isinstance(i, list) and len(i) > 0:
                # Check if it's a list of lists or list of strings
                if isinstance(i[0], list):
                    # It's a list of lists, flatten it
                    unique_genres = {y for x in i for y in x}
                else:
                    # It's a list of strings, use as is
                    unique_genres = set(i)
            else:
                # Handle other cases
                unique_genres = set()
            
            for j in unique_genres:
                genres[j] += 1
                genre_songs[j].append(song_name)
        
        total = len(df.index)
        relative = {k: v/total*100 for k,
                    v in sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]}
        genre_counts = {k: str(v) for k, v in sorted(
            genres.items(), key=lambda x: x[1], reverse=True)}

        # Create hovertext with song names
        hovertext = []
        for genre in relative.keys():
            count = genre_counts[genre]
            songs = genre_songs[genre]
            # Sample up to MAX_HOVER_ROWS songs
            sample_songs = songs[:MAX_HOVER_ROWS] if len(songs) > MAX_HOVER_ROWS else songs
            song_list = '<br>'.join(sample_songs)
            hovertext.append(f'{count} Songs<br><br>{song_list}')

        series = pd.Series(relative)
        title = 'Most Common Genres For Playlist: ' + self._playlist
        
        
        return _h_bar(series, title=title, xaxis='% of Songs', yaxis='Artist Genre', percents=True,
                      hovertext=hovertext)

    def graph_top_artists(self, top_n=10):
        df = self._playlist_df.copy()
        
        # Check if we have any data
        if len(df.index) == 0:
            # Return empty graph if no data
            fig = go.Figure()
            fig.update_layout(
                title='Most Common Artists For Playlist: ' + self._playlist,
                xaxis_title='Number of Songs',
                yaxis_title='Artist',
                annotations=[{
                    'text': 'No artist data available for this playlist',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return Markup(fig.to_html(full_html=False))

        df['artist'] = df['artist'].str.split(',')
        df_exploded = df.explode('artist')
        grouped = df_exploded.groupby('artist')['name'].agg(list)

        sorted_dict = dict(sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:top_n])
        series = pd.Series({i:len(j) for i,j in sorted_dict.items()})

        title = 'Most Common Artists For Playlist: ' + self._playlist
        return _h_bar(series, title=title, yaxis='Artist', xaxis='Number of Songs', long_names=True,
                       hovertext=['<br>'.join(_shorten_names(i)) for i in sorted_dict.values()])

    def graph_top_albums(self, top_n=10):
        df = self._playlist_df
        
        # Check if we have any data
        if len(df.index) == 0:
            # Return empty graph if no data
            fig = go.Figure()
            fig.update_layout(
                title='Most Common Albums For Playlist: ' + self._playlist,
                xaxis_title='Number of Songs',
                yaxis_title='Album',
                annotations=[{
                    'text': 'No album data available for this playlist',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return Markup(fig.to_html(full_html=False))

        grouped = df.groupby('album')['name'].agg(list)
        sorted_dict = dict(sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:top_n])
        series = pd.Series({i:len(j) for i,j in sorted_dict.items()})

        title = 'Most Common Albums For Playlist: ' + self._playlist
        return _h_bar(series, title=title, yaxis='Album', xaxis='Number of Songs', long_names=True,
                       hovertext=['<br>'.join(_shorten_names(i)) for i in sorted_dict.values()])

    def graph_similar_playlists_by_current_in_others(self, top_n=10):
        """Calculate similarity based on % of current playlist songs found in other playlists"""
        UNIQUE_SONGS_DF = self._unique_songs_df
        mask = UNIQUE_SONGS_DF['playlist'].apply(lambda x: self._playlist in x)
        df = UNIQUE_SONGS_DF[mask]

        # Fix playlist column to ensure it's properly formatted
        df = self._fix_playlist_column(df.copy())
        
        # Get all other playlists that share songs with current playlist
        other_playlists = set()
        for playlist_list in df['playlist']:
            # Now iterate over the actual playlist names
            for playlist_name in playlist_list:
                if playlist_name != self._playlist:
                    other_playlists.add(playlist_name)
        
        
        # Filter out empty playlists and ensure we have data to work with
        if len(other_playlists) == 0:
            # Return empty graph if no other playlists share songs
            fig = go.Figure()
            fig.update_layout(
                title=f'X % of {self._playlist} Songs are in These Playlists',
                xaxis_title=f'% of {self._playlist} Songs',
                yaxis_title='Playlist',
                annotations=[{
                    'text': f'No other playlists share songs with {self._playlist}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return Markup(fig.to_html(full_html=False))

        dicty = dict()
        for p in other_playlists:
            mask = df['playlist'].apply(lambda x: p in x)
            df2 = df[mask]
            dicty[p] = len(df2.index)

        total = len(df.index)
        relative = {k: v/total*100 for k,
                    v in sorted(dicty.items(), key=lambda x: x[1], reverse=True)[:top_n]}
        counts = {k: str(v) for k, v in sorted(
            dicty.items(), key=lambda x: x[1], reverse=True)}

        series = pd.Series(relative)
        title = f'X % of {self._playlist} Songs are in These Playlists'
        
        
        return _h_bar(series, title=title, xaxis=f'% of {self._playlist} Songs', yaxis='Playlist', percents=True,
                      hovertext=[counts[i] + ' Songs' for i in relative])

    def graph_similar_playlists_by_others_in_current(self, top_n=10):
        """Calculate similarity based on % of other playlist songs found in current playlist"""
        UNIQUE_SONGS_DF = self._unique_songs_df
        mask = UNIQUE_SONGS_DF['playlist'].apply(lambda x: self._playlist in x)
        current_playlist_df = UNIQUE_SONGS_DF[mask]

        # Fix playlist column to ensure it's properly formatted
        current_playlist_df = self._fix_playlist_column(current_playlist_df.copy())
        
        # Get all other playlists that share songs with current playlist
        other_playlists = set()
        for playlist_list in current_playlist_df['playlist']:
            # Now iterate over the actual playlist names
            for playlist_name in playlist_list:
                if playlist_name != self._playlist:
                    other_playlists.add(playlist_name)
        
        
        # Filter out empty playlists and ensure we have data to work with
        if len(other_playlists) == 0:
            # Return empty graph if no other playlists share songs
            fig = go.Figure()
            fig.update_layout(
                title=f'X % of These Playlists are in {self._playlist}',
                xaxis_title='% of Playlist Songs',
                yaxis_title='Playlist',
                annotations=[{
                    'text': f'No other playlists share songs with {self._playlist}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return Markup(fig.to_html(full_html=False))
        
        dicty = dict()
        counts = dict()
        
        for p in other_playlists:
            # Get all songs in the other playlist
            mask_other = UNIQUE_SONGS_DF['playlist'].apply(lambda x: p in x)
            other_playlist_df = UNIQUE_SONGS_DF[mask_other]
            
            # Find songs that are in both playlists
            mask_shared = other_playlist_df['playlist'].apply(lambda x: self._playlist in x)
            shared_songs = other_playlist_df[mask_shared]
            
            # Calculate percentage of other playlist that's in current playlist
            total_in_other = len(other_playlist_df.index)
            shared_count = len(shared_songs.index)
            
            if total_in_other > 0:
                dicty[p] = shared_count / total_in_other * 100
                counts[p] = str(shared_count)

        # Sort by percentage and take top N
        relative = dict(sorted(dicty.items(), key=lambda x: x[1], reverse=True)[:top_n])

        series = pd.Series(relative)
        title = f'X % of These Playlists are in {self._playlist}'
        
        
        return _h_bar(series, title=title, xaxis=f'% of Playlist Songs', yaxis='Playlist', percents=True,
                      hovertext=[counts[i] + ' Songs' for i in relative])

    def graph_similar_playlists(self, top_n=10):
        """Legacy method - kept for backward compatibility"""
        return self.graph_similar_playlists_by_current_in_others(top_n)

    def graph_song_features_boxplot(self):
        df = self._playlist_df
        
        # Check if we have any data
        if len(df.index) == 0:
            # Return empty graph if no data
            fig = go.Figure()
            fig.update_layout(
                title='Audio Features of Songs in ' + self._playlist,
                xaxis_title='Song Feature',
                yaxis_title='Level (Low -> High)',
                annotations=[{
                    'text': 'No audio features data available for this playlist',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return Markup(fig.to_html(full_html=False))

        x_data = FEATURE_COLS
        y_data = []

        df['popularity'] = df['popularity']/100
        y_data = [df[col] for col in FEATURE_COLS if col]

        title = 'Audio Features of Songs in ' + self._playlist
        text = [n + '<br>' + a for n, a in zip(df['name'], df['artist'])]
        xaxis = 'Song Feature'
        yaxis = 'Level (Low -> High)'
        return _boxplot(x_data, y_data, text, title, xaxis, yaxis)


# Top 50 Page ----------------------------------------------------------------------------------------------

class Top50Page():
    def __init__(self, path, unique_songs_df, top_artists, top_artists_pop):
        self._path = path
        self._top_artists = top_artists
        self._top_artists_pop = top_artists_pop
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        # Precompute dynamic graph HTML in-memory
        self._dynamic_graph = self.graph_all_by_time_range()

    def load_dynamic_graph(self):
        return self._dynamic_graph

    def _graph_song_features_boxplot(self, time_range, name=None, color=None):
        x_data = FEATURE_COLS
        y_data = []

        rank_col = 'songs_' + TIME_RANGE_DICT[time_range][1]
        mask = self._unique_songs_df[rank_col].apply(lambda x: type(x) == int)
        df = self._unique_songs_df[mask]

        df['popularity'] = df['popularity']/100
        y_data = [df[col] for col in FEATURE_COLS if col]

        title = 'Top 50 Songs (' + \
            TIME_RANGE_DICT[time_range][0] + ') Audio Features'
        text = [n + '<br>' + a + '<br>Rank: ' +
                str(r) for n, a, r in zip(df['name'], df['artist'], df[rank_col])]
        xaxis = 'Song Feature'
        yaxis = 'Level (Low -> High)'

        # name = 'Top 50 Songs'
        if name and color:
            return _boxplot(x_data, y_data, text, title, xaxis, yaxis, to_html=False, name=name, color=color)
        else:
            return _boxplot(x_data, y_data, text, title, xaxis, yaxis, markup=False, name=TIME_RANGE_DICT[time_range][0])

    def _graph_artist_features_boxplot(self, time_range, name=None, color=None):
        x_data = FEATURE_COLS
        y_data = []

        # Group by artist and calculate median for each feature
        artist_features = {}
        for feature in FEATURE_COLS:
            if feature == 'popularity':
                # For popularity, use the top_artists_pop data (already normalized to 0-1)
                artist_features[feature] = [p/100 for p in self._top_artists_pop[time_range]]
            else:
                # For other features, calculate median across all songs by each artist
                feature_medians = []
                for artist in self._top_artists[time_range]:
                    # Find all songs by this artist in the full unique_songs_df
                    artist_mask = self._unique_songs_df['artist'].apply(lambda x: artist in x.split(', '))
                    artist_songs = self._unique_songs_df[artist_mask]
                    
                    if len(artist_songs) > 0 and feature in artist_songs.columns:
                        # Filter out NaN values before calculating median
                        feature_values = artist_songs[feature].dropna()
                        if len(feature_values) > 0:
                            median_val = feature_values.median()
                            feature_medians.append(median_val)
                        else:
                            # If all values are NaN, use 0
                            feature_medians.append(0.0)
                    else:
                        # If no songs found or feature missing, use 0
                        feature_medians.append(0.0)
                
                artist_features[feature] = feature_medians
        
        # Convert to list format expected by _boxplot
        y_data = [artist_features[feature] for feature in FEATURE_COLS]
        
        # Validation: ensure all feature lists have the same length
        expected_length = len(self._top_artists[time_range])
        for feature in FEATURE_COLS:
            if len(artist_features[feature]) != expected_length:
                # Pad with zeros if missing values
                while len(artist_features[feature]) < expected_length:
                    artist_features[feature].append(0.0)
                # Truncate if too many values
                artist_features[feature] = artist_features[feature][:expected_length]
        
        # Rebuild y_data with validated lengths
        y_data = [artist_features[feature] for feature in FEATURE_COLS]

        title = 'Top 50 Artists (' + \
            TIME_RANGE_DICT[time_range][0] + ') Median Audio Features'
        text = [a + '<br>Rank: ' + str(i) for i, a in enumerate(self._top_artists[time_range], 1)]
        xaxis = 'Song Feature'
        yaxis = 'Level (Low -> High)'

        # name = 'Top 50 Artists AVG Songs'
        if name and color:
            return _boxplot(x_data, y_data, text, title, xaxis, yaxis, to_html=False, name=name, color=color)
        else:
            return _boxplot(x_data, y_data, text, title, xaxis, yaxis, markup=False, name=TIME_RANGE_DICT[time_range][0])

    def _graph_top_genres_and_top_songs_or_artists_by_genres(self, time_range, artists=False, name=None, color=None, top_n=10):
        '''Output = horizontal bar graph of % of Top 50 Songs by Artist Genre'''

        rank_col = 'artists_' if artists else 'songs_'
        rank_col += TIME_RANGE_DICT[time_range][1]

        mask = self._unique_songs_df[rank_col].apply(lambda x: type(x) == int)
        df = self._unique_songs_df[mask]

        total = len(df[rank_col].unique())   # total should be 50 but might be missing if not in playlists - should update to include them anyway

        num_by_genre = dict()
        top_by_genre = dict()
        # Fix: Correctly extract unique genres from nested list structure
        unique_genres = set()
        for genre_list in df['genres']:
            if isinstance(genre_list, list):
                for genre_group in genre_list:
                    if isinstance(genre_group, list):
                        # Handle nested list structure: [['genre1'], ['genre2']]
                        for genre in genre_group:
                            if isinstance(genre, str) and genre.strip():
                                unique_genres.add(genre.strip())
                    elif isinstance(genre_group, str) and genre_group.strip():
                        # Handle simple list structure: ['genre1', 'genre2']
                        unique_genres.add(genre_group.strip())
            elif isinstance(genre_list, str) and genre_list.strip():
                # Handle single string genre
                unique_genres.add(genre_list.strip())
        
        for g in unique_genres:
            # Fix: Correctly check if genre exists in both nested and simple structures
            def genre_exists_in_row(genre_row):
                if isinstance(genre_row, list):
                    for genre_group in genre_row:
                        if isinstance(genre_group, list):
                            # Nested structure: [['genre1'], ['genre2']]
                            return g in [genre.strip() for genre in genre_group if isinstance(genre, str)]
                        elif isinstance(genre_group, str):
                            # Simple structure: ['genre1', 'genre2']
                            return g == genre_group.strip()
                elif isinstance(genre_row, str):
                    # Single string genre
                    return g == genre_row.strip()
                return False
            
            mask = df['genres'].apply(genre_exists_in_row)
            df2 = df[mask]

            if artists:
                df2.drop_duplicates(subset=['artist'], inplace=True)
                # df2[rank_col] = df2[rank_col].astype(int)

            num_by_genre[g] = len(df2.index)

            df2 = df2.sort_values(by=rank_col)
            df2 = df2.head(MAX_HOVER_ROWS)  # only keep top in hovertext
            if artists:
                top_by_genre[g] = list(zip(df2['artist'], df2[rank_col]))
            else:
                top_by_genre[g] = list(zip(df2['name'], df2[rank_col]))

        relative = {k: v/total*100 for k,
                    v in sorted(num_by_genre.items(), key=lambda x: x[1], reverse=True)[:top_n]}
        counts = {k: str(v) for k, v in sorted(
            num_by_genre.items(), key=lambda x: x[1], reverse=True)}

        series = pd.Series(relative)
        series.sort_values(inplace=True)

        # name = 'Top 10 Artist Genres' if artists else 'Top 10 Song Genres'
        if artists:
            hovertext = [counts[i] + ' Artists<br><br>(Artist, Rank)<br>' + '<br>'.join(
                [str(j) for j in top_by_genre[i]]) for i in series.keys()]
        else:
            hovertext = [counts[i] + ' Songs<br><br>(Song, Rank)<br>' + '<br>'.join(
                [str(j) for j in top_by_genre[i]]) for i in series.keys()]

        if name and color:
            return _h_bar(series, percents=True, hovertext=hovertext, to_html=False, name=name, color=color)
        else:
            return _h_bar(series, percents=True, hovertext=hovertext, markup=False, name=TIME_RANGE_DICT[time_range][0])


    def graph_all_by_time_range(self):
        labels = [i[0] for i in TIME_RANGE_DICT.values()]
        fig = subplots.make_subplots(rows=4, cols=1, vertical_spacing=.08,
                                     subplot_titles=('<b>Top 50 Songs\' Features</b>', '<b>Top 50 Artists\' Median Song Features</b>',
                                                     '<b>Top 10 Song Genres & Top 5 Songs Per Genre</b>', '<b>Top 10 Artist Genres & Top 5 Artists Per Genre</b>'))

        # Multi-colors for 1 Time Range = Short, Med, Long
        for i in [0, 1, 2]:
            song_features = self._graph_song_features_boxplot(i)
            fig.add_traces(song_features.data, 1, 1)

            artist_features = self._graph_artist_features_boxplot(i)
            fig.add_traces(artist_features.data, 2, 1)

            genres_by_songs = self._graph_top_genres_and_top_songs_or_artists_by_genres(
                i)
            fig.add_traces(genres_by_songs.data, 3, 1)

            genres_by_artists = self._graph_top_genres_and_top_songs_or_artists_by_genres(
                i, artists=True)
            fig.add_traces(genres_by_artists.data, 4, 1)

        # Same-colors for all Time Ranges
        for i in [0, 1, 2]:
            song_features = self._graph_song_features_boxplot(
                i, name=labels[i], color=COLORS[i])
            fig.add_trace(song_features, 1, 1)

            artist_features = self._graph_artist_features_boxplot(
                i, name=labels[i], color=COLORS[i])
            fig.add_trace(artist_features, 2, 1)

            genres_by_songs = self._graph_top_genres_and_top_songs_or_artists_by_genres(
                i, name=labels[i], color=COLORS[i])
            fig.add_trace(genres_by_songs, 3, 1)

            genres_by_artists = self._graph_top_genres_and_top_songs_or_artists_by_genres(
                i, artists=True, name=labels[i], color=COLORS[i])
            fig.add_trace(genres_by_artists, 4, 1)

        # Create buttons for drop down menu
        buttons = []
        for i, label in enumerate(labels):
            # Create visibility array based on actual trace structure:
            # Time period 0: traces 0-17 (8 song + 8 artist + 1 song genres + 1 artist genres)
            # Time period 1: traces 18-35 (8 song + 8 artist + 1 song genres + 1 artist genres)  
            # Time period 2: traces 36-53 (8 song + 8 artist + 1 song genres + 1 artist genres)
            # "All" button: traces 54-65 (12 traces total)
            visibility = []
            
            # For each time period (3 total)
            for j in range(len(labels)):
                if j == i:  # Show traces for this time period
                    # Graph 1: 8 boxplot traces (songs features)
                    visibility.extend([True] * 8)
                    # Graph 2: 8 boxplot traces (artists features) 
                    visibility.extend([True] * 8)
                    # Graph 3: 1 bar chart trace (song genres)
                    visibility.extend([True] * 1)
                    # Graph 4: 1 bar chart trace (artist genres)
                    visibility.extend([True] * 1)
                else:  # Hide traces for other time periods
                    visibility.extend([False] * 18)
            
            # Hide the "All" button traces (12 traces)
            visibility.extend([False] * 12)
            
            button = dict(
                method='update',
                label=label,
                args=[{'visible': visibility},
                      {'boxmode': 'overlay', 'showlegend': False}]
            )
            buttons.append(button)

        allButton = dict(
            method='update',
            label='All',
            args=[{'visible': [False]*54 + [True]*12},
                  {'boxmode': 'group', 'showlegend': True}]
        )
        buttons += [allButton]

        updatemenus = list([
            dict(type='buttons',
                 direction='right',
                 active=0,
                 x=.5,
                 pad={"r": 10, "t": 10},
                 y=1.065,
                 buttons=buttons,
                 showactive=True,
                 ),
        ])

        # hoverlabel_font_color='white'
        fig.update_layout(height=1500, updatemenus=updatemenus, showlegend=False,
                          annotations=[
                              dict(text="Time Period",
                                   x=.3,
                                   xref="paper",
                                   y=1.045,
                                   align="left",
                                   showarrow=False)
                          ]
                          )

        # Set initial visibility: show first time period (18 traces) and hide the rest
        Ld = len(fig.data)
        
        # Hide all traces initially
        for k in range(Ld):
            fig.update_traces(visible=False, selector=k)
        
        # Show first time period traces (0-17)
        for k in range(18):
            fig.update_traces(visible=True, selector=k)

        # edit axis labels
        fig['layout']['xaxis']['title'] = 'Audio Features'
        fig['layout']['xaxis2']['title'] = 'Audio Features'
        fig['layout']['xaxis3']['title'] = '% of Top 50 Songs'
        fig['layout']['xaxis4']['title'] = '% of Top 50 Artists'

        fig['layout']['yaxis']['title'] = 'Level (Low-High)'
        fig['layout']['yaxis2']['title'] = 'Level (Low-High)'
        fig['layout']['yaxis3']['title'] = 'Genre'
        fig['layout']['yaxis4']['title'] = 'Genre'

        return Markup(fig.to_html(full_html=False))


# Analyze Single Artist Page ----------------------------------------------------------------------------------------------

class AnalyzeArtistPage(OptimizedPageBase):
    def __init__(self, artist, all_songs_df, unique_songs_df, global_artist_averages=None):
        super().__init__(None, all_songs_df, unique_songs_df)
        self._artist = artist
        self._global_artist_averages = global_artist_averages

        mask = self._all_songs_df['artist'].apply(lambda x: artist in x.split(', '))
        self._artist_df = self._all_songs_df[mask]
        # self._pure_artist_df = self._all_songs_df[self._all_songs_df['artist'] == artist]

        row = self._unique_songs_df[self._unique_songs_df['artist'].apply(lambda x: artist in x.split(', '))].iloc[0]
        ind = row['artist'].split(', ').index(artist) if type(row['artists_short_rank']) != int and row['artists_short_rank']!='N/A' else -1
        self._artist_ranks = [row['artists_short_rank'].split(', ')[ind] if ind != -1 else row['artists_short_rank'],
                              row['artists_med_rank'].split(', ')[ind] if ind != -1 else row['artists_med_rank'], 
                              row['artists_long_rank'].split(', ')[ind] if ind != -1 else row['artists_long_rank']]
        
        # Safely access genres with bounds checking
        if ind != -1 and ind < len(row['genres']):
            self._artist_genres = row['genres'][ind]
        elif len(row['genres']) > 0:
            self._artist_genres = row['genres'][0]
        else:
            self._artist_genres = ["N/A"]

        self._user_playlists = self._all_songs_df['playlist'].unique()

    def artist_genres(self):
        return self._artist + ' Genres: ' + ', '.join(self._artist_genres)

    def graph_count_timeline(self):
        """Create a timeline graph for a specific artist showing when their songs were added to playlists"""
        # Filter data for this specific artist
        artist_df = self._artist_df
        
        # Separate liked songs from other playlists
        liked_songs_df = artist_df[artist_df['playlist'] == 'Liked Songs']
        all_playlists_df = artist_df[artist_df['playlist'] != 'Liked Songs']
        
        # Create timeline data for main playlists (excluding Liked Songs)
        if len(all_playlists_df.index) > 0:
            main_timeline = all_playlists_df.groupby('date_added').size().reset_index(name='count')
            main_timeline = main_timeline.sort_values('date_added')
            
            # Create cumulative timeline for main playlists
            main_continuous_timeline = main_timeline.copy()
            main_continuous_timeline['count'] = main_continuous_timeline['count'].cumsum()
            
            # Create hovertext mapping for main timeline
            main_hovertext = []
            for date in main_timeline['date_added']:
                songs_on_date = all_playlists_df[all_playlists_df['date_added'] == date]['name'].tolist()
                hovertext = '<br>'.join(songs_on_date)
                main_hovertext.append(hovertext)
        else:
            main_timeline = pd.DataFrame(columns=['date_added', 'count'])
            main_continuous_timeline = pd.DataFrame(columns=['date_added', 'count'])
            main_hovertext = []
        
        # Create timeline data for liked songs
        if len(liked_songs_df.index) > 0:
            liked_timeline = liked_songs_df.groupby('date_added').size().reset_index(name='count')
            liked_timeline = liked_timeline.sort_values('date_added')
            
            # Create cumulative timeline for liked songs
            liked_continuous_timeline = liked_timeline.copy()
            liked_continuous_timeline['count'] = liked_continuous_timeline['count'].cumsum()
            
            # Create hovertext mapping for liked songs timeline
            liked_hovertext = []
            for date in liked_timeline['date_added']:
                songs_on_date = liked_songs_df[liked_songs_df['date_added'] == date]['name'].tolist()
                hovertext = '<br>'.join(songs_on_date)
                liked_hovertext.append(hovertext)
        else:
            liked_timeline = pd.DataFrame(columns=['date_added', 'count'])
            liked_continuous_timeline = pd.DataFrame(columns=['date_added', 'count'])
            liked_hovertext = []
        
        # Create figure with subplots
        fig = subplots.make_subplots(
            rows=1, cols=1
        )
        
        # Add main traces (excluding Liked Songs)
        if len(main_timeline.index) > 0:
            # Main line trace
            main_line_trace = go.Scatter(
                x=main_timeline['date_added'],
                y=main_timeline['count'],
                mode='lines+markers',
                name='Artist Songs Added',
                line=dict(color=MAIN_TIMELINE_COLOR),
                visible=True,
                hovertext=main_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Main continuous trace
            main_continuous_trace = go.Scatter(
                x=main_continuous_timeline['date_added'],
                y=main_continuous_timeline['count'],
                mode='lines+markers',
                name='Artist Songs Added (Cumulative)',
                line=dict(color=MAIN_TIMELINE_COLOR),
                visible=False,
                hovertext=main_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Main bar trace
            main_bar_trace = go.Bar(
                x=main_timeline['date_added'],
                y=main_timeline['count'],
                name='Artist Songs Added',
                marker_color=MAIN_TIMELINE_COLOR,
                visible=False,
                hovertext=main_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            fig.add_trace(main_line_trace)
            fig.add_trace(main_continuous_trace)
            fig.add_trace(main_bar_trace)
        
        # Add liked songs traces
        if len(liked_timeline.index) > 0:
            # Liked songs line trace
            liked_line_trace = go.Scatter(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'],
                mode='lines+markers',
                name='Liked Songs by Artist',
                line=dict(color=LIKED_TIMELINE_COLOR),
                visible=True,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Liked songs continuous trace
            liked_continuous_trace = go.Scatter(
                x=liked_continuous_timeline['date_added'],
                y=liked_continuous_timeline['count'],
                mode='lines+markers',
                name='Liked Songs by Artist (Cumulative)',
                line=dict(color=LIKED_TIMELINE_COLOR),
                visible=False,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            # Liked songs bar trace
            liked_bar_trace = go.Bar(
                x=liked_timeline['date_added'],
                y=liked_timeline['count'],
                name='Liked Songs by Artist',
                marker_color=LIKED_TIMELINE_COLOR,
                visible=False,
                hovertext=liked_hovertext,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<br>%{hovertext}<extra></extra>'
            )
            
            fig.add_trace(liked_line_trace)
            fig.add_trace(liked_continuous_trace)
            fig.add_trace(liked_bar_trace)
        
        # Add vertical dashed lines for each year anniversary
        if len(all_playlists_df.index) > 0:
            first_date = all_playlists_df['date_added'].min()
            # Convert string date to datetime object
            first_date_dt = datetime.datetime.fromisoformat(first_date.replace('Z', '+00:00'))
            first_year = first_date_dt.year
            current_year = datetime.datetime.now().year
            
            # Get month and day from today's date (current date)
            today = datetime.datetime.now()
            today_month = today.month
            today_day = today.day
            
            for year in range(first_year, current_year + 1):
                try:
                    anniversary_date = datetime.datetime(year, today_month, today_day)
                    
                    fig.add_vline(
                        x=anniversary_date,
                        line_width=3,
                        line_dash="dash",
                        line_color="green"
                    )
                except ValueError:
                    # Skip invalid dates (e.g., February 29 in non-leap years)
                    continue
        
        # Add horizontal lines for average constants when Continuous button is clicked
        # Use global averages from the HomePage calculation
        if self._global_artist_averages:
            avg_liked_songs, avg_total_songs = self._global_artist_averages
        else:
            avg_liked_songs = 0
            avg_total_songs = 0
        
        # Add horizontal line for average total songs per artist
        if avg_total_songs > 0:
            # Create horizontal line as a scatter trace for better visibility control
            # Get x range from the data
            if len(all_playlists_df.index) > 0:
                x_min = all_playlists_df['date_added'].min()
                x_max = all_playlists_df['date_added'].max()
                # Convert to datetime if they're strings
                if isinstance(x_min, str):
                    x_min = datetime.datetime.fromisoformat(x_min.replace('Z', '+00:00'))
                if isinstance(x_max, str):
                    x_max = datetime.datetime.fromisoformat(x_max.replace('Z', '+00:00'))
            else:
                x_min = datetime.datetime(2020, 1, 1)
                x_max = datetime.datetime(2024, 12, 31)
            
            avg_total_line = go.Scatter(
                x=[x_min, x_max],  # Span the full x range
                y=[avg_total_songs, avg_total_songs],
                mode='lines',
                name=f"Your Avg # of Songs Added per Artist = {avg_total_songs:.2f}",
                line=dict(color=MAIN_AVG_COLOR, dash='dash', width=2),
                visible=False,
                showlegend=True,
                hoverinfo='skip'
            )
            fig.add_trace(avg_total_line)
        
        # Add horizontal line for average liked songs per artist
        if avg_liked_songs > 0:
            # Create horizontal line as a scatter trace for better visibility control
            # Get x range from the data
            if len(all_playlists_df.index) > 0:
                x_min = all_playlists_df['date_added'].min()
                x_max = all_playlists_df['date_added'].max()
                # Convert to datetime if they're strings
                if isinstance(x_min, str):
                    x_min = datetime.datetime.fromisoformat(x_min.replace('Z', '+00:00'))
                if isinstance(x_max, str):
                    x_max = datetime.datetime.fromisoformat(x_max.replace('Z', '+00:00'))
            else:
                x_min = datetime.datetime(2020, 1, 1)
                x_max = datetime.datetime(2024, 12, 31)
            
            avg_liked_line = go.Scatter(
                x=[x_min, x_max],  # Span the full x range
                y=[avg_liked_songs, avg_liked_songs],
                mode='lines',
                name=f"Your Avg # of Liked Songs per Artist = {avg_liked_songs:.2f}",
                line=dict(color=LIKED_AVG_COLOR, dash='dash', width=2),
                visible=False,
                showlegend=True,
                hoverinfo='skip'
            )
            fig.add_trace(avg_liked_line)
        
        # Create buttons for Line, Continuous, Bar views
        buttons = []
        
        # Determine which traces exist
        has_main_traces = len(main_timeline.index) > 0
        has_liked_traces = len(liked_timeline.index) > 0
        
        if has_main_traces and has_liked_traces:
            # Both main and liked traces exist (6 traces total)
            line_visibility = [True, False, False, True, False, False]  # main line, main continuous, main bar, liked line, liked continuous, liked bar
            continuous_visibility = [False, True, False, False, True, False]
            bar_visibility = [False, False, True, False, False, True]
        elif has_main_traces:
            # Only main traces exist (3 traces total)
            line_visibility = [True, False, False]
            continuous_visibility = [False, True, False]
            bar_visibility = [False, False, True]
        elif has_liked_traces:
            # Only liked traces exist (3 traces total)
            line_visibility = [True, False, False]
            continuous_visibility = [False, True, False]
            bar_visibility = [False, False, True]
        else:
            # No traces exist
            line_visibility = []
            continuous_visibility = []
            bar_visibility = []
        
        # Add horizontal lines visibility (only visible for Continuous button)
        # Add horizontal line traces to visibility arrays
        if avg_total_songs > 0:
            line_visibility.append(False)  # horizontal line for total songs
            continuous_visibility.append(True)
            bar_visibility.append(False)
        if avg_liked_songs > 0:
            line_visibility.append(False)  # horizontal line for liked songs
            continuous_visibility.append(True)
            bar_visibility.append(False)
        

        
        # Line button
        if line_visibility:
            buttons.append(
                dict(
                    method='update',
                    label='Line',
                    args=[{'visible': line_visibility}]
                )
            )
        
        # Continuous button
        if continuous_visibility:
            buttons.append(
                dict(
                    method='update',
                    label='Continuous',
                    args=[{'visible': continuous_visibility}]
                )
            )
        
        # Bar button
        if bar_visibility:
            buttons.append(
                dict(
                    method='update',
                    label='Bar',
                    args=[{'visible': bar_visibility}]
                )
            )
        
        # Add button menu (visible buttons, not dropdown)
        updatemenus = [dict(
            type='buttons',
            direction='right',
            active=0,
            y=1.3,
            x=0.6,
            buttons=buttons
        )]
        
        # Update layout - title needed here not in subplot_title
        fig.update_layout(
            title='Timeline of Adding ' + self._artist + ' Songs to Playlists',
            barmode='stack',
            updatemenus=updatemenus
        )
        
        # Update axes
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        fig.update_yaxes(title='Daily Added Songs')
        
        return Markup(fig.to_html(full_html=False))

    def graph_top_rank_table(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(
            lambda x: self._artist in x.split(', '))]

        drop_df = df[(df['songs_short_rank'] == 'N/A') &
                     (df['songs_med_rank'] == 'N/A') & (df['songs_long_rank'] == 'N/A')]
        df = df.drop(drop_df.index)

        # Handle 'N/A', 'nan', 'None', and None values before converting to int
        df['songs_short_rank'].replace({'N/A': 51, 'nan': 51, 'None': 51, None: 51}, inplace=True)
        df['songs_short_rank'] = df['songs_short_rank'].astype(int)
        df = df.sort_values(by='songs_short_rank', ascending=True)
        df['songs_short_rank'].replace({51: 'N/A'}, inplace=True)

        names_col = ['<b>' + self._artist + '</b>'] + df['name'].to_list()
        short_col = ['<b>' + str(self._artist_ranks[0]) + '</b>'] + df['songs_short_rank'].to_list()
        med_col = ['<b>' + str(self._artist_ranks[1]) + '</b>'] + df['songs_med_rank'].to_list()
        long_col = ['<b>' + str(self._artist_ranks[2]) + '</b>'] + df['songs_long_rank'].to_list()

        fig = go.Figure(data=[go.Table(
            # columnorder = [1, 2, 3, 4],
            # columnwidth = [80,400],
            header=dict(
                values=[['<b>Top 50 Rank</b>'],
                        ['<b>Last 4 Weeks</b>'],
                        ['<b>Last 6 Months</b>'],
                        ['<b>All Time</b>']],
                line_color='darkslategray',
                fill_color='royalblue',
                # align=['left','center'],
                font=dict(color='white', size=18),
                height=40
            ),
            cells=dict(
                values=[names_col, short_col, med_col, long_col],
                line_color='darkslategray',
                # fill=dict(color=['paleturquoise', 'white']),
                align='center',
                font_size=18,
                height=40)
        )
        ])
        fig.update_layout(
            title_text='Artist & Artist Songs Rank in Top 50', height=max(TOP_RANK_TABLE_MIN_HEIGHT, (len(names_col)+1)*40))

        return Markup(fig.to_html(full_html=False))

    def graph_top_playlists_by_artist(self):
        return shared_graph_top_playlists_by_artist(self._all_songs_df, self._artist)

    def graph_top_songs_by_artist(self):
        return shared_graph_top_songs_by_artist(self._unique_songs_df, self._artist)

    def graph_song_features_boxplot(self):
        df = self._unique_songs_df[self._unique_songs_df['artist'].apply(
            lambda x: self._artist in x.split(', '))]

        x_data = FEATURE_COLS
        y_data = []

        df['popularity'] = df['popularity']/100
        y_data = [df[col] for col in FEATURE_COLS if col]

        title = 'Audio Features of ' + self._artist + ' Songs'
        text = [n + '<br>' + a for n, a in zip(df['name'], df['artist'])]
        xaxis = 'Song Feature'
        yaxis = 'Level (Low -> High)'
        return _boxplot(x_data, y_data, text, title, xaxis, yaxis)

    def graph_playlists_by_artist_genres(self):
        '''Return a stacked horizontal bar graph, with each bar being a playlist, 
        and each stack being % of artist genre in that playlist'''

        # Create dataframe - each row a playlist, each column a genre
        percents = defaultdict(list)
        names = defaultdict(dict)
        data = []
        
        # Flatten artist genres first for consistent processing
        flat_genre_names = []
        # self._artist_genres is already a list of strings, not a list of lists
        if self._artist_genres and len(self._artist_genres) > 0 and self._artist_genres != ["N/A"]:
            for genre_name in self._artist_genres:
                if genre_name and genre_name.strip():
                    flat_genre_names.append(genre_name.strip())
        else:
            flat_genre_names.append("N/A")
        
        for g in flat_genre_names:
            for p in self._user_playlists:
                df = self._unique_songs_df[self._unique_songs_df['playlist'].apply(
                    lambda x: p in x)]
                
                # Fix the genre matching logic - genres are stored as list of lists
                mask = df['genres'].apply(
                    lambda x: g in {z for y in x for z in y} if isinstance(x, list) else False)
                percents[p].append(len(df[mask])/len(df)*100)
                names[g][p] = [str(len(df[mask])) + ' Songs'] + df[mask].sample(min(len(df[mask]),MAX_HOVER_ROWS))['name'].to_list()

        for i in percents:
            percents[i].append(sum(percents[i]))

        df = pd.DataFrame.from_dict(
            percents, orient='index')
        
        df.columns = flat_genre_names + ['sum']
        df = df.sort_values(by='sum', ascending=False).head(10)


        for i, g in enumerate(flat_genre_names):
            series = pd.Series(
                dict(zip(df.index, df[g])))
            # Create a more descriptive trace name to handle single character genres and empty names
            if not g or g.strip() == "" or g == "N/A":
                trace_name = f"Unknown Genre {i+1}"
            elif len(g.strip()) <= 2:
                trace_name = f"{g.strip()} Genre"
            else:
                trace_name = g.strip()
            go_bar = _h_bar(series, name=trace_name, 
                            hovertext=['<br>'.join(names[g][i]) for i in df.index],
                            percents=True, to_html=False)
            data.append(go_bar)

        fig = go.Figure(data=data)
        fig.update_layout(barmode='stack', yaxis={
            'categoryorder': 'total ascending'}, yaxis_title='Playlist', xaxis_title='% of Songs With Genre', title='Playlists Most Likely To Have Songs Similar to ' + self._artist)
        return Markup(fig.to_html(full_html=False))


# Analyze Multiple Artists Page ----------------------------------------------------------------------------------------------

class AnalyzeArtistsPage():
    def __init__(self, artists, all_songs_df, unique_songs_df, global_artist_averages=None):
        self._artists = artists
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)
        self._global_artist_averages = global_artist_averages

        self._artist_genres = []
        for a in artists:
            row = unique_songs_df[unique_songs_df['artist'].apply(lambda x: a in x.split(', '))].iloc[0]
            ind = row['artist'].split(', ').index(a) if type(row['artists_short_rank']) != int and row['artists_short_rank']!='N/A' else -1
            
            # Safely access genres with bounds checking
            if ind != -1 and ind < len(row['genres']):
                self._artist_genres.append(row['genres'][ind])
            elif len(row['genres']) > 0:
                self._artist_genres.append(row['genres'][0])
            else:
                self._artist_genres.append(["N/A"])

    def graph_artist_timelines(self):
        """Create timeline graphs showing when each artist's songs were added to playlists vs Liked Songs"""
        if not self._artists or len(self._artists) == 0:
            return None
        
        # Get global artist averages
        global_avg_liked_songs_per_artist = 0
        global_avg_total_songs_per_artist = 0
        if self._global_artist_averages:
            global_avg_liked_songs_per_artist, global_avg_total_songs_per_artist = self._global_artist_averages
        
        # Create the figure
        fig = go.Figure()
        
        # Add traces for each artist
        for i, artist in enumerate(self._artists):
            color = COLORS[i % len(COLORS)]
            
            # Filter data for this artist (excluding Liked Songs playlist)
            artist_df = self._all_songs_df[
                (self._all_songs_df['artist'].apply(lambda x: artist in x.split(', '))) &
                (self._all_songs_df['playlist'] != 'Liked Songs')
            ]
            
            # Filter data for this artist in Liked Songs playlist
            artist_liked_df = self._all_songs_df[
                (self._all_songs_df['artist'].apply(lambda x: artist in x.split(', '))) &
                (self._all_songs_df['playlist'] == 'Liked Songs')
            ]
            
            if len(artist_df.index) > 0:
                # Group by date and count songs, create hovertext for non-liked songs
                timeline_data = artist_df.groupby('date_added').agg({
                    'name': lambda x: list(x),
                    'artist': lambda x: list(x)
                }).reset_index()
                
                # Create hovertext with song names separated by <br>
                hovertext = []
                for _, row in timeline_data.iterrows():
                    song_names = [f"{name} by {artist}" for name, artist in zip(row['name'], row['artist'])]
                    hovertext.append('<br>'.join(song_names))
                
                # Add Non Liked Songs Continuous trace
                fig.add_trace(go.Scatter(
                    x=timeline_data['date_added'],
                    y=timeline_data['name'].apply(len).cumsum(),
                    mode='lines+markers',
                    name=artist + ' Playlist Songs',
                    line=dict(color=color),
                    marker=dict(size=6),
                    hovertext=hovertext,
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Date: %{x}<br>' +
                                'Cumulative Songs Added: %{y}<extra></extra>'
                ))
            
            if len(artist_liked_df.index) > 0:
                # Group by date and count songs, create hovertext for liked songs
                liked_timeline_data = artist_liked_df.groupby('date_added').agg({
                    'name': lambda x: list(x),
                    'artist': lambda x: list(x)
                }).reset_index()
                
                # Create hovertext with song names separated by <br>
                liked_hovertext = []
                for _, row in liked_timeline_data.iterrows():
                    song_names = [f"{name} by {artist}" for name, artist in zip(row['name'], row['artist'])]
                    liked_hovertext.append('<br>'.join(song_names))
                
                # Add Liked Songs Continuous trace
                fig.add_trace(go.Scatter(
                    x=liked_timeline_data['date_added'],
                    y=liked_timeline_data['name'].apply(len).cumsum(),
                    mode='lines+markers',
                    name=f"{artist} Liked Songs",
                    line=dict(color=color),
                    marker=dict(size=6, symbol='diamond'),
                    hovertext=liked_hovertext,
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Date: %{x}<br>' +
                                'Cumulative Liked Songs: %{y}<extra></extra>'
                ))
        
        # Add horizontal average lines
        if global_avg_total_songs_per_artist > 0:
            fig.add_hline(
                y=global_avg_total_songs_per_artist,
                line_dash="dash",
                line_color=MAIN_AVG_COLOR,
                line_width=2,
                annotation_text=f"Your Avg # of Songs Added per Artist = {global_avg_total_songs_per_artist:.2f}",
                annotation_position="top right"
            )
        
        if global_avg_liked_songs_per_artist > 0:
            fig.add_hline(
                y=global_avg_liked_songs_per_artist,
                line_dash="dash",
                line_color=LIKED_AVG_COLOR,
                line_width=2,
                annotation_text=f"Your Avg # of Liked Songs per Artist = {global_avg_liked_songs_per_artist:.2f}",
                annotation_position="top left"
            )
        
        # Update layout
        fig.update_layout(
            title_text="Timeline of When Artists' Songs Were Added to Any Playlist",
            xaxis_title="Date",
            yaxis_title="Cumulative Number of Songs Added",
            height=600,
            showlegend=True
        )
        
        # Update x-axis to show dates properly
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        return Markup(fig.to_html(full_html=False))

    def graph_artist_genres(self):
        return _venn_diagram_artist_genres(
            self._artists, self._artist_genres)
    
    def graph_top_rank_table(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        figs = []
        for a in self._artists:
            df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(
                lambda x: a in x.split(', '))]

            drop_df = df[(df['songs_short_rank'] == 'N/A') &
                        (df['songs_med_rank'] == 'N/A') & (df['songs_long_rank'] == 'N/A')]
            df = df.drop(drop_df.index)

            # Handle 'N/A', 'nan', 'None', and None values before converting to int
            df['songs_short_rank'].replace({'N/A': 51, 'nan': 51, 'None': 51, None: 51}, inplace=True)
            df['songs_short_rank'] = df['songs_short_rank'].astype(int)
            df = df.sort_values(by='songs_short_rank', ascending=True)
            df['songs_short_rank'].replace({51: 'N/A'}, inplace=True)

            row = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))].iloc[0]
            ind = row['artist'].split(', ').index(a) if type(row['artists_short_rank']) != int and row['artists_short_rank']!='N/A' else -1
            artist_ranks = [row['artists_short_rank'].split(', ')[ind] if ind != -1 else row['artists_short_rank'],
                                row['artists_med_rank'].split(', ')[ind] if ind != -1 else row['artists_med_rank'], 
                                row['artists_long_rank'].split(', ')[ind] if ind != -1 else row['artists_long_rank']]
            names_col = ['<b>' + a + '</b>'] + df['name'].to_list()
            short_col = ['<b>' + str(artist_ranks[0]) + '</b>'] + df['songs_short_rank'].to_list()
            med_col = ['<b>' + str(artist_ranks[1]) + '</b>'] + df['songs_med_rank'].to_list()
            long_col = ['<b>' + str(artist_ranks[2]) + '</b>'] + df['songs_long_rank'].to_list()

            fig = go.Figure(data=[go.Table(
                # columnorder = [1, 2, 3, 4],
                # columnwidth = [80,400],
                header=dict(
                    values=[['<b>Top 50 Rank</b>'],
                            ['<b>Last 4 Weeks</b>'],
                            ['<b>Last 6 Months</b>'],
                            ['<b>All Time</b>']],
                    line_color='darkslategray',
                    fill_color='royalblue',
                    # align=['left','center'],
                    font=dict(color='white', size=18),
                    height=40
                ),
                cells=dict(
                    values=[names_col, short_col, med_col, long_col],
                    line_color='darkslategray',
                    # fill=dict(color=['paleturquoise', 'white']),
                    align='center',
                    font_size=18,
                    height=40)
            )
            ])
            fig.update_layout(
                title_text= a + '\'s' + ' Ranks in Top 50', height=max(TOP_RANK_TABLE_MIN_HEIGHT, (len(names_col)+1)*40))

            figs.append(Markup(fig.to_html(full_html=False)))
        return figs
    
    def graph_playlists_by_artists(self):
        '''Return a stacked horizontal bar graph
        full bar = sum of songs from playlist from given artists
        each color = # playlist songs per artist'''

        # Create dataframe - each row a playlist, each column an artist
        counts = defaultdict(list)
        data = []
        songs = defaultdict(list)
        for a in self._artists:
            df = self._all_songs_df[self._all_songs_df['artist'].apply(
                lambda x: a in x.split(', '))]
            for p in self._all_songs_df['playlist'].unique():
                adf = df[df['playlist'] == p]
                counts[p].append(len(adf))
                songs[p].append('<br>'.join(adf.head(MAX_HOVER_ROWS)['name']))

        for i in counts:
            counts[i].append(sum(counts[i]))

        df = pd.DataFrame.from_dict(
            counts, orient='index')
        df.columns = self._artists + ['sum']
        df = df.sort_values(by='sum', ascending=False).head(10)

        for i in range(len(self._artists)):
            hovertext = [songs[p][i] for p in df.index]
            series = pd.Series(
                dict(zip(df.index, df[self._artists[i]])))
            go_bar = _h_bar(series, name=self._artists[i], hovertext=hovertext, to_html=False)
            data.append(go_bar)

        fig = go.Figure(data=data)
        fig.update_layout(barmode='stack', yaxis={
            'categoryorder': 'total ascending'}, yaxis_title='Playlist', xaxis_title='# of Songs In Playlist', title='Most Common Playlists Among Artists')
        return Markup(fig.to_html(full_html=False))

    def _get_artist_boxplot(self, artist, color):
        df = self._unique_songs_df[self._unique_songs_df['artist'].apply(lambda x: artist in x.split(', '))]

        x_data = FEATURE_COLS
        y_data = []

        df['popularity'] = df['popularity']/100
        y_data = [df[col] for col in FEATURE_COLS if col]

        title = 'Audio Features of ' + artist + '\'s Songs'
        text = df['name'].to_list()
        xaxis = 'Song Feature'
        yaxis = 'Level (Low -> High)'
        return _boxplot(x_data, y_data, text, title, xaxis, yaxis, to_html=False, name=artist, color=color)

    def graph_artists_boxplots(self):
        '''See Boxplots of Artists Songs Side by Side'''
        fig = subplots.make_subplots(rows=1, cols=1)

        # Features have same colors for Same Playlist
        colorInd = 0
        for i in self._artists:
            artist_boxplot = self._get_artist_boxplot(i, COLORS[colorInd])
            fig.add_trace(artist_boxplot, 1, 1)
            colorInd += 1

        fig.update_layout(title_text="Comparing Artists\' Audio Features",
                  xaxis=dict(title="Audio Features"),
                  yaxis=dict(title="Level (Low-High)"),
                  boxmode='group',
                  showlegend=True)

        return Markup(fig.to_html(full_html=False))


# Analyze Single Song Page ----------------------------------------------------------------------------------------------
class SingleSongPage():
    def __init__(self, song_id, all_songs_df, unique_songs_df):
        mask = unique_songs_df['id'].apply(lambda x: song_id in x)
        self._song_df = unique_songs_df[mask]

        self._song = self._song_df.iloc[0]['name']
        self._artist = self._song_df.iloc[0]['artist']

        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        if len(self._song_df.index) > 0:
            self._artist_genres = self._song_df.iloc[0]['genres']
            self._song_genres = list(
                {j for i in self._artist_genres for j in i})
        else:
            self._artist_genres = []
            self._song_genres = None

    def get_song(self):
        return self._song
    
    def get_artist(self):
        return self._artist

    def graph_top_rank_table(self):
        return shared_graph_top_rank_table(self._song_df)

    def graph_song_features_vs_avg(self):
        """
        Create a vertical bar graph comparing audio features across different data sources.
        Features: all playlists median, artist median, and song's actual values.
        """
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        # Get the specific song data
        song_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
        song_df = song_df[song_df['artist'] == self._artist]

        if len(song_df) == 0:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Song not found in dataset</div>')

        # Filter for available feature columns
        available_feature_cols = []
        for col in FEATURE_COLS:
            if col in song_df.columns and not song_df[col].isna().all():
                available_feature_cols.append(col)
        
        if not available_feature_cols:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No audio features data available for this song</div>')

        # Get song's actual values
        song_values = song_df[available_feature_cols].iloc[0]

        # Get all playlists median
        all_playlists_median = UNIQUE_SONGS_DF[available_feature_cols].median()

        # Get artist median for each artist in the comma-separated list
        artist_medians = []
        for artist_name in self._artist.split(', '):
            artist_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(lambda x: artist_name in x)]
            if len(artist_df) > 0:
                artist_median = artist_df[available_feature_cols].median()
                artist_medians.append((artist_name, artist_median))

        # Prepare data for plotting
        data = []
        names = []
        
        # Add all playlists median
        data.append(all_playlists_median)
        names.append('All Playlists')
        
        # Add artist medians
        for artist_name, artist_median in artist_medians:
            data.append(artist_median)
            names.append(artist_name)
        
        # Add song's actual values
        data.append(song_values)
        names.append(self._song)

        # Normalize popularity to 0-1 scale
        for i, df in enumerate(data):
            if 'popularity' in df.index:
                data[i]['popularity'] = round(df['popularity'] / 100, 2)

        # Create the bar chart
        fig = go.Figure()
        
        for i, (name, values) in enumerate(zip(names, data)):
            # Convert to list and handle any NaN values
            y_values = []
            text_values = []
            for col in available_feature_cols:
                val = values[col]
                if pd.isna(val):
                    y_values.append(0)
                    text_values.append('N/A')
            else:
                    y_values.append(round(val, 3))
                    text_values.append(round(val, 3))
            
            fig.add_trace(go.Bar(
                name=name,
                x=available_feature_cols,
                y=y_values,
                text=text_values,
                textposition='auto',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Feature: %{x}<br>' +
                            'Value: %{y}<br>' +
                            '<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title='Song Audio Features vs. Medians',
            xaxis_title='Audio Features',
            yaxis_title='Feature Values',
            barmode='group',
            showlegend=True,
            height=500
        )

        return Markup(fig.to_html(full_html=False))

    def graph_song_percentiles_vs_avg(self):
        """
        Create percentile comparison graph for song audio features.
        
        Improvements made:
        - NaN values are filtered out before percentile calculation
        - Playlist percentiles use method='max' for consistent ranking
        - Artist percentiles use method='min' to avoid 100th percentile for single songs
        - Single song cases are set to 0 (since 0% of songs are below them)
        - Consistent data filtering and validation across all percentile calculations
        """
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        artist_dfs = []
        for a in self._artist.split(', '):
            mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))
            artist_df = UNIQUE_SONGS_DF[mask]
            for col in PERCENTILE_COLS:
                if col in artist_df.columns and not artist_df[col].isna().all():
                    # Filter out NaN values before calculating percentiles
                    valid_data = artist_df[col].dropna()
                    if len(valid_data) > 0:
                        sz = valid_data.size - 1
                        if sz > 0:
                            # Use method='min' for artist percentiles to avoid 100th percentile for single songs
                            artist_df[col + '_percentile'] = valid_data.rank(
                                method='min').apply(lambda x: round(100.0*(x-1)/sz, 2))
                        else:
                            # Single song case - set to 0 (since 0% of songs are below them)
                            artist_df[col + '_percentile'] = 0.0
                    else:
                        artist_df[col + '_percentile'] = 0.0
                else:
                    artist_df[col + '_percentile'] = 0.0
            artist_df = artist_df[artist_df['name'] == self._song]
            if len(artist_df.index) == 0:
                return None
            artist_dfs.append(artist_df)

        # UNIQUE_SONGS_DF already has _percentile columns, so get the matching song by song and artist name
        avg_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
        avg_df = avg_df[avg_df['artist'] == self._artist]

        # Filter for available percentile columns (only show features that have data)
        available_percentile_cols = []
        for col in PERCENTILE_COLS:
            percentile_col = col + '_percentile'
            if percentile_col in avg_df.columns and not avg_df[percentile_col].isna().all():
                available_percentile_cols.append(percentile_col)
        
        if not available_percentile_cols:
            return Markup('<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">No audio features data available for this song</div>')
        
        cols = available_percentile_cols

        dfs = {}
        dfs['All Playlists'] = avg_df[cols]
        for i, j in zip(self._artist.split(', '), [adf[cols] for adf in artist_dfs]):
            dfs[i] = j

        data = []
        # Create x-axis labels from the available percentile columns (remove '_percentile' suffix)
        x_labels = [col.replace('_percentile', '') for col in available_percentile_cols]
        
        for name, df in dfs.items():
            # Handle NaN values safely when converting to text
            y_values = df.iloc[0]
            text_values = []
            hover_values = []
            for val in y_values:
                if pd.isna(val):
                    text_values.append('N/A')
                    hover_values.append('N/A')
                else:
                    # Convert to integer for bar display (no decimals)
                    bar_val = int(round(val, 0))
                    text_values.append(bar_val)
                    # Keep 2 decimal places for hover text (precision)
                    hover_val = round(val, 2)
                    hover_values.append(hover_val)
            
            data.append(go.Bar(
                name=name, 
                x=x_labels, 
                y=y_values, 
                text=text_values, 
                textposition='auto',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Feature: %{x}<br>' +
                             'Percentile: %{customdata}<extra></extra>',
                customdata=hover_values
            ))

        fig = go.Figure(data=data)

        title = 'Percentile of Audio Features by All Playlists & Artist'

        fig.update_layout(barmode='group', title_text=title)

        return Markup(fig.to_html(full_html=False))

    def graph_date_added_to_playlist(self):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF[ALL_SONGS_DF['name'] == self._song]
        df = df[df['artist'] == self._artist]

        # today = datetime.datetime.now().astimezone().date()
        today = datetime.datetime.utcnow().date()
        new = pd.DataFrame({'Task': df['playlist'], 'Start': df['date_added'], 'Finish': [
                           today]*len(df['playlist'])})

        fig = ff.create_gantt(new)
        fig.update_layout(title_text='Timeline Of When ' +
                          self._song + ' Was Added To Playlists')

        return Markup(fig.to_html(full_html=False))

    def graph_top_playlists_by_artist(self, artist_name):
        return shared_graph_top_playlists_by_artist(self._all_songs_df, artist_name)

    def graph_top_songs_by_artist(self, artist_name):
        return shared_graph_top_songs_by_artist(self._unique_songs_df, artist_name)

    def graph_artist_genres(self):
        if len(self._artist_genres) == 1:
            return [', '.join(self._song_genres), False]
        else:
            fig = _venn_diagram_artist_genres(
                self._artist.split(', '), self._artist_genres)
            return [fig, True]

    # What percentage of songs in a playlist or among all playlists have these genres?
    def graph_song_genres_vs_avg(self, playlist=False):
        """Graph percentage of songs with same genres as current artist"""
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df
        # Require genres column to compute percentages
        if 'genres' not in UNIQUE_SONGS_DF.columns or 'genres' not in ALL_SONGS_DF.columns:
            return None
        
        if playlist:
            # New functionality: Percentage of Songs with Same Genres as Artist in Current Playlist
            if not self._playlist:
                return None
            
            # Get all songs in the current playlist
            playlist_mask = ALL_SONGS_DF['playlist'] == self._playlist
            playlist_df = ALL_SONGS_DF[playlist_mask]
            
            if len(playlist_df.index) == 0:
                return None
            
            title = 'Percentage of Songs With Same Genres As ' + self._artist + ' in ' + self._playlist
            xaxis = '% of Songs with Genres'

            percents = []
            song_counts = []
            length = len(playlist_df.index) if len(playlist_df.index) > 0 else 1
            for g in self._song_genres:
                try:
                    mask = playlist_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                    count = len(playlist_df[mask].index)
                    percents.append(count/length*100)
                    song_counts.append(count)
                except Exception:
                    percents.append(0.0)
                    song_counts.append(0)

            # Create hovertext with song counts
            hovertext = [f"{count} songs ({percent:.1f}%)" for count, percent in zip(song_counts, percents)]
            
            series = pd.Series(dict(zip(self._song_genres, percents)))
            return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True, hovertext=hovertext)
        else:
            # Original functionality for overall playlists
            avg_df = ALL_SONGS_DF
            title = 'Percentage of Songs With Same Genres As ' + self._artist + ' Across All Playlists'
            xaxis = '% of Songs with Genres'

            percents = []
            length = len(avg_df.index) if len(avg_df.index) > 0 else 1
            for g in self._song_genres:
                try:
                    mask = avg_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                    percents.append(len(avg_df[mask].index)/length*100)
                except Exception:
                    percents.append(0.0)

            series = pd.Series(dict(zip(self._song_genres, percents)))
            return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True)

    def graph_most_common_artists_with_same_genres(self):
        """Graph most common artists in playlist with same genres as current artist"""
        ALL_SONGS_DF = self._all_songs_df
        # Require genres column to compute percentages
        if 'genres' not in ALL_SONGS_DF.columns:
            return None
        
        if not self._playlist:
            return None
        
        # Get all songs in the current playlist
        playlist_mask = ALL_SONGS_DF['playlist'] == self._playlist
        playlist_df = ALL_SONGS_DF[playlist_mask]
        
        if len(playlist_df.index) == 0:
            return None
        
        # Find artists in the playlist that share genres with the current artist
        shared_genre_artists = {}
        
        # Check if current artist has genres other than "N/A"
        current_has_non_na_genres = any(genre != "N/A" for genre in self._song_genres)
        
        for _, row in playlist_df.iterrows():
            # Split the artist field on commas to handle multiple artists per song
            song_artists = [artist.strip() for artist in row['artist'].split(',')]
            
            # Check if this artist has any shared genres
            try:
                artist_genres = row['genres']
                if artist_genres and isinstance(artist_genres, list):
                    # Flatten the nested genre structure
                    flat_artist_genres = {genre for genre_list in artist_genres for genre in genre_list}
                    
                    # Check for shared genres
                    shared_genres = set(self._song_genres) & flat_artist_genres
                    
                    if shared_genres:
                        # Process each artist in the song separately
                        for artist_name in song_artists:
                            # Skip the current artist
                            if artist_name == self._artist:
                                continue
                            
                            # Include all artists that share any genres, including N/A
                            
                            # This artist shares genres with the current artist
                            if artist_name not in shared_genre_artists:
                                shared_genre_artists[artist_name] = {
                                    'count': 0,
                                    'songs': [],
                                    'shared_genres': list(shared_genres)
                                }
                            
                            shared_genre_artists[artist_name]['count'] += 1
                            shared_genre_artists[artist_name]['songs'].append(row['name'])
            except Exception:
                continue
        
        if not shared_genre_artists:
            # No artists with shared genres found
            fig = go.Figure()
            fig.update_layout(
                title_text=f'Most Common Artists in {self._playlist} with Same Genres as {self._artist}',
                xaxis_title='Number of Songs',
                yaxis_title='Artist',
                annotations=[{
                    'text': f'No artists in {self._playlist} share genres with {self._artist}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                height=300
            )
            return Markup(fig.to_html(full_html=False))
        
        # Create data for the horizontal bar chart
        artists = []
        counts = []
        hovertexts = []
        
        # Sort by count (descending) and take top 10 artists
        sorted_artists = sorted(shared_genre_artists.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        
        for artist_name, data in sorted_artists:
            artists.append(artist_name)
            counts.append(data['count'])
            
            # Create hovertext with shared genres first, then songs below
            shared_genres_text = ', '.join(data['shared_genres'])
            # Cap the songs to MAX_HOVER_ROWS for hovertext
            songs_text = '<br>'.join(data['songs'][:MAX_HOVER_ROWS])
            hovertext = f"<b>Shared Genres:</b> {shared_genres_text}<br><br><b>Songs:</b><br>{songs_text}"
            hovertexts.append(hovertext)
        
        # Create the horizontal bar chart
        fig = go.Figure(data=[go.Bar(
            x=counts,
            y=artists,
            orientation='h',
            text=counts,
            textposition='auto',
            hovertext=hovertexts,
            hovertemplate='<b>%{hovertext}</b><extra></extra>',
            marker_color=COLORS[0]  # Use first color from COLORS
        )])
        
        # Update layout
        fig.update_layout(
            title_text=f'Most Common Artists in {self._playlist} with Same Genres as {self._artist}',
            xaxis_title='Number of Songs',
            yaxis_title='Artist',
            yaxis={'categoryorder': 'total ascending'},
            height=min(max((len(artists)+1)*40, 300), 600),  # Responsive height
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        fig.update_yaxes(automargin=True)
        
        return Markup(fig.to_html(full_html=False))

    def graph_all_artists(self):
        artist_top_graphs = []
        for i in self._artist.split(', '):
            artist_top_playlists_bar = self.graph_top_playlists_by_artist(i)
            artist_top_songs_table = self.graph_top_songs_by_artist(i)
            artist_top_graphs.append(
                (artist_top_playlists_bar, artist_top_songs_table))
        return artist_top_graphs


# My Playlists Page ----------------------------------------------------------------------------------------------

class MyPlaylistsPage(OptimizedPageBase):
    def __init__(self, path, all_songs_df, top_artists, top_songs):
        # Initialize parent class with optimizations
        super().__init__(path, all_songs_df, pd.DataFrame())  # unique_songs_df not needed for this page
        
        self._top_artists = top_artists
        self._top_songs = top_songs

        # Progressive loading: Don't precompute all components upfront
        # Components will be loaded on-demand when requested

    def load_playlists_by_length(self):
        return self.load_component_progressively('playlists_by_length', self._graph_playlists_by_length)
    
    def load_popularity_histogram(self):
        return self.load_component_progressively('popularity_histogram', self._graph_popularity_histogram)
    
    def load_avg_boxplot(self):
        return self.load_component_progressively('avg_boxplot', self._graph_playlists_avg_features_boxplot)
    
    def load_first_last_added(self):
        return self.load_component_progressively('first_last_added', self._graph_playlists_first_last_added)
    
    def load_playlists_by_artists(self):
        return self.load_component_progressively('top_playlists_by_artists', self._graph_top_playlists_by_top_artists)

    def load_playlists_by_songs(self):
        return self.load_component_progressively('top_playlists_by_songs', self._graph_top_playlists_by_top_50_songs)

    def load_playlists_by_explicit(self):
        return self.load_component_progressively('playlists_by_explicit', self._graph_playlists_by_explicit)

    def _graph_playlists_by_length(self):
        labels = ['# Songs', 'Duration (Hours)']
        num_songs = self._graph_num_songs_histogram()
        duration = self._graph_duration_histogram()

        fig = _make_single_subplot(labels, [num_songs, duration],
                                   'Length',
                                   '# Playlists',
                                   'Playlists\' Length by # Songs & Duration', 2)

        return Markup(fig.to_html(full_html=False))
        

    def _graph_num_songs_histogram(self):
        # Use pre-computed playlist counts for better performance
        if hasattr(self, '_playlist_counts'):
            df = self._playlist_counts.reset_index()
        else:
            df = self._all_songs_df['playlist'].value_counts().reset_index()
        
        fig = px.histogram(df, title="Playlists by # of Songs", marginal="rug",
                       hover_data='playlist')

        return fig

    def _graph_duration_histogram(self):
        # Use pre-computed playlist durations for better performance
        if hasattr(self, '_playlist_durations'):
            df = self._playlist_durations.reset_index()
        else:
            df = self._all_songs_df.groupby('playlist')['duration'].sum().reset_index()
        
        df['duration'] = round(df['duration'] / 60 / 60, 2)
        fig = px.histogram(df, title="Playlists by Duration (Hours)", marginal="rug",
                        hover_data='playlist')
        return fig
    
    def _graph_playlists_by_explicit(self):
        labels = ['# Explicit Songs', '% Explicit Songs']
        explicit_num = self._graph_num_explicit()
        explicit_percent = self._graph_num_explicit(percent=True)

        fig = _make_single_subplot(labels, [explicit_num, explicit_percent],
                                   'Amount Explicit',
                                   '# Playlists',
                                   'Playlists by # and % Explicit Songs', 2)

        return Markup(fig.to_html(full_html=False))
        

    def _graph_num_explicit(self, percent=False):
        # Use pre-computed explicit content data for better performance
        if percent:
            if hasattr(self, '_playlist_explicit_percentages'):
                df = self._playlist_explicit_percentages.reset_index()
            else:
                df = self._all_songs_df.groupby('playlist')['explicit'].mean().reset_index()
            df['explicit'] = round(df['explicit'], 2)
        else:
            if hasattr(self, '_playlist_explicit_counts'):
                df = self._playlist_explicit_counts.reset_index()
            else:
                df = self._all_songs_df.groupby('playlist')['explicit'].sum().reset_index()
        
        fig = px.histogram(df, marginal="rug", hover_data='playlist')
        return fig

    def _graph_popularity_histogram(self):
        labels = ['Average Popularity', 'Median Popularity']
        avg_popularity = self._graph_avg_popularity_histogram()
        median_popularity = self._graph_median_popularity_histogram()

        fig = _make_single_subplot(labels, [avg_popularity, median_popularity],
                                   'Popularity',
                                   '# Playlists',
                                   'Playlists by Average and Median Song Popularity', 2)

        return Markup(fig.to_html(full_html=False))

    def _graph_avg_popularity_histogram(self):
        df = self._all_songs_df.groupby('playlist')['popularity'].mean().reset_index()
        fig = px.histogram(df, title="Playlists by Avg Song Popularity", marginal="rug",
                        hover_data='playlist')
        return fig

    def _graph_median_popularity_histogram(self):
        df = self._all_songs_df.groupby('playlist')['popularity'].median().reset_index()
        fig = px.histogram(df, title="Playlists by Median Song Popularity", marginal="rug",
                        hover_data='playlist')
        return fig

    def _graph_playlists_avg_features_boxplot(self):
        try:
            # Filter for available FEATURE_COLS only (columns that exist and have non-null values)
            available_feature_cols = [col for col in FEATURE_COLS if col in self._all_songs_df.columns]
            
            if not available_feature_cols:
                # Fallback to basic features if no audio features available
                available_feature_cols = ['popularity', 'duration']
            
            # Handle playlist column - it contains lists, so we need to explode it first
            # Create a copy of the DataFrame and explode the playlist column
            working_df = self._all_songs_df.copy()
            working_df = working_df.explode('playlist')
            
            # Now use groupby on the exploded DataFrame
            df = working_df.groupby('playlist')[available_feature_cols].median().reset_index()

            x_data = available_feature_cols
            y_data = []

            # Normalize popularity to 0-1 scale if it exists
            if 'popularity' in df.columns:
                df['popularity'] = df['popularity']/100
            
            # Prepare y_data - each feature gets a list of median values from all playlists
            y_data = [df[col] for col in available_feature_cols]

            # Create feature-specific hover text for each trace
            hover_text = []
            for feature_idx, feature_name in enumerate(available_feature_cols):
                feature_hover_text = []
                
                for _, row in df.iterrows():
                    playlist_name = row['playlist']
                    
                    # Get the original songs for this playlist to calculate coverage
                    playlist_songs = working_df[working_df['playlist'] == playlist_name]
                    
                    # Count UNIQUE songs (not exploded rows)
                    unique_song_ids = playlist_songs['id'].unique()
                    total_songs = len(unique_song_ids)
                    
                    # Count songs with THIS SPECIFIC audio feature
                    songs_with_this_feature = 0
                    
                    for song_id in unique_song_ids:
                        song_rows = playlist_songs[playlist_songs['id'] == song_id]
                        if feature_name in song_rows.columns and song_rows[feature_name].notna().any():
                            songs_with_this_feature += 1
                    
                    # Calculate percentage of songs with this specific feature
                    coverage_pct = (songs_with_this_feature / total_songs * 100) if total_songs > 0 else 0
                    
                    # Create hover text with exactly 4 rows for this specific feature
                    hover_line = f"<b>{playlist_name}</b><br>"
                    hover_line += f"Total Songs: {total_songs}<br>"
                    hover_line += f"Songs with {feature_name.title()}: {songs_with_this_feature}<br>"
                    hover_line += f"% of Songs with {feature_name.title()}: {coverage_pct:.1f}%"
                    
                    feature_hover_text.append(hover_line.strip())
                
                hover_text.append(feature_hover_text)

            title = 'My Playlists\' Median Audio Features'
            xaxis = 'Song Feature'
            yaxis = 'Level (Low -> High)'
            
            # Create the boxplot manually since _boxplot doesn't support per-trace hovertext
            fig = go.Figure()
            
            for i, (feature_name, feature_data) in enumerate(zip(available_feature_cols, y_data)):
                fig.add_trace(go.Box(
                    y=feature_data,
                    name=feature_name,
                    boxpoints='all',
                    text=hover_text[i],  # Use the specific hovertext for this feature
                    marker_color=COLORS[i % len(COLORS)]
                ))
            
            fig.update_layout(
                title_text=title,
                xaxis_title=xaxis,
                yaxis_title=yaxis
            )
            
            return Markup(fig.to_html(full_html=False))
        
            return fig
        except Exception as e:
            # Fallback to a simple error message if the boxplot fails
            error_html = f'<div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">Error generating audio features boxplot: {str(e)}</div>'
            return Markup(error_html)

    def _graph_playlists_first_last_added(self):
        df = self._all_songs_df.groupby('playlist')['date_added'].agg([('CreatedDate', 'min'), ('LastAddedDate', 'max')]).reset_index()
        # df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
        df = df.sort_values(by='CreatedDate')
        fig = px.timeline(df, x_start='CreatedDate', x_end='LastAddedDate', y='playlist', color='playlist')

        fig.update_layout(title_text='My Playlists\' Timelines By First and Last Date Added')
        fig.update_layout(xaxis_title='Date')
        fig.update_layout(height=len(df)*30)

        final = Markup(fig.to_html(full_html=False))
        return final
    
    def _graph_top_playlists_by_top_50_songs(self):
        ALL_SONGS_DF = self._all_songs_df

        final = []
        for time_range in [0, 1, 2]:
            dicty = self._top_songs[time_range]
            playlist_dict = defaultdict(int)
            for name, playlist in zip(ALL_SONGS_DF['name'], ALL_SONGS_DF['playlist']):
                if name in dicty.keys():
                    playlist_dict[playlist] += 1

            final.append(playlist_dict)
            
        df = pd.DataFrame(final, index=[
                        '# Top Short Term Songs', '# Top Medium Term Songs', '# Top Long Term Songs'])
        df = df.T.fillna(0)
        df['Playlist'] = df.index
        fig = px.scatter(df, x="# Top Short Term Songs", y="# Top Medium Term Songs",
                            size="# Top Long Term Songs", hover_name='Playlist')
        title = 'Top Playlists by Number of Top 50 Songs In Them'
        fig.update_layout(title_text=title)
        return Markup(fig.to_html(full_html=False))


    def _graph_top_playlists_by_top_n_artists(self, artists=10):
        ALL_SONGS_DF = self._all_songs_df

        final = []
        for time_range in [0, 1, 2]:
            dicty = self._top_artists[time_range]

            playlist_dict = defaultdict(int)
            for name, playlist in zip(ALL_SONGS_DF['artist'], ALL_SONGS_DF['playlist']):
                found = False
                for n in name.split(', '):
                    if n in dicty[:artists]:
                        if not found:
                            playlist_dict[playlist] += 1
                            found = True

            final.append(playlist_dict)

        df = pd.DataFrame(final, index=['# Top Short Term Artist Songs',
                            '# Top Medium Term Artist Songs', '# Top Long Term Artist Songs'])
        df = df.T.fillna(0)
        df['Playlist'] = df.index

        fig = px.scatter(df, x="# Top Short Term Artist Songs", y="# Top Medium Term Artist Songs",
                            size="# Top Long Term Artist Songs", hover_name='Playlist')
        return fig


    def _graph_top_playlists_by_top_artists(self):
        labels = ['Top 10', 'Top 25', 'Top 50']
        fig = subplots.make_subplots(rows=1, cols=1)

        top_10 = self._graph_top_playlists_by_top_n_artists()
        top_25 = self._graph_top_playlists_by_top_n_artists(25)
        top_50 = self._graph_top_playlists_by_top_n_artists(50)
        figs = [top_10, top_25, top_50]

        fig = _make_single_subplot(labels, figs, '# Songs From Short Term Top Artists',
                                   '# Songs From Med. Term Top Artists',
                                   'Top Playlists by Number of Top N Artists\' Songs')

        return Markup(fig.to_html(full_html=False))
        