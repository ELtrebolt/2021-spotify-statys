# visualization.py = contains Page Classes which have some unique and shared functions to make graphs

# Imports --------------------------------------------------------------------
import pickle
from collections import defaultdict
import datetime
from re import A
import numpy as np
import pandas as pd
from markupsafe import Markup
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

# Constants -------------------------------------------------------------------

PERCENTILE_COLS = ['popularity', 'duration']
FEATURE_COLS = ['popularity']

LABEL_CUTOFF_LENGTH = 25
MAX_HOVER_ROWS = 10
MAX_HBARS = 10
TIME_RANGE_DICT = {0: ['Last 4 Weeks', 'short_rank'], 1: [
    'Last 6 Months', 'med_rank'], 2: ['All Time', 'long_rank']}
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
          '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
TOP_RANK_TABLE_MIN_HEIGHT = 300

# Helper Functions -----------------------------------------------------------------------------------------

def _shorten_names(listy):
    return [i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in listy[:MAX_HOVER_ROWS]]

def _h_bar(series, title=None, xaxis=None, yaxis=None, percents=False,
           long_names=False, hovertext=None, to_html=True, name=None, color=None, markup=True, fixHeight=False):
    if percents:
        texty = [str(round(i, 2)) + '%' for i in series]
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


def _venn_diagram_artist_genres(artists, genres):
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


def _dump(path, obj):
    with open(path, 'wb') as f:   
        pickle.dump(obj, f)


def _load(path):
    with open(path, 'rb') as f:  
        return pickle.load(f)

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
    else:
        fig = _timeline_trace(df, continuous, playlists, bar)

    if not to_html:
        return fig

    if song_name and artists and song_in_playlist:
        for song_date in song_dates:
            fig.add_vline(x=song_date, line_width=3,
                          line_dash="dash", line_color="green")

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


def shared_graph_count_timelines(all_songs_df, title, playlists=None, artists=None, to_html=True):
    today = datetime.datetime.now().astimezone()
    line_timeline = shared_graph_count_timeline(
        all_songs_df, continuous=False, playlists=playlists, artists=artists, to_html=False)
    continuous_timeline = shared_graph_count_timeline(
        all_songs_df, continuous=True, playlists=playlists, artists=artists, to_html=False)
    bar_timeline = shared_graph_count_timeline(
        all_songs_df, bar=True, playlists=playlists, artists=artists, to_html=False)

    labels = ["Line", "Continuous", 'Bar']
    fig = subplots.make_subplots(rows=1, cols=1, vertical_spacing=.05,
                                 subplot_titles=(title))

    # Multi-colors for 1 Time Range = Short, Med, Long
    length = len(line_timeline.data)
    for i in range(length):
        fig.add_trace(line_timeline.data[i], 1, 1)
        fig.add_trace(continuous_timeline.data[i], 1, 1)
        fig.add_trace(bar_timeline.data[i], 1, 1)

    # Create buttons for drop down menu
    buttons = []
    for i, label in enumerate(labels):
        visibility = [i == j for j in range(len(labels))]
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

    # hoverlabel_font_color='white'
    fig.update_layout(updatemenus=updatemenus,
                      showlegend=True, title=title, yaxis_title='# Songs', xaxis_title='Date', barmode='stack')

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

    # Add vertical lines for current date
    first_date = all_songs_df['date_added'].sort_values(
        ascending=True).iloc[0]
    first_year = str(first_date)[:4]
    current_year = str(today.date())[:4]
    today = str(today.date())[5:].split('-')
    month = int(today[0])
    day = int(today[1])
    for year in range(int(first_year), int(current_year)+1):
        fig.add_vline(x=datetime.datetime(year, month, day), line_width=3,
                      line_dash="dash", line_color="green")

    # Only show 1st Trace on page-load
    Ld = len(fig.data)
    for k in range(Ld):
        if k % 3 == 0:
            fig.update_traces(visible=True, selector=k)
        else:
            fig.update_traces(visible=False, selector=k)

    if to_html:
        return Markup(fig.to_html(full_html=False))
    else:
        return fig


def shared_graph_top_playlists_by_artist(ALL_SONGS_DF, artist_name):
    mask = ALL_SONGS_DF['artist'].apply(lambda x: artist_name in x.split(', '))
    df = ALL_SONGS_DF[mask]

    series = df['playlist'].value_counts(ascending=True).head(10)
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

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['# Playlists Song Is In', 'Artist Song(s)'],
            line_color='darkslategray',
            fill_color='royalblue',
            font=dict(color='white', size=18),
            height=40
        ),
        cells=dict(
            values=[list(d.keys()), list(d.values())],
            line_color='darkslategray',
            align='center',
            font_size=[18, 14],
            height=30)
    )
    ])

    fig.update_layout(autosize=True, title_text='Most Common Songs For Artist: ' + artist_name,
                      xaxis_title="Number of Playlists The Artist's Song Is In",
                      height=min(max((len(df)+1)*25, 300), 500))

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


# Currently Playing Page ----------------------------------------------------------------------------------------------
class CurrentlyPlayingPage():
    def __init__(self, song, artist, playlist, all_songs_df, unique_songs_df):
        self._song = song
        self._artist = artist
        self._playlist = playlist

        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        song_df = self._unique_songs_df[self._unique_songs_df['artist'] == artist]
        self._song_df = song_df[song_df['name'] == song]

        # Genres may be missing if Setup 2 not completed; guard accordingly
        has_genres_col = 'genres' in self._song_df.columns
        if len(self._song_df.index) > 0 and has_genres_col:
            try:
                self._artist_genres = self._song_df.iloc[0]['genres']
                self._song_genres = list({j for i in self._artist_genres for j in i})
            except Exception:
                self._artist_genres = []
                self._song_genres = None
        else:
            self._artist_genres = []
            self._song_genres = None

    def graph_top_rank_table(self):
        return shared_graph_top_rank_table(self._song_df)

    def graph_song_features_vs_avg(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        song_df = self._song_df[FEATURE_COLS]

        if self._playlist:
            playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']
                                       == self._playlist][FEATURE_COLS]
        artist_dfs = [UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(lambda x: i in x)]
                        [FEATURE_COLS] for i in self._artist.split(', ')]
        avg_df = UNIQUE_SONGS_DF[FEATURE_COLS]

        if self._playlist:
            dfs = [playlist_df, avg_df, song_df]
        else:
            dfs = [avg_df, song_df]

        for df in dfs:
            df['popularity'] = round(df['popularity']/100, 2)
        dfs = [df.median(axis=0) for df in dfs]

        for df in artist_dfs:
            df['popularity'] = round(df['popularity']/100, 2)
        artist_dfs = [df.median(axis=0) for df in artist_dfs]

        # add song values to last so features and percentiles avgs match colors
        song_vals = dfs[-1]
        dfs = dfs[:-1]
        dfs += artist_dfs
        dfs.append(song_vals)

        if self._playlist:
            names = [self._playlist, 'All Playlists'] + self._artist.split(', ') + [self._song]
        else:
            names = ['All Playlists'] + self._artist.split(', ') + [self._song]

        if self._playlist:
            title = 'Song Audio Features vs. Median Song from Playlist, All Playlists, & Artist'
        else:
            title = 'Song Audio Features vs. Median Song from All Playlists & Artist'

        data = []
        for name, df in zip(names, dfs):
            data.append(go.Bar(name=name, x=FEATURE_COLS, y=[df.iloc[0]], text=[df.iloc[0]], textposition='auto'))
        fig = go.Figure(data=data)
        fig.update_layout(barmode='group', title_text=title)
        return Markup(fig.to_html(full_html=False))

    def graph_song_percentiles_vs_avg(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        # Since playlist and artist df length will vary, make new percentile cols for them and then isolate song
        if self._playlist:
            playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] == self._playlist]

            exists_df = playlist_df[playlist_df['name'] == self._song]
            exists_df = exists_df[exists_df['artist'] == self._artist]
            if len(exists_df) == 0:
                song_df = ALL_SONGS_DF[ALL_SONGS_DF['name'] == self._song]
                song_df = song_df[song_df['artist'] == self._artist]

                playlist_df = pd.concat([playlist_df, song_df])

            for col in PERCENTILE_COLS:
                sz = playlist_df[col].size-1
                playlist_df[col + '_percentile'] = playlist_df[col].rank(
                    method='max').apply(lambda x: 100.0*(x-1)/sz)
            playlist_df = playlist_df[(playlist_df['name'] == self._song) & (playlist_df['artist'] == self._artist)]

        artist_dfs = []
        for a in self._artist.split(', '):
            mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))
            artist_df = UNIQUE_SONGS_DF[mask]
            for col in PERCENTILE_COLS:
                sz = artist_df[col].size-1
                if sz > 0:
                    artist_df[col + '_percentile'] = artist_df[col].rank(
                        method='max').apply(lambda x: 100.0*(x-1)/sz)
                else:
                    artist_df[col + '_percentile'] = [0]
            artist_df = artist_df[artist_df['name'] == self._song]
            if len(artist_df.index) == 0:
                return None
            artist_dfs.append(artist_df)

        # UNIQUE_SONGS_DF already has _percentile columns, so get the matching song by song and artist name
        avg_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
        avg_df = avg_df[avg_df['artist'] == self._artist]

        cols = [i + '_percentile' for i in PERCENTILE_COLS]

        dfs = {}
        if self._playlist:
            dfs[self._playlist] = playlist_df[cols]
        dfs['All Playlists'] = avg_df[cols]
        for i, j in zip(self._artist.split(', '), [adf[cols] for adf in artist_dfs]):
            dfs[i] = j

        data = []
        for name, df in dfs.items():
            data.append(go.Bar(name=name, x=PERCENTILE_COLS, y=df.iloc[0], text=df.iloc[0].astype(int), textposition='auto'))

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
            return None 
        return shared_graph_count_timeline(self._all_songs_df, playlists=[self._playlist], song_name=self._song, artists=[self._artist])

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
        if self._song_genres is None:
            return None
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df
        # Require genres column to compute percentages
        if 'genres' not in UNIQUE_SONGS_DF.columns or 'genres' not in ALL_SONGS_DF.columns:
            return None
        if playlist:
            mask = UNIQUE_SONGS_DF['playlist'].apply(lambda x: self._playlist in x)
            avg_df = UNIQUE_SONGS_DF[mask]
            title = 'Percentage of Songs With Same Genres As ' + self._artist + ' in ' + self._playlist
            xaxis = '% of Songs in ' + self._playlist
        else:
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

    def graph_all_artists(self):
        artist_top_graphs = []
        for i in self._artist.split(', '):
            artist_top_playlists_bar = self.graph_top_playlists_by_artist(i)
            artist_top_songs_table = self.graph_top_songs_by_artist(i)
            artist_top_graphs.append(
                (artist_top_playlists_bar, artist_top_songs_table))
        return artist_top_graphs


# Home Page ----------------------------------------------------------------------------------------------

class HomePage():
    def __init__(self, path, all_songs_df, unique_songs_df):
        self._today = datetime.datetime.now().astimezone()
        self._path = path
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        # Generate page fragments in-memory
        self._on_this_date = self._graph_on_this_date()
        self._timeline = self._graph_count_timeline()
        self._last_added = self._graph_last_added()
        self._totals = self._get_library_totals()

    def load_on_this_date(self):
        return self._on_this_date

    def load_timeline(self):
        return self._timeline

    def load_last_added(self):
        return self._last_added

    def load_totals(self):
        return self._totals

    def load_first_times(self):
        # Always return a list so templates that iterate do not split characters
        if hasattr(self, '_first_times') and isinstance(self._first_times, list) and len(self._first_times) > 0:
            return self._first_times
        return ['No recorded data of adding songs to a playlist on this date']

    def _graph_on_this_date(self):
        '''Output1 = Table = Year | Playlist | Songs Added
        Output2 = List = You created playlist / added song from artist for the first time'''
        ALL_SONGS_DF = self._all_songs_df
        today = str(self._today.date())[5:]
        df = ALL_SONGS_DF[ALL_SONGS_DF['date_added'].apply(lambda x: today in x)]
        first_times = []

        df['year'] = [i[:4] for i in df['date_added']]
        df = df[['name', 'artist', 'playlist', 'year']]

        if len(df.index) == 0:
            first_times.append('No recorded data of adding songs to a playlist on this date')
        else:
            for a in {j for i in df['artist'] for j in i.split(', ')}:
                artist_df = ALL_SONGS_DF[ALL_SONGS_DF['artist'].apply(
                    lambda x: a in x.split(', '))]
                first_date = artist_df['date_added'].sort_values(
                    ascending=True).iloc[0]
                if str(first_date)[5:] == today:
                    years_ago = int(str(self._today.date())[
                                    :4])-int(str(first_date)[:4])
                    first_times.append(
                        (years_ago, 'You first added a song from the artist ' + a + ' ' + str(years_ago) + ' years ago today!'))
            first_times = [i[1] for i in sorted(first_times, key = lambda x: x[0], reverse=True)]
        
            for p in df['playlist'].unique():
                playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist'] == p]
                first_date = playlist_df['date_added'].sort_values(
                    ascending=True).iloc[0]
                if str(first_date)[5:] == today:
                    years_ago = int(str(self._today.date())[
                                    :4])-int(str(first_date)[:4])
                    first_times.append(
                        'You created the playlist ' + p + ' ' + str(years_ago) + ' years ago today!')

        self._first_times = first_times
        
        df = df.groupby(['playlist', 'year'], as_index=False)[
            ['name', 'artist']].agg(lambda x: ', '.join(x))
        df = df.sort_values(by='year')
        try:
            df.columns=df.columns.str.strip()
            years = df['year'].to_list()
            playlists = df['playlist'].to_list()
            names = df['name'].to_list()
            artists = [', '.join(list(dict.fromkeys((i.split(', '))))) for i in df['artist']]
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
                values=[years, playlists, names, artists],
                line_color='darkslategray',
                align='center',
                font_size=[18, 14],
                height=30)
        )
        ])

        return Markup(fig.to_html(full_html=False))


    def _graph_count_timeline(self):
        fig = shared_graph_count_timelines(
            self._all_songs_df, title='<b>Timeline of Adding Songs to Playlists</b>', to_html=False)
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

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Playlist', 'Songs Added'],
                line_color='darkslategray',
                fill_color='royalblue',
                font=dict(color='white', size=18),
                height=40
            ),
            cells=dict(
                values=[list(df['playlist']), [', '.join(i)
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

        overall_data = [len(ALL_SONGS_DF.index), len(UNIQUE_SONGS_DF),
                        len(ALL_SONGS_DF['playlist'].unique()), len(
                            UNIQUE_SONGS_DF['artist'].unique()),
                        len(UNIQUE_SONGS_DF['album'].unique())]

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
            subset['playlists'] = [', '.join(i) for i in subset['playlist']]
            subset = subset[['name', 'num_playlists', 'playlists']]

            title = 'Top ' + str(first+1)  + '-' + str(last) + ' Most Common Songs Across ' + \
                str(len(self._all_songs_df['playlist'].unique())) + ' Playlists'
            bar = px.bar(subset, x='num_playlists', y='name', orientation='h', title=title, text='num_playlists', hover_data="playlists")
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
        # df['link'] = [REDIRECT_URI + yaxis.lower() + 's/' + i for i in df[yaxis.lower()]]

        graphs = []
        for i in range(buttons):
            first = i*10
            last = (i+1)*10
            subset = df.iloc[first:last].sort_values(by='Count')

            title = 'Top ' + str(first+1)  + '-' + str(last) + f' Most Common {yaxis}s Across ' + \
                str(len(self._all_songs_df['playlist'].unique())) + ' Playlists'
            if albums:
                bar = px.bar(subset, x='Count', y=yaxis, orientation='h', title=title, text='Count', hover_data='Artist')
            else:
                bar = px.bar(subset, x='Count', y=yaxis, orientation='h', title=title, text='Count')
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

        labels = []
        for i in range(buttons):
            start = i * 10 + 1
            end = (i + 1) * 10
            labels.append(f"{start}-{end}")

        final = _make_single_subplot(labels, graphs, xaxis, yaxis, 
                'Top ' + str(buttons*10) + f' Most Common {yaxis}s Across ' + \
                str(len(ALL_SONGS_DF['playlist'].unique())) + ' Playlists')

        if albums:
            return Markup(final.to_html(full_html=False))
        else:
            return Markup(final.to_html(full_html=False))


# Analyze Multiple Playlists Page ----------------------------------------------------------------------------------------------

class AnalyzePlaylistsPage():
    def __init__(self, playlists, all_songs_df, unique_songs_df):
        self._playlists = playlists
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

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
        return shared_graph_count_timelines(
            self._all_songs_df, title='Timeline of When Songs Were Added to Playlists',
            playlists=self._playlists)

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

class AnalyzePlaylistPage():
    def __init__(self, playlist, all_songs_df, unique_songs_df):
        self._playlist = playlist
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        self._playlist_df = self._all_songs_df[self._all_songs_df['playlist']
                                               == self._playlist]

    def graph_count_timeline(self):
        return shared_graph_count_timelines(self._all_songs_df, title='Timeline of When Songs Were Added to ' + self._playlist,
                                            playlists=[self._playlist])

    def graph_playlist_genres(self):
        df = self._playlist_df
        genres = defaultdict(int)
        for i in df['genres']:
            unique_genres = {y for x in i for y in x}
            for j in unique_genres:
                genres[j] += 1
        total = len(df.index)
        relative = {k: v/total*100 for k,
                    v in sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]}
        genres = {k: str(v) for k, v in sorted(
            genres.items(), key=lambda x: x[1], reverse=True)}

        series = pd.Series(relative)
        title = 'Most Common Genres For Playlist: ' + self._playlist
        return _h_bar(series, title=title, xaxis='% of Songs', yaxis='Artist Genre', percents=True,
                      hovertext=[genres[i] + ' Songs' for i in relative])

    def graph_top_artists(self, top_n=10):
        df = self._playlist_df.copy()

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

        grouped = df.groupby('album')['name'].agg(list)
        sorted_dict = dict(sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:top_n])
        series = pd.Series({i:len(j) for i,j in sorted_dict.items()})

        title = 'Most Common Albums For Playlist: ' + self._playlist
        return _h_bar(series, title=title, yaxis='Album', xaxis='Number of Songs', long_names=True,
                       hovertext=['<br>'.join(_shorten_names(i)) for i in sorted_dict.values()])

    def graph_similar_playlists(self, top_n=10):
        UNIQUE_SONGS_DF = self._unique_songs_df
        mask = UNIQUE_SONGS_DF['playlist'].apply(lambda x: self._playlist in x)
        df = UNIQUE_SONGS_DF[mask]

        other_playlists = {j for i in df['playlist']
                           for j in i if j != self._playlist}
        dicty = dict()
        for p in other_playlists:
            mask = df['playlist'].apply(lambda x: p in x)
            df2 = df[mask]
            dicty[p] = len(df2.index)

        total = len(df.index)
        relative = {k: v/total*100 for k,
                    v in sorted(dicty.items(), key=lambda x: x[1], reverse=True)[:10]}
        counts = {k: str(v) for k, v in sorted(
            dicty.items(), key=lambda x: x[1], reverse=True)}

        series = pd.Series(relative)
        title = 'Most Similar Playlists By Songs Shared: ' + self._playlist
        return _h_bar(series, title=title, xaxis='% of ' + self._playlist + ' Songs', yaxis='Playlist', percents=True,
                      hovertext=[counts[i] + ' Songs' for i in relative])

    def graph_song_features_boxplot(self):
        df = self._playlist_df

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

        dicty = defaultdict(list)
        artists = self._top_artists[time_range]
        pops = self._top_artists_pop[time_range]
        for a,p in zip(artists, pops):
            dicty['popularity'].append(p/100)

        y_data = list(dicty.values())

        title = 'Top 50 Artists (' + \
            TIME_RANGE_DICT[time_range][0] + ') Median Audio Features'
        text = [a + '<br>Rank: ' + str(i) for i, a in enumerate(artists, 1)]
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
        unique_genres = {k for i in df['genres'] for j in i for k in j}
        for g in unique_genres:
            mask = df['genres'].apply(lambda x: g in {z for y in x for z in y})
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
        fig = subplots.make_subplots(rows=4, cols=1, vertical_spacing=.05,
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
            # True False False for 1st False True False for 2nd False False True for 3rd
            visibility = [i == j for j in range(len(labels))]
            # 8 Feature Cols + 8 Feature Cols + 1 Bar Chart + 1 Bar Chart
            visibility = [k for j in [[i]*4 for i in visibility]
                          for k in j] + [False]*12
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
            args=[{'visible': [False]*12+[True]*12},
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

        # Only show 1st 18 Traces on page-load
        Ld = len(fig.data)
        num = 4
        for k in range(num, Ld):
            fig.update_traces(visible=False, selector=k)

        # edit axis labels
        fig['layout']['xaxis']['title'] = 'Song Features'
        fig['layout']['xaxis2']['title'] = 'Artist Features'
        fig['layout']['xaxis3']['title'] = '% of Top 50 Songs'
        fig['layout']['xaxis4']['title'] = '% of Top 50 Artists'

        fig['layout']['yaxis']['title'] = 'Level (Low-High)'
        fig['layout']['yaxis2']['title'] = 'Level (Low-High)'
        fig['layout']['yaxis3']['title'] = 'Genre'
        fig['layout']['yaxis4']['title'] = 'Genre'

        return Markup(fig.to_html(full_html=False))


# Analyze Single Artist Page ----------------------------------------------------------------------------------------------

class AnalyzeArtistPage():
    def __init__(self, artist, all_songs_df, unique_songs_df):
        self._artist = artist
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        mask = all_songs_df['artist'].apply(lambda x: artist in x.split(', '))
        self._artist_df = all_songs_df[mask]
        # self._pure_artist_df = all_songs_df[all_songs_df['artist'] == artist]

        row = unique_songs_df[unique_songs_df['artist'].apply(lambda x: artist in x.split(', '))].iloc[0]
        ind = row['artist'].split(', ').index(artist) if type(row['artists_short_rank']) != int and row['artists_short_rank']!='N/A' else -1
        self._artist_ranks = [row['artists_short_rank'].split(', ')[ind] if ind != -1 else row['artists_short_rank'],
                              row['artists_med_rank'].split(', ')[ind] if ind != -1 else row['artists_med_rank'], 
                              row['artists_long_rank'].split(', ')[ind] if ind != -1 else row['artists_long_rank']]
        self._artist_genres = row['genres'][ind] if ind != -1 else row['genres'][0]

        self._user_playlists = all_songs_df['playlist'].unique()

    def artist_genres(self):
        return self._artist + ' Genres: ' + ', '.join(self._artist_genres)

    def graph_count_timeline(self):
        return shared_graph_count_timelines(
            self._all_songs_df, title='Timeline of Adding ' + self._artist + ' Songs to Playlists', artists=[self._artist])

    def graph_top_rank_table(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(
            lambda x: self._artist in x.split(', '))]

        drop_df = df[(df['songs_short_rank'] == 'N/A') &
                     (df['songs_med_rank'] == 'N/A') & (df['songs_long_rank'] == 'N/A')]
        df = df.drop(drop_df.index)

        df['songs_short_rank'].replace({'N/A': 51}, inplace=True)
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
        for g in self._artist_genres:
            for p in self._user_playlists:
                df = self._unique_songs_df[self._unique_songs_df['playlist'].apply(
                    lambda x: p in x)]
                mask = df['genres'].apply(
                    lambda x: g in {z for y in x for z in y})
                percents[p].append(len(df[mask])/len(df)*100)
                names[g][p] = [str(len(df[mask])) + ' Songs'] + df[mask].sample(min(len(df[mask]),MAX_HOVER_ROWS))['name'].to_list()

        for i in percents:
            percents[i].append(sum(percents[i]))

        df = pd.DataFrame.from_dict(
            percents, orient='index')
        df.columns = self._artist_genres + ['sum']
        df = df.sort_values(by='sum', ascending=False).head(10)

        for g in self._artist_genres:
            series = pd.Series(
                dict(zip(df.index, df[g])))
            go_bar = _h_bar(series, name=g, 
                            hovertext=['<br>'.join(names[g][i]) for i in df.index],
                            percents=True, to_html=False)
            data.append(go_bar)

        fig = go.Figure(data=data)
        fig.update_layout(barmode='stack', yaxis={
            'categoryorder': 'total ascending'}, yaxis_title='Playlist', xaxis_title='% of Songs With Genre', title='Playlists Most Likely To Have Songs Similar to ' + self._artist)
        return Markup(fig.to_html(full_html=False))


# Analyze Multiple Artists Page ----------------------------------------------------------------------------------------------

class AnalyzeArtistsPage():
    def __init__(self, artists, all_songs_df, unique_songs_df):
        self._artists = artists
        self._all_songs_df = pd.DataFrame(all_songs_df)
        self._unique_songs_df = pd.DataFrame(unique_songs_df)

        self._artist_genres = []
        for a in artists:
            row = unique_songs_df[unique_songs_df['artist'].apply(lambda x: a in x.split(', '))].iloc[0]
            ind = row['artist'].split(', ').index(a) if type(row['artists_short_rank']) != int and row['artists_short_rank']!='N/A' else -1
            self._artist_genres.append(row['genres'][ind] if ind != -1 else row['genres'][0])

    def graph_artist_timelines(self):
        return shared_graph_count_timelines(
            self._all_songs_df, title='Timeline of When Artists\' Songs Were Added to Any Playlist',
            artists=self._artists)

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

            df['songs_short_rank'].replace({'N/A': 51}, inplace=True)
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
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        artist_dfs = [UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist'].apply(lambda x: i in x)]
                            [FEATURE_COLS] for i in self._artist.split(', ')]
        avg_df = UNIQUE_SONGS_DF[FEATURE_COLS]
        dfs = [avg_df, self._song_df[FEATURE_COLS]]

        for df in dfs:
            df['popularity'] = df['popularity']/100
        dfs = [df.median(axis=0) for df in dfs]

        for df in artist_dfs:
            df['popularity'] = df['popularity']/100
        artist_dfs = [df.median(axis=0) for df in artist_dfs]

        # add song values to last so features and percentiles avgs match colors
        song_vals = dfs[-1]
        dfs = dfs[:-1]
        dfs += artist_dfs
        dfs.append(song_vals)

        names = ['All Playlists'] + self._artist.split(', ') + [self._song]
        title = 'Song Audio Features vs. Median Song from All Playlists & Artist'

        data = []
        for name, df in zip(names, dfs):
            data.append(go.Bar(name=name, x=FEATURE_COLS, y=[df.iloc[0]], text=[df.iloc[0]], textposition='auto'))
        fig = go.Figure(data=data)
        fig.update_layout(barmode='group', title_text=title)
        return Markup(fig.to_html(full_html=False))

    def graph_song_percentiles_vs_avg(self):
        UNIQUE_SONGS_DF = self._unique_songs_df

        artist_dfs = []
        for a in self._artist.split(', '):
            mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: a in x.split(', '))
            artist_df = UNIQUE_SONGS_DF[mask]
            for col in PERCENTILE_COLS:
                sz = artist_df[col].size-1
                if sz > 0:
                    artist_df[col + '_percentile'] = artist_df[col].rank(
                        method='max').apply(lambda x: 100.0*(x-1)/sz)
                else:
                    artist_df[col + '_percentile'] = [0]
            artist_df = artist_df[artist_df['name'] == self._song]
            if len(artist_df.index) == 0:
                return None
            artist_dfs.append(artist_df)

        # UNIQUE_SONGS_DF already has _percentile columns, so get the matching song by song and artist name
        avg_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name'] == self._song]
        avg_df = avg_df[avg_df['artist'] == self._artist]

        cols = [i + '_percentile' for i in PERCENTILE_COLS]

        dfs = {}
        dfs['All Playlists'] = avg_df[cols]
        for i, j in zip(self._artist.split(', '), [adf[cols] for adf in artist_dfs]):
            dfs[i] = j

        data = []
        for name, df in dfs.items():
            data.append(go.Bar(name=name, x=PERCENTILE_COLS, y=df.iloc[0], text=df.iloc[0].astype(int), textposition='auto'))

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
        if self._song_genres != None:
            UNIQUE_SONGS_DF = self._unique_songs_df
            ALL_SONGS_DF = self._all_songs_df
            avg_df = ALL_SONGS_DF
            title = 'Percentage of Songs With Same Genres As ' + \
                self._artist + ' Across All Playlists'
            xaxis = '% of Songs with Genres'

            percents = []
            length = len(avg_df.index)
            for g in self._song_genres:
                mask = avg_df['genres'].apply(
                    lambda x: g in {z for y in x for z in y})
                percents.append(len(avg_df[mask].index)/length*100)

            series = pd.Series(dict(zip(self._song_genres, percents)))
            return _h_bar(series, title=title, xaxis=xaxis, percents=True, fixHeight=True)
        return None

    def graph_all_artists(self):
        artist_top_graphs = []
        for i in self._artist.split(', '):
            artist_top_playlists_bar = self.graph_top_playlists_by_artist(i)
            artist_top_songs_table = self.graph_top_songs_by_artist(i)
            artist_top_graphs.append(
                (artist_top_playlists_bar, artist_top_songs_table))
        return artist_top_graphs


# My Playlists Page ----------------------------------------------------------------------------------------------

class MyPlaylistsPage():
    def __init__(self, path, all_songs_df, top_artists, top_songs):
        self._path = path
        self._all_songs_df = pd.DataFrame(all_songs_df)

        self._top_artists = top_artists
        self._top_songs = top_songs

        # Precompute page fragments in-memory
        self._playlists_by_length = self._graph_playlists_by_length()
        self._avg_boxplot = self._graph_playlists_avg_features_boxplot()
        self._first_last_added = self._graph_playlists_first_last_added()
        self._top_playlists_by_songs = self._graph_top_playlists_by_top_50_songs()
        self._top_playlists_by_artists = self._graph_top_playlists_by_top_artists()
        self._playlists_by_explicit = self._graph_playlists_by_explicit()

    def load_playlists_by_length(self):
        return self._playlists_by_length
    
    def load_avg_boxplot(self):
        return self._avg_boxplot
    
    def load_first_last_added(self):
        return self._first_last_added
    
    def load_playlists_by_artists(self):
        return self._top_playlists_by_artists

    def load_playlists_by_songs(self):
        return self._top_playlists_by_songs

    def load_playlists_by_explicit(self):
        return self._playlists_by_explicit

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
        df = self._all_songs_df['playlist'].value_counts().reset_index()
        fig = px.histogram(df, title="Playlists by # of Songs", marginal="rug",
                       hover_data='playlist')

        return fig

    def _graph_duration_histogram(self):
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
        if percent:
            df = self._all_songs_df.groupby('playlist')['explicit'].mean().reset_index()
            df['explicit'] = round(df['explicit'], 2)
        else:
            df = self._all_songs_df.groupby('playlist')['explicit'].sum().reset_index()
        fig = px.histogram(df, marginal="rug", hover_data='playlist')
        return fig

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
        labels = ['Average', 'Median']
        avg = self._graph_avg_popularity_histogram()
        median = self._graph_median_popularity_histogram()

        fig = _make_single_subplot(labels, [avg, median],
                                   'Popularity',
                                   '# Playlists',
                                   'Playlists by Popularity', 2)

        return Markup(fig.to_html(full_html=False))

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
        