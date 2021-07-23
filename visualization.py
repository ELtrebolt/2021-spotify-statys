# Imports --------------------------------------------------------------------------------------------------

# Visualization
import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# Venn Diagrams
from io import BytesIO
from matplotlib import pyplot as plt
import base64
from matplotlib_venn import venn2
from matplotlib_venn import venn3

# Convert Graph HTML to Insert-Ready
from flask import Markup

# Data
import pandas as pd
import datetime
from collections import defaultdict

# Constants ------------------------------------------------------------------------------------------------

PERCENTILE_COLS = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']
FEATURE_COLS = ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence']
OTHER_COLS = ['loudness', 'tempo', 'duration']
LABEL_CUTOFF_LENGTH = 25

# Helper Functions -----------------------------------------------------------------------------------------

def _h_bar(series, title, xaxis, yaxis=None, percents=False, long_names=False, hovertext=None):
    if percents:
        texty = [str(round(i, 2)) + '%' for i in series]
    else:
        texty = series
        
    if long_names:
        y_labels = [i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in series.keys()]
    else:
        y_labels = series.keys()
    
    fig = go.Figure(go.Bar(
                    x=series,
                    y=y_labels,
                    orientation='h',
                    text=texty, 
                    textposition='auto',
                    hovertext=hovertext
                    ))

    fig.update_layout(title_text=title, xaxis_title=xaxis, yaxis={'categoryorder':'total ascending'})
    fig.update_yaxes(automargin=True)

    if yaxis:
        fig.update_layout(yaxis_title=yaxis)

    return Markup(fig.to_html(full_html=False))

# Shared Functions -----------------------------------------------------------------------------------------

def common_graph_count_timeline(ALL_SONGS_DF, playlist_name=None, song_name=None, artist=None, continuous=False):
    if playlist_name:
        df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']==playlist_name][['name', 'artist', 'date_added']]
        if song_name and artist:
            song_df = df[df['name']==song_name]
            song_df = song_df[song_df['artist']==artist]
            song_dates = [song_df.iloc[0]['date_added']]
    else:
        df = ALL_SONGS_DF[['name', 'artist', 'date_added']]
        if song_name and artist:
            song_df = df[df['name']==song_name]
            song_df = song_df[song_df['artist']==artist]
            song_dates = song_df['date_added'].unique()

    df = df.groupby(['date_added'], as_index=False)[['name', 'artist']].agg(lambda x: list(x))
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

    title = playlist_name if playlist_name else 'Any Playlist'

    fig = px.line(df, x='date_added', y='total_count', title='Timeline of When Songs Were Added to ' + title, hover_name = 'songs')
    if song_name and artist:
        for song_date in song_dates:
            fig.add_vline(x=song_date, line_width=3, line_dash="dash", line_color="green")

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
    fig.update_layout(yaxis_title='Total Songs' if continuous else 'Daily Added Songs')

    return Markup(fig.to_html(full_html=False))

# def graph_feature_rank_table(col, top_n, lowest=False):
#     table_cols = ['name', 'artist', col, col + '_rank', 'playlist']
#     df = session['UNIQUE_SONGS_DF'].sort_values(col, ascending=lowest).head(top_n)
    
#     fig = go.Figure(data=[go.Table(
#         columnwidth = [250, 250, 150, 150, 500],
#         header=dict(values=list(table_cols),
#                     fill_color='paleturquoise',
#                     align='center'),
#         cells=dict(values=[df[i] for i in table_cols],
#                 fill_color='lavender',
#                 align='center'))
#     ])
    
#     # fig.update_layout(autosize=True)

#     # fig.update_traces(domain=dict(row=2), selector=dict(type='table'))

#     return Markup(fig.to_html(full_html=False))

# Currently Playing Stats Page ----------------------------------------------------------------------------------------------

class CurrentlyPlayingPage():
    def __init__(self, song, artist, playlist, all_songs_df, unique_songs_df):
        self._song = song
        self._artist = artist
        self._playlist = playlist
        
        self._unique_songs_df = unique_songs_df
        self._all_songs_df = all_songs_df

        song_df = unique_songs_df[unique_songs_df['artist']==artist]
        song_df = song_df[song_df['name']==song]
        
        if len(song_df.index) > 0:
            self._artist_genres = song_df.iloc[0]['genres']
            self._song_genres = list({j for i in self._artist_genres for j in i})
        else:
            self._artist_genres = []
            self._song_genres = None

    def graph_top_rank_table(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name']==self._song]
        df = df[df['artist']==self._artist]

        values = [['Artists', 'Songs'], [df['artists_short_rank'], df['songs_short_rank']], 
                [df['artists_med_rank'], df['songs_med_rank']], [df['artists_long_rank'], df['songs_long_rank']]]

        fig = go.Figure(data=[go.Table(
            # columnorder = [1, 2, 3, 4],
            # columnwidth = [80,400],
            header = dict(
                values = [['<b>Top 50 Rank</b>'],
                            ['<b>Short Term</b>'],
                            ['<b>Medium Term</b>'],
                            ['<b>Long Term</b>']],
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
        fig.update_layout(title_text='Song / Artist Rank in User\'s Top 50', height=300)

        return Markup(fig.to_html(full_html=False))

    def graph_song_features_vs_avg(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df

        song_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name']==self._song]
        song_df = song_df[song_df['artist']==self._artist][FEATURE_COLS]
        
        if self._playlist:
            playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']==self._playlist][FEATURE_COLS]
        artist_dfs = [UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist']==i][FEATURE_COLS] for i in self._artist.split(', ')]
        avg_df = UNIQUE_SONGS_DF[FEATURE_COLS]

        if self._playlist:
            dfs = [playlist_df, avg_df, song_df]
        else:
            dfs = [avg_df, song_df]

        for df in dfs:
            df['popularity'] = df['popularity']/100
        dfs = [df.mean(axis=0) for df in dfs]

        for df in artist_dfs:
            df['popularity'] = df['popularity']/100
        artist_dfs = [df.mean(axis=0) for df in artist_dfs]
    
        # add song values to last so features and percentiles avgs match colors
        song_vals = dfs[-1]
        dfs = dfs[:-1]
        dfs.append(artist_dfs)
        dfs.append(song_vals)

        fig = go.Figure()
        if self._playlist:
            names = [self._playlist, 'All Playlists', self._artist.split(', '), self._song]
        else:
            names = ['All Playlists', self._artist.split(', '), self._song]

        for series, name in zip(dfs, names):
            if type(name) == list:
                for s, n in zip(series, name):
                    fig.add_trace(go.Scatterpolar(
                        r=s,
                        theta=FEATURE_COLS,
                        fill='toself',
                        name=n
                    ))
            else:
                fig.add_trace(go.Scatterpolar(
                    r=series,
                    theta=FEATURE_COLS,
                    fill='toself',
                    name=name
                ))

        if self._playlist:
            title = 'Song Audio Features vs. AVG Song from Playlist, All Playlists, & Artists'
        else:
            title = 'Song Audio Features vs. AVG Song from All Playlists & Artists'
        
        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True,
        title_text=title
        # margin=dict(l=20, r=20, t=20, b=20)
        )

        return Markup(fig.to_html(full_html=False))

    def graph_song_percentiles_vs_avg(self):
        UNIQUE_SONGS_DF = self._unique_songs_df
        ALL_SONGS_DF = self._all_songs_df
        
        # Since playlist and artist df length will vary, make new percentile cols for them and then isolate song
        if self._playlist:
            playlist_df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']==self._playlist]
            for col in PERCENTILE_COLS:
                sz = playlist_df[col].size-1
                playlist_df[col + '_percentile'] = playlist_df[col].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
            playlist_df = playlist_df[playlist_df['name']==self._song]
            playlist_df = playlist_df[playlist_df['artist']==self._artist]

        artist_dfs = []
        for a in self._artist.split(', '):
            mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: a in x)
            artist_df = UNIQUE_SONGS_DF[mask]
            for col in PERCENTILE_COLS:
                sz = artist_df[col].size-1
                if sz > 0:
                    artist_df[col + '_percentile'] = artist_df[col].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
                else:
                    artist_df[col + '_percentile'] = [0]
            artist_df = artist_df[artist_df['name']==self._song]
            if len(artist_df.index) == 0:
                return None
            artist_dfs.append(artist_df)

        # UNIQUE_SONGS_DF already has _percentile columns, so get the matching song by song and artist name
        avg_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['name']==self._song]
        avg_df = UNIQUE_SONGS_DF[UNIQUE_SONGS_DF['artist']==self._artist]

        cols = [i + '_percentile' for i in PERCENTILE_COLS]
        if self._playlist:
            dfs = [playlist_df[cols], avg_df[cols], [df[cols] for df in artist_dfs]]
        else:
            dfs = [avg_df[cols], [df[cols] for df in artist_dfs]]
        
        data = []
        if self._playlist:
            names = [self._playlist, 'All Playlists', self._artist.split(', ')]
        else:
            names = ['All Playlists', self._artist.split(', ')]

        for df, name in zip(dfs, names):
            if type(name) == list:
                for d, n in zip(df, name):
                    data.append(go.Bar(name=n, x=PERCENTILE_COLS, y=d.iloc[0], text=d.iloc[0].astype(int), textposition='auto'))
            else:
                data.append(go.Bar(name=name, x=PERCENTILE_COLS, y=df.iloc[0], text=df.iloc[0].astype(int), textposition='auto'))

        fig = go.Figure(data=data)

        if self._playlist:
            title = 'Percentile of Song Audio Features by Playlist, All Playlists, & Artists'
        else:
            title = 'Percentile of Audio Features by All Playlists & Artists'

        fig.update_layout(barmode='group', title_text=title)

        return Markup(fig.to_html(full_html=False))

    def graph_date_added_to_playlist(self):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF[ALL_SONGS_DF['name']==self._song]
        df = df[df['artist']==self._artist]

        today = datetime.date.today()
        new = pd.DataFrame({'Task':df['playlist'], 'Start':df['date_added'], 'Finish':[today]*len(df['playlist'])})

        fig = ff.create_gantt(new)
        fig.update_layout(title_text='Timeline Of When ' + self._song + ' Was Added To Playlists')

        return Markup(fig.to_html(full_html=False))

    def graph_count_timeline(self):
        return common_graph_count_timeline(self._all_songs_df, self._playlist, song_name=self._song, artist=self._artist)

    def graph_top_playlists_by_artist(self, artist_name):
        ALL_SONGS_DF = self._all_songs_df

        mask = ALL_SONGS_DF['artist'].apply(lambda x: artist_name in x)
        df = ALL_SONGS_DF[mask]

        series = df['playlist'].value_counts(ascending=True)

        return _h_bar(series, title='Most Popular Playlists For Artist: ' + artist_name,
                      xaxis='Number of Artist Songs in the Playlist')

        fig = go.Figure(go.Bar(
            x=series,
            y=series.keys(),
            orientation='h',
            text=series, 
            textposition='auto'))

        fig.update_layout(title_text='Most Popular Playlists For Artist: ' + artist_name,
            xaxis_title="Number of Artist Songs in the Playlist")
        fig.update_yaxes(automargin=True)

        return Markup(fig.to_html(full_html=False))

    def graph_top_songs_by_artist(self, artist_name):
        UNIQUE_SONGS_DF = self._unique_songs_df
        mask = UNIQUE_SONGS_DF['artist'].apply(lambda x: artist_name in x)
        df = UNIQUE_SONGS_DF[mask]

        # df.sort_values(by='num_playlists', inplace=True)
        d = defaultdict(list)
        for i, j in zip(df['name'], df['num_playlists']):
            d[j].append(i)
        d = {k:', '.join(v) for k,v in sorted(d.items(), key = lambda x: x[0], reverse=True)}

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
                
        fig.update_layout(autosize=True, title_text='Most Popular Songs For Artist: ' + artist_name, 
            xaxis_title="Number of Playlists The Artist's Song Is In")
        

        return Markup(fig.to_html(full_html=False))

    def graph_artist_genres(self):
        if len(self._artist_genres) == 1:
            return [', '.join(self._song_genres), False]
        elif len(self._artist_genres) == 2:
            left = set(self._artist_genres[0])
            right = set(self._artist_genres[1])

            img = BytesIO()

            plt.rcParams.update({'font.size': 18})
            plt.figure(figsize=(18,10))

            fig = venn2([left, right], tuple(self._artist.split(', ')))
            bubbles = ['10', '01', '11']
            text = [left-right, right-left, left&right]
            for i, j in zip(bubbles, text):
                try:
                    fig.get_label_by_id(i).set_text('\n'.join(j))
                except:
                    pass
            for text in fig.set_labels:
                text.set_fontsize(18)
            
            # Save it to a temporary buffer
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)

            # Embed the result in the html output
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            return [plot_url, True]

        elif len(self._artist_genres) >= 3:
            first = set(self._artist_genres[0])
            second = set(self._artist_genres[1])
            third = set(self._artist_genres[2])

            img = BytesIO()

            plt.rcParams.update({'font.size': 18})
            plt.figure(figsize=(18,10))

            fig = venn3([first, second, third], set_labels=tuple(self._artist.split(', ')) )

            bubbles = ['100', '010', '001', '110', '011', '101', '111']
            text = [first-second-third, second-first-third, third-first-second, 
                    first&second-third, second&third-first, first&third-second, 
                    first&second&third]
            
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
            return [plot_url, True]

    # What percentage of songs in a playlist or among all playlists have these genres?
    def graph_song_genres_vs_avg(self, playlist=False):
        if self._song_genres != None:
            UNIQUE_SONGS_DF = self._unique_songs_df
            ALL_SONGS_DF = self._all_songs_df
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
            length = len(avg_df.index)
            for g in self._song_genres:
                mask = avg_df['genres'].apply(lambda x: g in {z for y in x for z in y})
                percents.append(len(avg_df[mask].index)/length*100)

            series = pd.Series(dict(zip(self._song_genres, percents)))
            return _h_bar(series, title=title, xaxis=xaxis, percents=True)

            fig = go.Figure(go.Bar(
                    x=percents,
                    y=self._song_genres,
                    orientation='h',
                    text=[str(round(i, 2)) + '%' for i in percents], 
                    textposition='auto'))

            fig.update_layout(title_text=title, xaxis_title=xaxis, yaxis={'categoryorder':'total ascending'})

            return Markup(fig.to_html(full_html=False))
        return None

    def graph_all_artists(self):
        artist_top_graphs = []
        for i in self._artist.split(', '):
            artist_top_playlists_bar = self.graph_top_playlists_by_artist(i)
            artist_top_songs_table = self.graph_top_songs_by_artist(i)
            artist_top_graphs.append((artist_top_playlists_bar, artist_top_songs_table))
        return artist_top_graphs

# Timeline Home Page ----------------------------------------------------------------------------------------------

class HomePage():
    def __init__(self, all_songs_df, unique_songs_df):
        self._today = datetime.date.today()
        
        self._all_songs_df = all_songs_df
        self._unique_songs_df = unique_songs_df

    def graph_on_this_date(self):
        ALL_SONGS_DF = self._all_songs_df
        today = str(self._today)[5:]
        df = ALL_SONGS_DF[ALL_SONGS_DF['date_added'].apply(lambda x: today in x)]
        if len(df.index) == 0:
            return 'No recorded data of adding songs to a playlist'

        df['year'] = [i[:4] for i in df['date_added']]
        df = df[['name', 'playlist', 'year']]
        df = df.groupby(['playlist', 'year'], as_index=False)[['name']].agg(lambda x: list(x))
        df = df.sort_values(by='year')
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Year', 'Playlist', 'Songs Added'],
                line_color='darkslategray',
                fill_color='royalblue',
                font=dict(color='white', size=18),
                height=40
                ),
            cells=dict(
                values=[df['year'].to_list(), df['playlist'].to_list(), [', '.join(i) for i in df['name']] ],
                line_color='darkslategray',
                align='center',
                font_size=[18, 14],
                height=30)
                )
                ])

        return Markup(fig.to_html(full_html=False))

    def graph_count_timeline(self):
        return common_graph_count_timeline(self._all_songs_df, continuous=False)

    def graph_last_added(self):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF.sort_values(by='date_added',ascending=False)
        last_date = df['date_added'].to_list()[0]
        distance = (self._today - datetime.date(*map(int, last_date.split('-')))).days

        df = ALL_SONGS_DF[ALL_SONGS_DF['date_added']==last_date]
        num_songs = len(df.index)
        df = df[['name', 'playlist']]
        df = df.groupby(['playlist'], as_index=False)[['name']].agg(lambda x: list(x))

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Playlist', 'Songs Added'],
                line_color='darkslategray',
                fill_color='royalblue',
                font=dict(color='white', size=18),
                height=40
                ),
            cells=dict(
                values=[list(df['playlist']), [', '.join(i) for i in df['name']] ],
                line_color='darkslategray',
                align='center',
                font_size=[18, 14],
                height=30)
                )
                ])

        return [distance, Markup(fig.to_html(full_html=False)), num_songs, len(df['playlist'])]

    def get_library_totals(self):
        ALL_SONGS_DF = self._all_songs_df
        UNIQUE_SONGS_DF = self._unique_songs_df

        overall_data = [len(ALL_SONGS_DF.index), len(UNIQUE_SONGS_DF), 
                        len(ALL_SONGS_DF['playlist'].unique()), len(UNIQUE_SONGS_DF['artist'].unique()),
                        len(UNIQUE_SONGS_DF['album'].unique())]
        return overall_data

# Overall Stats Page ----------------------------------------------------------------------------------------------

class OverallPage():
    def __init__(self, session):
        self._all_songs_df = session['ALL_SONGS_DF']
        self._unique_songs_df = session['UNIQUE_SONGS_DF']

        self._session = session

    def graph_top_genres_by_followed_artists(self):
        artists = self._session['SPOTIFY'].current_user_followed_artists()['artists']['items']
        
        d = defaultdict(int)
        d2 = defaultdict(list)
        
        for a in artists:
            for g in a['genres']:
                d[g] += 1
                d2[g].append(a['name'])
        
        top_n = 10
        data = {i[0]:i[1] for i in sorted(d.items(), key = lambda x: x[1], reverse=True)[:top_n][::-1]}
        
        series = pd.Series(data)
        return _h_bar(series, title='Top Genres by Followed Artists', xaxis='# of Followed Artists',
                        hovertext=[', '.join(d2[i]) for i in data.keys()], yaxis='Genre', long_names=True)
        
        fig = go.Figure(go.Bar(
                x=list(data.values()),
                y=[i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in data.keys()],
                orientation='h',
                hovertext=[', '.join(d2[i]) for i in data.keys()], 
                text=list(data.values()),
                textposition='auto'))

        title = 'Top Genres by Followed Artists'
        xaxis = '# of Followed Artists'
        yaxis = 'Genre'
        fig.update_layout(title_text=title, xaxis_title=xaxis, yaxis_title=yaxis, yaxis={'categoryorder':'total ascending'})

        return Markup(fig.to_html(full_html=False))

    def graph_top_playlists_by_top_50(self, artists=False):
        ALL_SONGS_DF = self._all_songs_df

        final = []
        for time_range in [0, 1, 2]:
            if artists:
                dicty = self._session['TOP_ARTISTS'][time_range]
            else:
                dicty = self._session['TOP_SONGS'][time_range]

            playlist_dict = defaultdict(int)
            if artists:
                for name, playlist in zip(ALL_SONGS_DF['artist'], ALL_SONGS_DF['playlist']):
                    for n in name.split(', '):
                        if n in dicty[:artists]:
                            playlist_dict[playlist] += 1
                            break
            else:
                for name, playlist in zip(ALL_SONGS_DF['name'], ALL_SONGS_DF['playlist']):
                    if name in dicty.keys():
                        playlist_dict[playlist] += 1

            final.append(playlist_dict)
        df = pd.DataFrame(final, index = ['# Short Term Songs', '# Medium Term Songs', '# Long Term Songs'])
        df = df.T.fillna(0)
        df['Playlist'] = df.index

        fig = px.scatter(df, x="# Short Term Songs", y="# Medium Term Songs",
                size="# Long Term Songs", hover_name='Playlist')
        
        if artists:
            title = 'Top Playlists by Number of Top ' + str(artists) + ' Artists\' Songs'
        else:
            title = 'Top Playlists by Number of Top 50 Songs In Them'
        fig.update_layout(title_text=title)
    
        return Markup(fig.to_html(full_html=False))

    def graph_top_songs_by_num_playlists(self, top_n):
    
        df = self._unique_songs_df.sort_values(by='num_playlists', ascending=False)
        df = df.head(top_n)

        series = pd.Series(dict(zip(df['name'], df['num_playlists'])))
        title = 'Top ' + str(top_n) + ' Songs Across ' + str(len(self._session['PLAYLIST_DICT'])) + ' Playlists'
        
        return _h_bar(series, title=title, xaxis='Number of Playlists Song is In', yaxis='Song Name',
                        long_names=True, hovertext = [', '.join(i) for i in df['playlist']])

        fig = go.Figure(go.Bar(
                x=df['num_playlists'].to_list(),
                y=[i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in df['name']],
                orientation='h',
                hovertext=[', '.join(i) for i in df['playlist']], 
                text=df['num_playlists'].to_list(),
                textposition='auto'))

        fig.update_layout(title_text='Top ' + str(top_n) + ' Songs Across ' + str(len(self._session['PLAYLIST_DICT'])) + ' Playlists', 
            xaxis_title='Number of Playlists Song is In', yaxis={'categoryorder':'total ascending'},
            yaxis_title='Song Name')

        return Markup(fig.to_html(full_html=False))

    def graph_top_artists_and_albums_by_num_playlists(self, top_n, albums=False):
        ALL_SONGS_DF = self._all_songs_df
        dicty = dict()
        if albums:
            title = 'Top ' + str(top_n) + ' Albums Across ' + str(len(ALL_SONGS_DF['playlist'].unique())) + ' Playlists'
            yaxis = 'Album Name'
            for a in ALL_SONGS_DF['album'].unique():
                a_df = ALL_SONGS_DF[ALL_SONGS_DF['album'] == a]
                dicty[a] = len(a_df.index)
        else:
            title = 'Top ' + str(top_n) + ' Artists Across ' + str(len(ALL_SONGS_DF['playlist'].unique())) + ' Playlists'
            yaxis = 'Artist Name'
            for a in ALL_SONGS_DF['artist'].unique():
                mask = ALL_SONGS_DF['artist'].apply(lambda x: a in x)
                a_df = ALL_SONGS_DF[mask]
                dicty[a] = len(a_df.index)
        dicty = {k:v for k,v in sorted(dicty.items(), key=lambda x: x[1], reverse=True)[:top_n]}

        series = pd.Series(dicty)

        return _h_bar(series, title=title, xaxis='Number of Playlists Song is In', yaxis='Song Name',
                        long_names=True, hovertext=list(series.keys()))

        fig = go.Figure(go.Bar(
                x=list(dicty.values()),
                y=[i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in dicty.keys()],
                orientation='h',
                hovertext=list(dicty.keys()),
                text=list(dicty.values()), 
                textposition='auto'))

        fig.update_layout(title_text=title, 
            xaxis_title='Number of Songs in All Playlists', yaxis={'categoryorder':'total ascending'},
            yaxis_title=yaxis)

        return Markup(fig.to_html(full_html=False))

# Analyze Multiple Playlists Page ----------------------------------------------------------------------------------------------

class PlaylistIntersectionPage():
    def __init__(self, playlists, unique_songs_df):
        self._playlists = playlists
        self._unique_songs_df = unique_songs_df

    def get_intersection_of_playlists(self):
        mask = self._unique_songs_df['playlist'].apply(lambda x: set([i.lower() for i in self._playlists]).issubset(set([j.lower() for j in x])))
        df = self._unique_songs_df[mask]

        if len(df.index) >= 1:
            return [df['name'], len(df.index)]
        else:
            return [None, 0]

# Analyze Single Playlist Page ----------------------------------------------------------------------------------------------

class AnalyzePlaylistPage():
    def __init__(self, playlist, all_songs_df, unique_songs_df):
        self._playlist = playlist
        self._all_songs_df = all_songs_df
        self._unique_songs_df = unique_songs_df

    def graph_count_timeline(self):
        return common_graph_count_timeline(self._all_songs_df, self._playlist)

    def graph_playlist_genres(self):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']==self._playlist]
        genres = defaultdict(int)
        for i in df['genres']:
            unique_genres = {y for x in i for y in x}
            for j in unique_genres:
                genres[j] += 1
        total = len(df.index)
        relative = {k:v/total*100 for k, v in sorted(genres.items(), key = lambda x: x[1], reverse=True)[:10]}
        genres = {k:str(v) for k,v in sorted(genres.items(), key = lambda x: x[1], reverse=True)}
        
        series = pd.Series(relative)
        title = 'Most Popular Genres For Playlist: ' + self._playlist
        return _h_bar(series, title=title, xaxis = '% of Songs', yaxis='Artist Genre', percents=True,
                        hovertext=[genres[i] + ' Songs' for i in relative])

        fig = go.Figure(go.Bar(
                x=list(genres.values()),
                y=list(genres.keys()),
                orientation='h',
                text=[relative[i] + '%' for i in genres], 
                textposition='auto'))

        fig.update_layout(title_text='Most Popular Genres For Playlist: ' + self._playlist,
            xaxis_title="Number of Songs", yaxis_title='Artist Genre', yaxis={'categoryorder':'total ascending'})

        return Markup(fig.to_html(full_html=False))

    def graph_top_artists(self, top_n=10):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']==self._playlist]

        dicty = defaultdict(int)
        for i in df['artist']:
            for j in i.split(', '):
                dicty[j] += 1

        total = len(df.index)
        relative = {k:str(round(v/total*100, 2)) for k, v in dicty.items()}
        dicty = {k:v for k,v in sorted(dicty.items(), key = lambda x: x[1], reverse=True)[:10]}

        title = 'Most Popular Artists For Playlist: ' + self._playlist

        series = pd.Series(dicty)
        return _h_bar(series, title=title, yaxis='Artist', xaxis='Number of Songs', long_names=True, 
                        hovertext=[relative[i] + '% of Playlist' for i in dicty])

        fig = go.Figure(go.Bar(
                x=list(dicty.values()),
                y=[i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in dicty.keys()],
                orientation='h',
                text=[relative[i] + '%' for i in dicty], 
                textposition='auto'))

        title = 'Most Popular Artists For Playlist: ' + self._playlist
        yaxis = 'Artist'

        fig.update_layout(title_text=title,
            xaxis_title="Number of Songs", yaxis_title=yaxis, yaxis={'categoryorder':'total ascending'})

        return Markup(fig.to_html(full_html=False))

    def graph_top_albums(self, top_n=10):
        ALL_SONGS_DF = self._all_songs_df
        df = ALL_SONGS_DF[ALL_SONGS_DF['playlist']==self._playlist]
        
        dicty = defaultdict(int)
        for i in df['album']:
            dicty[i] += 1

        total = len(df.index)
        relative = {k:str(round(v/total*100, 2)) for k, v in dicty.items()}
        dicty = {k:v for k,v in sorted(dicty.items(), key = lambda x: x[1], reverse=True)[:10]}

        title = 'Most Popular Albums For Playlist: ' + self._playlist

        series = pd.Series(dicty)
        return _h_bar(series, title=title, yaxis='Album', xaxis='Number of Songs', long_names=True, 
                        hovertext=[relative[i] + '% of Playlist' for i in dicty])

        fig = go.Figure(go.Bar(
                x=list(dicty.values()),
                y=[i if len(i) < LABEL_CUTOFF_LENGTH else i[:LABEL_CUTOFF_LENGTH] + '...' for i in dicty.keys()],
                orientation='h',
                text=[relative[i] + '%' for i in dicty], 
                textposition='auto'))

        title = 'Most Popular Albums For Playlist: ' + self._playlist
        yaxis = 'Album'

        fig.update_layout(title_text=title,
            xaxis_title="Number of Songs", yaxis_title=yaxis, yaxis={'categoryorder':'total ascending'})

        return Markup(fig.to_html(full_html=False))

    def graph_similar_playlists(self, top_n=10):
        UNIQUE_SONGS_DF = self._unique_songs_df
        mask = UNIQUE_SONGS_DF['playlist'].apply(lambda x: self._playlist in x)
        df = UNIQUE_SONGS_DF[mask]

        other_playlists = {j for i in df['playlist'] for j in i if j != self._playlist}
        dicty = dict()
        for p in other_playlists:
            mask = df['playlist'].apply(lambda x: p in x)
            df2 = df[mask]
            dicty[p] = len(df2.index)

        total = len(df.index)
        relative = {k:v/total*100 for k, v in sorted(dicty.items(), key = lambda x: x[1], reverse=True)[:10]}
        counts = {k:str(v) for k,v in sorted(dicty.items(), key = lambda x: x[1], reverse=True)}
        
        series = pd.Series(relative)
        title = 'Most Similar Playlists By Songs Shared: ' + self._playlist
        return _h_bar(series, title=title, xaxis = '% of ' + self._playlist + ' Songs', yaxis='Playlist', percents=True,
                        hovertext=[counts[i] + ' Songs' for i in relative])