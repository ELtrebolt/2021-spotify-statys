def setup_2(session):
    yield '<h1>Grouping Your Data</h1><br>'

    session['UNIQUE_SONGS_DF'] = session['setup'].get_unique_songs_df(session['ALL_SONGS_DF'])
    yield 'Getting Unique Songs...1/7<br>\n'

    session['TOP_ARTISTS'] = session['setup'].get_top_artists()
    yield 'Getting Top Artists...2/7<br>\n'

    session['TOP_SONGS'] = session['setup'].get_top_songs()
    yield 'Getting Top Songs...3/7<br>\n'

    session['setup'].add_top_artists_rank(session['UNIQUE_SONGS_DF'], session['TOP_ARTISTS'])
    yield 'Adding Top Artists Rank...4/7<br>\n'

    session['setup'].add_top_songs_rank(session['UNIQUE_SONGS_DF'], session['TOP_SONGS'])
    yield 'Adding Top Songs Rank...5/7<br>\n'

    session['setup'].add_genres(session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])
    yield 'Getting Artist Genres...6/7<br>\n'

    session['ALL_SONGS_DF'] = session['ALL_SONGS_DF'].to_dict('list')
    session['UNIQUE_SONGS_DF'] = session['UNIQUE_SONGS_DF'].to_dict('list')
    yield 'Finalizing Data...7/7<br>\n'

    session['setup_2'] = True
    yield '<script>window.location.href="' + REDIRECT_URI + '"</script>'

    def setup_1(session):
        #start_time = time.time()
        count = 1
        total = len(session['PLAYLIST_DICT'])
        session['ALL_SONGS_DF'] = pd.DataFrame()
        yield '<h1>Collecting Your Playlists</h1><br>'

        for name, _id in list(session['PLAYLIST_DICT'].items()):
            #end_time = time.time()
            #if end_time - start_time > 25:       #10=23 secs, 15=25 secs, 20=23 secs, 25=24 secs & good
                #break
            df = collection.get_playlist(session['SP'], name, _id)
            session['ALL_SONGS_DF'] = pd.concat([session['ALL_SONGS_DF'], df])
            yield name + '   ' + str(count) + '/' + str(total) + '<br/>\n' 
            count += 1
        session['ALL_SONGS_DF'].drop(columns='index',inplace=True)

        #user_id = session['USER_ID']
        #session['ALL_SONGS_DF'].to_csv(f"../data/{user_id}/all_songs_df.csv")
        session['setup_1'] = True
        session['setup_2'] = False
        yield '<script>window.location.href="' + REDIRECT_URI + '"</script>'

    #return Response(stream_with_context(setup_1(session)), mimetype='text/html')

    user_id = session['USER_ID']
        session['ALL_SONGS_DF'] = pd.read_csv(f'../data/{user_id}/all_songs_df.csv')

        yield 'Getting Unique Songs...1/7<br>\n'
        session['UNIQUE_SONGS_DF'] = session['setup'].get_unique_songs_df(session['ALL_SONGS_DF'])
        
        yield 'Getting Top Artists...2/7<br>\n'
        session['TOP_ARTISTS'] = session['setup'].get_top_artists()

        yield 'Getting Top Songs...3/7<br>\n'
        session['TOP_SONGS'] = session['setup'].get_top_songs()

        yield 'Adding Top Artists Rank...4/7<br>\n'
        session['setup'].add_top_artists_rank(session['UNIQUE_SONGS_DF'], session['TOP_ARTISTS'])

        yield 'Adding Top Songs Rank...5/7<br>\n'
        session['setup'].add_top_songs_rank(session['UNIQUE_SONGS_DF'], session['TOP_SONGS'])

        yield 'Getting Artist Genres...6/7<br>\n'
        session['setup'].add_genres(session['ALL_SONGS_DF'], session['UNIQUE_SONGS_DF'])

        yield 'Finalizing Data...7/7<br>\n'
        session['ALL_SONGS_DF'] = session['ALL_SONGS_DF'].to_dict('list')
        session['UNIQUE_SONGS_DF'] = session['UNIQUE_SONGS_DF'].to_dict('list')

        session['setup'].done()
        yield '<script>window.location.href="' + REDIRECT_URI + '"</script>'