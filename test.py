import pandas as pd
import os

def analyze_missing_audio_features():
    """Analyze my_all_tracks.csv to find songs missing ReccoBeats features but available in Kaggle dataset."""
    
    # File paths
    my_tracks_file = '.data/my-data/my_all_tracks.csv'
    kaggle_file = '.data/kaggle_all_tracks.csv'
    
    print("🔍 Analyzing missing audio features...")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(my_tracks_file):
        print(f"❌ Error: {my_tracks_file} not found!")
        return
    
    if not os.path.exists(kaggle_file):
        print(f"❌ Error: {kaggle_file} not found!")
        return
    
    try:
        # Load the datasets
        print(f"📊 Loading {my_tracks_file}...")
        my_tracks_df = pd.read_csv(my_tracks_file)
        print(f"✅ Loaded {len(my_tracks_df)} tracks from my_all_tracks.csv")
        
        print(f"📊 Loading {kaggle_file}...")
        kaggle_df = pd.read_csv(kaggle_file)
        print(f"✅ Loaded {len(kaggle_df)} tracks from Kaggle dataset")
        
        # Show column information
        print(f"\n📋 My tracks columns: {list(my_tracks_df.columns)}")
        print(f"📋 Kaggle columns: {list(kaggle_df.columns)}")
        
        # Define audio feature columns
        audio_features = ['danceability', 'energy', 'instrumentalness', 'key', 'liveness', 
                         'loudness', 'mode', 'speechiness', 'tempo', 'valence']
        
        # Check which audio features exist in each dataset
        my_audio_cols = [col for col in audio_features if col in my_tracks_df.columns]
        kaggle_audio_cols = [col for col in audio_features if col in kaggle_df.columns]
        
        print(f"\n🎵 Audio features in my tracks: {my_audio_cols}")
        print(f"🎵 Audio features in Kaggle: {kaggle_audio_cols}")
        
        # Find tracks missing ReccoBeats features
        print(f"\n🔍 Analyzing missing features...")
        
        # Create a copy for analysis
        analysis_df = my_tracks_df.copy()
        
        # Add missing features flag
        missing_features = []
        available_in_kaggle = []
        
        for idx, row in analysis_df.iterrows():
            track_id = row.get('id', '')
            track_name = row.get('name', 'Unknown')
            artist = row.get('artist', 'Unknown')
            
            # Check if track has any audio features from ReccoBeats
            has_features = False
            for col in my_audio_cols:
                if pd.notna(row.get(col)) and row.get(col) is not None:
                    has_features = True
                    break
            
            if not has_features:
                missing_features.append({
                    'id': track_id,
                    'name': track_name,
                    'artist': artist,
                    'missing_features': True
                })
                
                # Check if this track exists in Kaggle dataset
                kaggle_match = kaggle_df[
                    (kaggle_df['id'] == track_id) | 
                    ((kaggle_df['name'] == track_name) & (kaggle_df['artist'] == artist))
                ]
                
                if not kaggle_match.empty:
                    available_in_kaggle.append({
                        'id': track_id,
                        'name': track_name,
                        'artist': artist,
                        'kaggle_has_features': True,
                        'kaggle_features': {}
                    })
                    
                    # Get the features from Kaggle
                    kaggle_row = kaggle_match.iloc[0]
                    for col in kaggle_audio_cols:
                        if col in kaggle_row and pd.notna(kaggle_row[col]):
                            available_in_kaggle[-1]['kaggle_features'][col] = kaggle_row[col]
        
        # Summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"   Total tracks in my dataset: {len(my_tracks_df)}")
        print(f"   Tracks missing ReccoBeats features: {len(missing_features)}")
        print(f"   Tracks available in Kaggle: {len(available_in_kaggle)}")
        
        if missing_features:
            print(f"\n❌ TRACKS MISSING RECCOBEATS FEATURES:")
            print("-" * 80)
            for i, track in enumerate(missing_features[:20]):  # Show first 20
                print(f"{i+1:3d}. {track['name']} - {track['artist']} (ID: {track['id']})")
            
            if len(missing_features) > 20:
                print(f"   ... and {len(missing_features) - 20} more tracks")
        
        if available_in_kaggle:
            print(f"\n✅ TRACKS AVAILABLE IN KAGGLE DATASET:")
            print("-" * 80)
            for i, track in enumerate(available_in_kaggle[:20]):  # Show first 20
                feature_count = len(track['kaggle_features'])
                print(f"{i+1:3d}. {track['name']} - {track['artist']} (ID: {track['id']}) - {feature_count} features")
            
            if len(available_in_kaggle) > 20:
                print(f"   ... and {len(available_in_kaggle) - 20} more tracks")
            
            # Save detailed analysis to CSV
            analysis_file = '.data/missing_features_analysis.csv'
            analysis_df_export = pd.DataFrame(available_in_kaggle)
            analysis_df_export.to_csv(analysis_file, index=False)
            print(f"\n💾 Detailed analysis saved to: {analysis_file}")
        
        # Show sample of tracks with features vs without
        print(f"\n🎯 SAMPLE COMPARISON:")
        print("-" * 80)
        
        # Sample tracks with features
        tracks_with_features = analysis_df[analysis_df[my_audio_cols[0]].notna()].head(3)
        if not tracks_with_features.empty:
            print("Tracks WITH ReccoBeats features:")
            for _, row in tracks_with_features.iterrows():
                feature_count = sum(1 for col in my_audio_cols if pd.notna(row.get(col)))
                print(f"  ✅ {row.get('name', 'Unknown')} - {row.get('artist', 'Unknown')} ({feature_count} features)")
        
        # Sample tracks without features
        tracks_without_features = analysis_df[analysis_df[my_audio_cols[0]].isna()].head(3)
        if not tracks_without_features.empty:
            print("\nTracks WITHOUT ReccoBeats features:")
            for _, row in tracks_without_features.iterrows():
                print(f"  ❌ {row.get('name', 'Unknown')} - {row.get('artist', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_missing_audio_features()