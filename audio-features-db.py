#!/usr/bin/env python3
"""
Audio Features Database Processing Script
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import gzip
import os
import logging
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioFeaturesProcessor:
    def __init__(self, data_dir: str = '.data'):
        self.data_dir = data_dir
        self.kaggle_file = os.path.join(data_dir, 'kaggle_all_tracks.csv')
        self.kworb_file = os.path.join(data_dir, 'kworb_top_listeners.csv')
        self.unique_songs_file = os.path.join(data_dir, 'qf26s87ilixm0wn6njz7amx2f', 'unique_songs.ndjson.gz')
        
        # Output files
        self.my_tracks_file = os.path.join(data_dir, 'my_all_tracks.csv')
        self.popular_tracks_file = os.path.join(data_dir, 'popular_all_tracks.csv')
        self.db_tracks_file = os.path.join(data_dir, 'db_all_tracks.csv')
        
        os.makedirs(data_dir, exist_ok=True)
    
    def scrape_kworb_top_listeners(self) -> pd.DataFrame:
        """Scrape top 2500 Spotify artists by monthly listeners from Kworb."""
        logger.info("Scraping Kworb top listeners webpage...")
        
        url = "https://kworb.net/spotify/listeners.html"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                raise Exception("No table found on the webpage")
            
            artists_data = []
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    try:
                        rank = int(cells[0].get_text(strip=True))
                        artist_name = cells[1].get_text(strip=True)
                        monthly_listeners = cells[2].get_text(strip=True) if len(cells) > 2 else "N/A"
                        
                        artists_data.append({
                            'rank': rank,
                            'artist_name': artist_name,
                            'monthly_listeners': monthly_listeners
                        })
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(artists_data)
            logger.info(f"Scraped {len(df)} artists from Kworb")
            
            df.to_csv(self.kworb_file, index=False)
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Kworb: {e}")
            if os.path.exists(self.kworb_file):
                return pd.read_csv(self.kworb_file)
            else:
                raise
    
    def load_kaggle_tracks(self) -> pd.DataFrame:
        """Load Kaggle all tracks CSV."""
        logger.info(f"Loading Kaggle tracks from {self.kaggle_file}")
        
        if not os.path.exists(self.kaggle_file):
            raise FileNotFoundError(f"Kaggle file not found: {self.kaggle_file}")
        
        df = pd.read_csv(self.kaggle_file, engine='python')
        logger.info(f"Loaded {len(df)} tracks from Kaggle")
        
        # Show available columns for debugging
        logger.info(f"Kaggle columns: {list(df.columns)}")
        
        # Show sample of first few rows to understand data structure
        logger.info(f"First few rows sample:")
        logger.info(df.head(2).to_string())
        
        return df
    
    def load_unique_songs(self) -> pd.DataFrame:
        """Load user's unique songs from NDJSON.GZ."""
        logger.info(f"Loading unique songs from {self.unique_songs_file}")
        
        if not os.path.exists(self.unique_songs_file):
            raise FileNotFoundError(f"Unique songs file not found: {self.unique_songs_file}")
        
        try:
            with gzip.open(self.unique_songs_file, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
            
            songs_data = []
            for line in lines:
                try:
                    song = json.loads(line.strip())
                    songs_data.append(song)
                except json.JSONDecodeError:
                    continue
            
            df = pd.DataFrame(songs_data)
            logger.info(f"Loaded {len(df)} unique songs")
            return df
            
        except Exception as e:
            logger.error(f"Error loading unique songs: {e}")
            raise
    
    def process_my_tracks(self, kaggle_df: pd.DataFrame, unique_songs_df: pd.DataFrame) -> pd.DataFrame:
        """Create my_all_tracks.csv by left joining Kaggle data with user's unique songs."""
        logger.info("Processing my tracks...")
        
        # Extract Spotify IDs from unique songs
        unique_songs_df['spotify_id'] = unique_songs_df['id'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )
        
        valid_unique_songs = unique_songs_df.dropna(subset=['spotify_id'])
        logger.info(f"Found {len(valid_unique_songs)} songs with valid Spotify IDs")
        
        # Find ID column in Kaggle data
        kaggle_id_col = 'id'
        if kaggle_id_col not in kaggle_df.columns:
            if 'track_id' in kaggle_df.columns:
                kaggle_id_col = 'track_id'
            elif 'spotify_id' in kaggle_df.columns:
                kaggle_id_col = 'spotify_id'
            else:
                raise ValueError(f"Could not find Spotify ID column in Kaggle data")
        
        # Convert Kaggle ID column to string to ensure compatibility
        kaggle_df[kaggle_id_col] = kaggle_df[kaggle_id_col].astype(str)
        valid_unique_songs['spotify_id'] = valid_unique_songs['spotify_id'].astype(str)
        
        # Left join
        my_tracks = pd.merge(
            valid_unique_songs,
            kaggle_df,
            left_on='spotify_id',
            right_on=kaggle_id_col,
            how='left',
            suffixes=('_user', '_kaggle')
        )
        
        logger.info(f"Created my tracks dataset with {len(my_tracks)} rows")
        my_tracks.to_csv(self.my_tracks_file, index=False)
        return my_tracks
    
    def process_popular_tracks(self, kaggle_df: pd.DataFrame, kworb_df: pd.DataFrame) -> pd.DataFrame:
        """Create popular_all_tracks.csv by filtering tracks with top artists."""
        logger.info("Processing popular tracks...")
        
        top_artists = set(kworb_df['artist_name'].str.lower().str.strip())
        logger.info(f"Found {len(top_artists)} top artists from Kworb")
        
        # Find artist column in Kaggle data - use 'artists' (plural)
        kaggle_artist_col = 'artists'
        if kaggle_artist_col not in kaggle_df.columns:
            if 'artist' in kaggle_df.columns:
                kaggle_artist_col = 'artist'
            elif 'artist_name' in kaggle_df.columns:
                kaggle_artist_col = 'artist_name'
            else:
                raise ValueError(f"Could not find artist column in Kaggle data. Available columns: {list(kaggle_df.columns)}")
        
        # Filter tracks with top artists
        def has_top_artist(artist_str):
            if pd.isna(artist_str):
                return False
            
            try:
                # The artists column is a string representation of a list
                # Use ast.literal_eval to convert it to an actual list
                kaggle_artists_list = ast.literal_eval(artist_str)
                
                # Check if any of the Kaggle artists are in the top artists set
                has_match = any(artist.lower().strip() in top_artists for artist in kaggle_artists_list)
                
                return has_match
                
            except (ValueError, SyntaxError) as e:
                # Fallback: try to parse as comma-separated string
                logger.warning(f"Could not parse artists string '{artist_str}': {e}")
                kaggle_artists = [a.strip().lower() for a in str(artist_str).split(',')]
                has_match = any(kaggle_artist in top_artists for kaggle_artist in kaggle_artists)
                return has_match
        
        # Show some examples of what we're looking for
        logger.info(f"Looking for tracks with artists from top {len(top_artists)} artists")
        logger.info(f"Sample top artists: {list(top_artists)[:5]}")
        
        # Show sample of Kaggle artist data
        sample_artists = kaggle_df[kaggle_artist_col].dropna().head(5)
        logger.info(f"Sample Kaggle artists: {list(sample_artists)}")
        
        # Test the matching logic with a few examples
        test_cases = [
            "Justin Bieber",
            "Ed Sheeran, Justin Bieber", 
            "Taylor Swift",
            "Unknown Artist"
        ]
        logger.info("Testing artist matching logic:")
        for test_case in test_cases:
            result = has_top_artist(test_case)
            logger.info(f"  '{test_case}' -> {result}")
        
        # Count how many tracks have top artists before filtering
        logger.info("Analyzing artist matching results...")
        
        # Process tracks with progress updates every 1%
        total_tracks = len(kaggle_df)
        progress_interval = max(1, total_tracks // 100)  # Update every 1%
        
        logger.info(f"Processing {total_tracks} tracks with progress updates every {progress_interval} tracks...")
        
        # Create a list to store results
        match_results = []
        
        for idx, artist_str in enumerate(kaggle_df[kaggle_artist_col]):
            # Check if this track has a top artist
            has_match = has_top_artist(artist_str)
            match_results.append(has_match)
            
            # Show progress every 1%
            if (idx + 1) % progress_interval == 0:
                progress_pct = ((idx + 1) / total_tracks) * 100
                matches_so_far = sum(match_results)
                logger.info(f"Progress: {progress_pct:.1f}% ({idx + 1}/{total_tracks}) - Found {matches_so_far} matches so far")
        
        # Convert results to pandas Series for analysis
        match_series = pd.Series(match_results, index=kaggle_df.index)
        match_counts = match_series.value_counts()
        logger.info(f"Artist matching results: {match_counts}")
        
        popular_mask = match_series
        popular_tracks = kaggle_df[popular_mask].copy()
        
        # Add some debugging to show what's happening
        logger.info(f"Sample of tracks with top artists:")
        sample_popular = popular_tracks.head(5)
        for idx, row in sample_popular.iterrows():
            logger.info(f"  - {row.get('name', 'Unknown')} by {row.get(kaggle_artist_col, 'Unknown')}")
        
        logger.info(f"Found {len(popular_tracks)} popular tracks from top artists")
        popular_tracks.to_csv(self.popular_tracks_file, index=False)
        return popular_tracks
    
    def create_db_tracks(self, my_tracks_df: pd.DataFrame, popular_tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Create db_all_tracks.csv with consistent column structure from popular_tracks."""
        logger.info("Creating combined database...")
        
        # Get the target columns from popular_tracks_df (this will be our standard)
        target_columns = list(popular_tracks_df.columns)
        logger.info(f"Target columns from popular_tracks: {target_columns}")
        
        # Show all columns in each dataset for debugging
        logger.info(f"All columns in my_tracks: {list(my_tracks_df.columns)}")
        logger.info(f"All columns in popular_tracks: {list(popular_tracks_df.columns)}")
        
        # Create a clean version of my_tracks_df with only the target columns
        # Map columns properly considering the suffixes from the left join
        clean_my_tracks = pd.DataFrame(index=my_tracks_df.index)
        
        # Debug: Show what we're working with
        logger.info(f"my_tracks_df has {len(my_tracks_df)} rows")
        logger.info(f"First few rows of my_tracks_df:")
        logger.info(my_tracks_df.head(2).to_string())
        
        for col in target_columns:
            # Special case: for 'artists' column, prioritize 'artist' column over 'artists'
            if col == 'artists' and 'artist' in my_tracks_df.columns:
                # Convert single artist to list format to match Kaggle data structure
                clean_my_tracks[col] = my_tracks_df['artist'].apply(lambda x: [x] if pd.notna(x) else [])
                logger.info(f"Column '{col}' mapped from 'artist' column in my_tracks_df")
            # Special case: for 'name' column, prioritize 'name_user' over 'name_kaggle'
            elif col == 'name' and 'name_user' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df['name_user']
                logger.info(f"Column '{col}' mapped from 'name_user' in my_tracks_df")
            # Special case: for 'id' column, prioritize 'id_user' over 'id_kaggle' and extract first value
            elif col == 'id' and 'id_user' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df['id_user'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                logger.info(f"Column '{col}' mapped from 'id_user' in my_tracks_df (extracted first value)")
            # Special case: for 'album' column, prioritize 'album_user' over 'album_kaggle'
            elif col == 'album' and 'album_user' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df['album_user']
                logger.info(f"Column '{col}' mapped from 'album_user' in my_tracks_df")
            # Special case: for 'explicit' column, prioritize 'explicit_user' over 'explicit_kaggle'
            elif col == 'explicit' and 'explicit_user' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df['explicit_user']
                logger.info(f"Column '{col}' mapped from 'explicit_user' in my_tracks_df")
            # Special case: for 'duration_ms' column, convert 'duration' to milliseconds
            elif col == 'duration_ms' and 'duration' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df['duration'].apply(lambda x: int(x * 1000) if pd.notna(x) else pd.NA)
                logger.info(f"Column '{col}' mapped from 'duration' in my_tracks_df (converted to ms)")
            # Special case: for audio feature columns, prioritize _user over _kaggle
            elif col in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']:
                if f'{col}_user' in my_tracks_df.columns:
                    clean_my_tracks[col] = my_tracks_df[f'{col}_user']
                    logger.info(f"Column '{col}' mapped from '{col}_user' in my_tracks_df")
                elif f'{col}_kaggle' in my_tracks_df.columns:
                    clean_my_tracks[col] = my_tracks_df[f'{col}_kaggle']
                    logger.info(f"Column '{col}' mapped from '{col}_kaggle' in my_tracks_df")
                else:
                    clean_my_tracks[col] = pd.NA
                    logger.info(f"Column '{col}' not found in my_tracks_df, filling with NaN")
            # Check if column exists directly in my_tracks_df
            elif col in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df[col]
                logger.info(f"Column '{col}' found directly in my_tracks_df")
            # Check if it exists with _kaggle suffix (from the left join)
            elif f'{col}_kaggle' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df[f'{col}_kaggle']
                logger.info(f"Column '{col}' mapped from '{col}_kaggle' in my_tracks_df")
            # Check if it exists with _user suffix (from the left join)
            elif f'{col}_user' in my_tracks_df.columns:
                clean_my_tracks[col] = my_tracks_df[f'{col}_user']
                logger.info(f"Column '{col}' mapped from '{col}_user' in my_tracks_df")
            else:
                # Column doesn't exist, fill with NaN
                clean_my_tracks[col] = pd.NA
                logger.info(f"Column '{col}' not found in my_tracks_df, filling with NaN")
        
        # Debug: Check if we have any data in clean_my_tracks
        logger.info(f"clean_my_tracks has {len(clean_my_tracks)} rows")
        logger.info(f"clean_my_tracks columns: {list(clean_my_tracks.columns)}")
        logger.info(f"Sample of clean_my_tracks:")
        logger.info(clean_my_tracks.head(2).to_string())
        
        # Ensure popular_tracks_df has the same column order
        popular_tracks_clean = popular_tracks_df[target_columns].copy()
        
        # Debug: Check popular_tracks_clean
        logger.info(f"popular_tracks_clean has {len(popular_tracks_clean)} rows")
        
        # Now concatenate with consistent column structure
        combined_df = pd.concat([clean_my_tracks, popular_tracks_clean], ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} rows with {len(target_columns)} columns")
        
        # Debug: Check if my tracks made it into the combined dataset
        logger.info(f"Checking for my tracks in combined dataset...")
        # Look for tracks that might be from my dataset (those with NaN in some columns)
        my_track_indicators = combined_df.isna().any(axis=1)
        my_track_count = my_track_indicators.sum()
        logger.info(f"Found {my_track_count} rows that might be from my dataset (have NaN values)")
        
        # Debug: Check for specific artist "quietdrive" in both datasets
        logger.info(f"Checking for 'quietdrive' in my_tracks_df...")
        if 'artist' in my_tracks_df.columns:
            quietdrive_in_my = my_tracks_df[my_tracks_df['artist'].str.contains('quietdrive', case=False, na=False)]
            logger.info(f"Found {len(quietdrive_in_my)} tracks with 'quietdrive' in my_tracks_df (artist column)")
        elif 'artists' in my_tracks_df.columns:
            quietdrive_in_my = my_tracks_df[my_tracks_df['artists'].str.contains('quietdrive', case=False, na=False)]
            logger.info(f"Found {len(quietdrive_in_my)} tracks with 'quietdrive' in my_tracks_df")
        elif 'artists_kaggle' in my_tracks_df.columns:
            quietdrive_in_my = my_tracks_df[my_tracks_df['artists_kaggle'].str.contains('quietdrive', case=False, na=False)]
            logger.info(f"Found {len(quietdrive_in_my)} tracks with 'quietdrive' in my_tracks_df (artists_kaggle)")
        else:
            logger.info("No artist columns found in my_tracks_df")
        
        logger.info(f"Checking for 'quietdrive' in popular_tracks_df...")
        quietdrive_in_popular = popular_tracks_df[popular_tracks_df['artists'].str.contains('quietdrive', case=False, na=False)]
        logger.info(f"Found {len(quietdrive_in_popular)} tracks with 'quietdrive' in popular_tracks_df")
        
        logger.info(f"Checking for 'quietdrive' in combined_df...")
        quietdrive_in_combined = combined_df[combined_df['artists'].str.contains('quietdrive', case=False, na=False)]
        logger.info(f"Found {len(quietdrive_in_combined)} tracks with 'quietdrive' in combined_df")
        
        # Show sample of the combined data
        logger.info(f"Sample of combined data:")
        sample_combined = combined_df.head(5)
        for idx, row in sample_combined.iterrows():
            logger.info(f"  Row {idx}: {row.get('name', 'Unknown')} by {row.get('artists', 'Unknown')}")
        
        # Save the combined dataset
        combined_df.to_csv(self.db_tracks_file, index=False)
        logger.info(f"Saved combined database to {self.db_tracks_file}")
        
        return combined_df
    
    def run(self):
        """Main execution method."""
        try:
            logger.info("Starting Audio Features Database Processing...")
            
            kworb_df = self.scrape_kworb_top_listeners()
            kaggle_df = self.load_kaggle_tracks()
            unique_songs_df = self.load_unique_songs()
            
            my_tracks_df = self.process_my_tracks(kaggle_df, unique_songs_df)
            popular_tracks_df = self.process_popular_tracks(kaggle_df, kworb_df)
            db_tracks_df = self.create_db_tracks(my_tracks_df, popular_tracks_df)
            
            logger.info("=" * 50)
            logger.info("PROCESSING COMPLETE!")
            logger.info(f"Kworb top artists: {len(kworb_df)}")
            logger.info(f"Kaggle tracks: {len(kaggle_df)}")
            logger.info(f"User unique songs: {len(unique_songs_df)}")
            logger.info(f"My tracks: {len(my_tracks_df)}")
            logger.info(f"Popular tracks: {len(popular_tracks_df)}")
            logger.info(f"Combined database: {len(db_tracks_df)}")
            logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise

def main():
    """Main function."""
    try:
        processor = AudioFeaturesProcessor()
        processor.run()
        print("Audio features database processing completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
