#!/usr/bin/env python3
"""
Data Quality Assessment Script
Analyzes the Reddit dataset for completeness, quality, and research suitability.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

def analyze_data_quality(csv_path):
    """Comprehensive data quality analysis"""
    print("=== DATA QUALITY ASSESSMENT ===\n")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded dataset: {csv_path}")
        print(f"✓ Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # 1. Data Structure Analysis
    print("1. DATA STRUCTURE ANALYSIS")
    print("-" * 40)
    print("Columns:", list(df.columns))
    print(f"Data types:\n{df.dtypes}\n")
    
    # 2. Missing Values Analysis
    print("2. MISSING VALUES ANALYSIS")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing %': missing_pct
    })
    print(missing_summary[missing_summary['Missing Count'] > 0])
    if missing_summary['Missing Count'].sum() == 0:
        print("✓ No missing values found!")
    print()
    
    # 3. Duplicates Analysis
    print("3. DUPLICATES ANALYSIS")
    print("-" * 40)
    total_duplicates = df.duplicated().sum()
    id_duplicates = df.duplicated(subset=['id']).sum()
    print(f"Total duplicate rows: {total_duplicates}")
    print(f"Duplicate IDs: {id_duplicates}")
    
    # Check for near-duplicates in text content
    if 'text' in df.columns:
        text_duplicates = df.duplicated(subset=['text']).sum()
        print(f"Duplicate text content: {text_duplicates}")
    print()
    
    # 4. Temporal Analysis
    print("4. TEMPORAL ANALYSIS")
    print("-" * 40)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        # Posts per day
        daily_posts = df.groupby(df['timestamp'].dt.date).size()
        print(f"Posts per day - Mean: {daily_posts.mean():.1f}, Median: {daily_posts.median():.1f}")
        print(f"Most active day: {daily_posts.idxmax()} ({daily_posts.max()} posts)")
        print(f"Least active day: {daily_posts.idxmin()} ({daily_posts.min()} posts)")
    print()
    
    # 5. Content Quality Analysis
    print("5. CONTENT QUALITY ANALYSIS")
    print("-" * 40)
    
    # Text content analysis
    text_cols = [col for col in ['text', 'title', 'selftext'] if col in df.columns]
    for col in text_cols:
        if col in df.columns:
            non_empty = df[col].notna() & (df[col] != '')
            avg_length = df[non_empty][col].str.len().mean()
            print(f"{col} - Non-empty: {non_empty.sum()}/{len(df)} ({non_empty.mean()*100:.1f}%)")
            print(f"{col} - Average length: {avg_length:.1f} characters")
    
    # Engagement metrics
    if 'score' in df.columns:
        print(f"\nEngagement Metrics:")
        print(f"Score - Mean: {df['score'].mean():.1f}, Median: {df['score'].median():.1f}")
        print(f"Score - Min: {df['score'].min()}, Max: {df['score'].max()}")
        
    if 'num_comments' in df.columns:
        print(f"Comments - Mean: {df['num_comments'].mean():.1f}, Median: {df['num_comments'].median():.1f}")
        print(f"Comments - Min: {df['num_comments'].min()}, Max: {df['num_comments'].max()}")
    print()
    
    # 6. Subreddit Distribution
    print("6. SUBREDDIT DISTRIBUTION")
    print("-" * 40)
    if 'subreddit' in df.columns:
        subreddit_counts = df['subreddit'].value_counts()
        print(f"Number of subreddits: {len(subreddit_counts)}")
        print("Top 10 subreddits:")
        for subreddit, count in subreddit_counts.head(10).items():
            print(f"  {subreddit}: {count} posts ({count/len(df)*100:.1f}%)")
    print()
    
    # 7. Search Methods Analysis
    print("7. DATA COLLECTION ANALYSIS")
    print("-" * 40)
    if 'search_method' in df.columns:
        method_counts = df['search_method'].value_counts()
        print("Search methods used:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} posts ({count/len(df)*100:.1f}%)")
    
    if 'search_keyword' in df.columns:
        keyword_counts = df['search_keyword'].value_counts()
        print(f"\nSearch keywords used:")
        for keyword, count in keyword_counts.items():
            print(f"  '{keyword}': {count} posts ({count/len(df)*100:.1f}%)")
    print()
    
    # 8. Research Suitability Assessment
    print("8. RESEARCH SUITABILITY ASSESSMENT")
    print("-" * 40)
    
    # Calculate quality scores
    quality_scores = {}
    
    # Data completeness (0-100)
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    quality_scores['completeness'] = completeness
    
    # Content richness (based on text length)
    if 'text' in df.columns:
        avg_text_length = df['text'].str.len().mean()
        content_richness = min(avg_text_length / 100, 100)  # Cap at 100
        quality_scores['content_richness'] = content_richness
    
    # Temporal coverage (more days = better)
    if 'timestamp' in df.columns:
        days_covered = (df['timestamp'].max() - df['timestamp'].min()).days
        temporal_coverage = min(days_covered / 30 * 100, 100)  # 30 days = 100%
        quality_scores['temporal_coverage'] = temporal_coverage
    
    # Engagement diversity
    if 'score' in df.columns:
        score_std = df['score'].std()
        engagement_diversity = min(score_std / 10, 100)  # Normalize
        quality_scores['engagement_diversity'] = engagement_diversity
    
    print("Quality Metrics:")
    for metric, score in quality_scores.items():
        status = "✓" if score > 70 else "⚠" if score > 40 else "✗"
        print(f"  {status} {metric.replace('_', ' ').title()}: {score:.1f}/100")
    
    overall_score = np.mean(list(quality_scores.values()))
    print(f"\nOverall Quality Score: {overall_score:.1f}/100")
    
    # Recommendations
    print("\n9. RECOMMENDATIONS")
    print("-" * 40)
    
    if overall_score > 80:
        print("✓ Dataset quality is EXCELLENT for research")
        print("  - Proceed with analysis")
        print("  - Consider this as your primary dataset")
    elif overall_score > 60:
        print("⚠ Dataset quality is GOOD for research")
        print("  - Can proceed with analysis")
        print("  - Consider collecting additional data for robustness")
    elif overall_score > 40:
        print("⚠ Dataset quality is MODERATE for research")
        print("  - Proceed with caution")
        print("  - Strongly recommend collecting more data")
    else:
        print("✗ Dataset quality is POOR for research")
        print("  - Not recommended for primary analysis")
        print("  - Collect significantly more data before proceeding")
    
    # Specific recommendations
    if len(df) < 1000:
        print("  - Consider collecting more posts (target: 1000+)")
    if 'timestamp' in df.columns:
        days_covered = (df['timestamp'].max() - df['timestamp'].min()).days
        if days_covered < 7:
            print("  - Extend temporal coverage (target: 7+ days)")
    if df.duplicated().sum() > 0:
        print("  - Remove duplicate entries")
    
    print("\n=== ASSESSMENT COMPLETE ===")

if __name__ == "__main__":
    # Default path
    csv_path = "data/raw/reddit_Taxas Trump Brasil_20250710_203130.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
    
    analyze_data_quality(csv_path)