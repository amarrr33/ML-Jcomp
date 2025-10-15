"""
Dataset Loader Utility for SecureAI
Loads and processes the expanded CyberSecEval3 dataset
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and process the CyberSecEval3 expanded dataset"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.dataset_path = self.data_dir / "cyberseceval3-visual-prompt-injection-expanded.csv"
        self.sources_path = self.data_dir / "cyberseceval3-visual-prompt-injection-expanded_sources.json"
        
        self.df = None
        self.sources = None
        
    def load(self) -> pd.DataFrame:
        """Load the dataset and sources"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        
        logger.info(f"Loading source metadata from {self.sources_path}")
        with open(self.sources_path, 'r') as f:
            self.sources = json.load(f)
        
        # Create source lookup
        self.source_map = {s['id']: s for s in self.sources}
        
        # Add language column
        self.df['language'] = self.df['id'].astype(str).map(
            lambda x: self.source_map.get(x, {}).get('language', 'unknown')
        )
        
        logger.info(f"Loaded {len(self.df)} entries across {self.df['language'].nunique()} languages")
        return self.df
    
    def get_by_language(self, language: str) -> pd.DataFrame:
        """Filter dataset by language"""
        if self.df is None:
            self.load()
        return self.df[self.df['language'] == language].copy()
    
    def get_by_risk_category(self, category: str) -> pd.DataFrame:
        """Filter by risk category"""
        if self.df is None:
            self.load()
        return self.df[self.df['risk_category'] == category].copy()
    
    def get_by_injection_type(self, injection_type: str) -> pd.DataFrame:
        """Filter by injection type"""
        if self.df is None:
            self.load()
        return self.df[self.df['injection_type'] == injection_type].copy()
    
    def get_sample(self, n: int = 100, language: Optional[str] = None, 
                   stratify: bool = True) -> pd.DataFrame:
        """Get a sample of the dataset"""
        if self.df is None:
            self.load()
        
        df = self.df if language is None else self.get_by_language(language)
        
        if stratify and len(df) >= n:
            # Stratified sampling by risk category
            return df.groupby('risk_category', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // 2))
            ).head(n)
        else:
            return df.sample(min(n, len(df)))
    
    def get_train_test_split(self, test_size: float = 0.2, 
                             language: Optional[str] = None,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets"""
        if self.df is None:
            self.load()
        
        df = self.df if language is None else self.get_by_language(language)
        
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df['risk_category']
        )
        
        return train_df, test_df
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df is None:
            self.load()
        
        stats = {
            'total_entries': len(self.df),
            'languages': self.df['language'].value_counts().to_dict(),
            'risk_categories': self.df['risk_category'].value_counts().to_dict(),
            'injection_types': self.df['injection_type'].value_counts().to_dict(),
        }
        
        return stats
    
    def prepare_for_detection(self, df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Prepare dataset entries for detection pipeline"""
        if df is None:
            df = self.df if self.df is not None else self.load()
        
        entries = []
        for _, row in df.iterrows():
            entry = {
                'id': row['id'],
                'system_prompt': row['system_prompt'],
                'user_input_text': row['user_input_text'],
                'image_description': row['image_description'],
                'image_text': row['image_text'],
                'judge_question': row['judge_question'],
                'injection_technique': row['injection_technique'],
                'injection_type': row['injection_type'],
                'risk_category': row['risk_category'],
                'language': row.get('language', 'english')
            }
            entries.append(entry)
        
        return entries


if __name__ == "__main__":
    # Test the loader
    loader = DatasetLoader()
    df = loader.load()
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    stats = loader.get_statistics()
    print(f"\nTotal Entries: {stats['total_entries']:,}")
    
    print("\nLanguages:")
    for lang, count in sorted(stats['languages'].items()):
        print(f"  {lang.capitalize():15} {count:,} entries")
    
    print("\nRisk Categories:")
    for cat, count in sorted(stats['risk_categories'].items()):
        print(f"  {cat:25} {count:,} entries")
    
    print("\n" + "="*80)
