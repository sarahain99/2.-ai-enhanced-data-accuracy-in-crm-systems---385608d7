import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
from typing import Tuple, Dict
import logging

class CRMDataCleaner:
    def __init__(self):
        self.standard_address_mapping = {
            'street': 'St', 'st': 'St',
            'avenue': 'Ave', 'ave': 'Ave',
            'road': 'Rd', 'rd': 'Rd',
            'lane': 'Ln', 'ln': 'Ln',
            'drive': 'Dr', 'dr': 'Dr',
            'boulevard': 'Blvd', 'blvd': 'Blvd'
        }
        self.valid_segments = ['Enterprise', 'SMB', 'Mid-Market']
        self.required_fields = ['name', 'email']
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Main cleaning pipeline with enhanced error handling"""
        try:
            original_count = len(df)
            report = {'original_count': original_count}
            
            # Deep copy and initial cleaning
            df = df.copy(deep=True)
            df = self._convert_empty_strings_to_nan(df)
            df = self._standardize_column_names(df)
            
            # Data cleaning pipeline
            df, dup_report = self._handle_duplicates(df)
            report.update(dup_report)
            
            df = self._standardize_data_formats(df)
            
            df, validation_report = self._validate_and_clean(df)
            report.update(validation_report)
            
            df = self._final_cleanup(df)
            
            report.update({
                'cleaned_count': len(df),
                'rows_removed': original_count - len(df),
                'cleaning_status': 'success'
            })
            
            return df, report
            
        except Exception as e:
            self.logger.error(f"Cleaning failed: {str(e)}")
            return pd.DataFrame(), {
                'cleaning_status': 'failed',
                'error': str(e)
            }

    def _convert_empty_strings_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all empty/whitespace strings to NaN"""
        return df.replace(r'^\s*$', np.nan, regex=True)

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to snake_case"""
        df.columns = (
            df.columns
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.strip('_')
        )
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced duplicate handling with multiple strategies"""
        report = {}
        original_count = len(df)
        
        # 1. Remove exact duplicates
        df = df.drop_duplicates()
        exact_dups = original_count - len(df)
        
        # 2. Fuzzy matching on key fields
        if all(col in df.columns for col in ['name', 'email', 'phone']):
            df, fuzzy_dups = self._remove_fuzzy_duplicates(df)
        else:
            fuzzy_dups = 0
            
        report.update({
            'exact_duplicates_removed': exact_dups,
            'fuzzy_duplicates_removed': fuzzy_dups,
            'total_duplicates_removed': exact_dups + fuzzy_dups
        })
        
        return df, report

    def _remove_fuzzy_duplicates(self, df: pd.DataFrame, threshold: int = 88) -> Tuple[pd.DataFrame, int]:
        """Advanced fuzzy matching with multiple field comparisons"""
        from collections import defaultdict
        
        # Group similar records
        groups = defaultdict(list)
        for idx, row in df.iterrows():
            key = (
                self._normalize_text(row.get('name', '')),
                self._normalize_text(row.get('email', '').split('@')[0]),
                str(row.get('phone', ''))[:7]  # First 7 digits only
            )
            groups[key].append(idx)
        
        # Select best record from each group
        keep_indices = []
        removed = 0
        
        for group_indices in groups.values():
            if len(group_indices) > 1:
                # Score records based on completeness
                records = df.loc[group_indices]
                scores = records.notna().sum(axis=1)
                best_idx = scores.idxmax()
                keep_indices.append(best_idx)
                removed += len(group_indices) - 1
            else:
                keep_indices.append(group_indices[0])
        
        return df.loc[keep_indices].reset_index(drop=True), removed

    def _normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy matching"""
        if pd.isna(text):
            return ''
        return (
            str(text)
            .lower()
            .translate(str.maketrans('', '', '.,!?;:'))
            .replace(' ', '')
        )

    def _standardize_data_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data standardization"""
        # Address standardization
        if 'address' in df.columns:
            df['address'] = (
                df['address']
                .astype(str)
                .str.lower()
                .apply(self._standardize_address)
                .str.title()
            )
        
        # Phone standardization
        if 'phone' in df.columns:
            df['phone'] = (
                df['phone']
                .astype(str)
                .str.replace(r'[^\d]', '', regex=True)
                .apply(lambda x: f"{x[:3]}-{x[3:7]}" if len(x) >= 7 else np.nan)
            )
        
        # Email standardization
        if 'email' in df.columns:
            df['email'] = (
                df['email']
                .str.lower()
                .str.strip()
                .replace(r'^\s*$', np.nan, regex=True)
            )
        
        # Company name standardization
        if 'company' in df.columns:
            df['company'] = (
                df['company']
                .str.strip()
                .str.title()
                .replace(r'^\s*$', np.nan, regex=True)
            )
        
        return df

    def _standardize_address(self, address: str) -> str:
        """Standardize address components"""
        if pd.isna(address):
            return address
            
        for full, abbrev in self.standard_address_mapping.items():
            address = re.sub(rf'\b{full}\b', abbrev, address)
        return address

    def _validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Strict data validation with comprehensive reporting"""
        report = {}
        original_count = len(df)
        
        # Email validation
        if 'email' in df.columns:
            email_mask = (
                df['email']
                .str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)
            )
            report['invalid_emails_removed'] = (~email_mask).sum()
            df = df[email_mask]
        
        # Phone validation
        if 'phone' in df.columns:
            phone_mask = df['phone'].str.match(r'^\d{3}-\d{4}$', na=False)
            report['invalid_phones_removed'] = (~phone_mask).sum()
            df = df[phone_mask]
        
        # Segment validation
        if 'segment' in df.columns:
            segment_mask = df['segment'].isin(self.valid_segments) | df['segment'].isna()
            report['invalid_segments_removed'] = (~segment_mask).sum()
            df = df[segment_mask]
        
        # Required fields validation
        missing_fields = {}
        for field in self.required_fields:
            if field in df.columns:
                missing_mask = df[field].isna()
                missing_fields[field] = missing_mask.sum()
                df = df[~missing_mask]
        report['missing_required_fields'] = missing_fields
        
        report['rows_removed_during_validation'] = original_count - len(df)
        return df, report

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data polishing"""
        # Fill missing segments
        if 'segment' in df.columns:
            df['segment'] = df['segment'].fillna('Unknown')
        
        # Ensure proper data types
        if 'customer_id' in df.columns:
            df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
        
        if 'last_purchase_date' in df.columns:
            df['last_purchase_date'] = pd.to_datetime(
                df['last_purchase_date'], 
                errors='coerce'
            )
        
        # Sort by customer_id if exists
        if 'customer_id' in df.columns:
            df = df.sort_values('customer_id')
        
        return df.reset_index(drop=True)