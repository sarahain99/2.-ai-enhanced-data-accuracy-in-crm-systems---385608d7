import re
import pandas as pd
import numpy as np
from typing import Dict, Any
import phonenumbers
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation class"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.validation_report = {
            'errors': [],
            'warnings': [],
            'removed_rows': 0,
            'initial_count': len(df),
            'final_count': len(df)
        }
    
    def validate(self) -> Dict[str, Any]:
        """Run all validation checks"""
        try:
            self._validate_emails()
            self._validate_phones()
            self._check_required_fields()
            self._validate_dates()
            self._validate_postal_codes()
            self._check_value_ranges()
            
            self.validation_report['final_count'] = len(self.df)
            self.validation_report['removed_rows'] = (
                self.validation_report['initial_count'] - self.validation_report['final_count']
            )
            
            return self.validation_report
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise
    
    def _validate_emails(self):
        """Validate email addresses with comprehensive regex"""
        if 'email' not in self.df.columns:
            self.validation_report['warnings'].append("No email column found")
            return
            
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = ~self.df['email'].astype(str).str.match(email_regex, na=False)
        
        if invalid_emails.any():
            count = invalid_emails.sum()
            self.df = self.df[~invalid_emails]
            msg = f"Removed {count} rows with invalid email addresses"
            self.validation_report['errors'].append(msg)
            logger.info(msg)
    
    def _validate_phones(self):
        """Validate phone numbers using phonenumbers library"""
        if 'phone' not in self.df.columns:
            self.validation_report['warnings'].append("No phone column found")
            return
            
        def is_valid_phone(phone):
            try:
                if pd.isna(phone):
                    return False
                parsed = phonenumbers.parse(str(phone), "US")
                return phonenumbers.is_valid_number(parsed)
            except:
                return False
        
        invalid_phones = ~self.df['phone'].apply(is_valid_phone)
        if invalid_phones.any():
            count = invalid_phones.sum()
            self.df = self.df[~invalid_phones]
            msg = f"Removed {count} rows with invalid phone numbers"
            self.validation_report['errors'].append(msg)
            logger.info(msg)
    
    def _check_required_fields(self):
        """Check for required fields with missing values"""
        required_fields = ['name', 'email']  # Can be customized
        
        for field in required_fields:
            if field in self.df.columns:
                missing = self.df[field].isna()
                if missing.any():
                    count = missing.sum()
                    self.df = self.df[~missing]
                    msg = f"Removed {count} rows with missing {field} values"
                    self.validation_report['errors'].append(msg)
                    logger.info(msg)
            else:
                msg = f"Required field {field} not found in data"
                self.validation_report['warnings'].append(msg)
                logger.warning(msg)
    
    def _validate_dates(self):
        """Validate date fields"""
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        
        for col in date_cols:
            # Check if dates are in the future where inappropriate
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                future_dates = self.df[col] > pd.Timestamp.now()
                if future_dates.any():
                    count = future_dates.sum()
                    msg = f"Found {count} rows with future dates in {col}"
                    self.validation_report['warnings'].append(msg)
                    logger.warning(msg)
    
    def _validate_postal_codes(self):
        """Validate US and Canadian postal codes"""
        if 'postal_code' not in self.df.columns:
            return
            
        us_pattern = r'^\d{5}(?:-\d{4})?$'
        ca_pattern = r'^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$'
        
        def validate_postal(code):
            if pd.isna(code):
                return False
            code = str(code).upper().replace(' ', '')
            return (re.match(us_pattern, code) is not None) or \
                   (re.match(ca_pattern, code) is not None)
        
        invalid = ~self.df['postal_code'].apply(validate_postal)
        if invalid.any():
            count = invalid.sum()
            self.df = self.df[~invalid]
            msg = f"Removed {count} rows with invalid postal codes"
            self.validation_report['errors'].append(msg)
            logger.info(msg)
    
    def _check_value_ranges(self):
        """Validate numerical fields are within reasonable ranges"""
        num_cols = self.df.select_dtypes(include=np.number).columns
        
        for col in num_cols:
            # Example checks - customize per your data
            if 'age' in col.lower():
                invalid = (self.df[col] < 0) | (self.df[col] > 120)
                if invalid.any():
                    count = invalid.sum()
                    msg = f"Found {count} rows with invalid age values in {col}"
                    self.validation_report['warnings'].append(msg)
                    logger.warning(msg)
            
            if 'amount' in col.lower() or 'price' in col.lower():
                negative_values = self.df[col] < 0
                if negative_values.any():
                    count = negative_values.sum()
                    msg = f"Found {count} rows with negative values in {col}"
                    self.validation_report['warnings'].append(msg)
                    logger.warning(msg)

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function to use the DataValidator class"""
    validator = DataValidator(df)
    return validator.validate()