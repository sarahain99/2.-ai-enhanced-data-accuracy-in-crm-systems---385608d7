import streamlit as st
import pandas as pd
from utils.cleaner import CRMDataCleaner
import time
import base64
import io

def get_table_download_link(df):
    """Generates a download link for the cleaned DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="cleaned_crm_data.csv">Download cleaned data</a>'

def main():
    # Configure page
    st.set_page_config(
        page_title="CRM Data Cleaner Pro",
        page_icon="🧼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #1E90FF;
        }
        .st-b7 {
            color: #1E90FF;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧼 CRM Data Cleaner Pro")
    st.markdown("""
    Upload your CRM data to automatically:
    - 🚫 Remove duplicates (exact & fuzzy matches)
    - 🧹 Clean and standardize formats
    - ✅ Validate against business rules
    - 📥 Download analysis-ready data
    """)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload your raw CRM data in CSV format"
    )
    
    cleaner = CRMDataCleaner()
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            with st.expander("🔍 Raw Data Preview", expanded=True):
                st.dataframe(df.head())
                st.info(f"Original data: {len(df)} records")
            
            # Processing options
            st.sidebar.header("Cleaning Options")
            strict_mode = st.sidebar.checkbox(
                "Strict Validation Mode",
                value=True,
                help="Enable stricter validation rules"
            )
            
            if st.button("✨ Clean Data", key="clean_button"):
                with st.spinner("Cleaning your data..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    # Simulate progress for better UX
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    
                    # Clean the data
                    cleaned_df, report = cleaner.clean_dataset(df)
                    
                    # Display results
                    st.success("Data cleaning complete!")
                    st.balloons()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Cleaned Data Preview")
                        st.dataframe(cleaned_df.head())
                        st.info(f"Cleaned data: {len(cleaned_df)} records")
                    
                    with col2:
                        st.subheader("Cleaning Report")
                        
                        if report.get('cleaning_status') == 'failed':
                            st.error(f"Cleaning failed: {report.get('error')}")
                        else:
                            st.json({
                                "Records Removed": report.get('rows_removed', 0),
                                "Exact Duplicates Removed": report.get('exact_duplicates_removed', 0),
                                "Fuzzy Duplicates Removed": report.get('fuzzy_duplicates_removed', 0),
                                "Invalid Emails Removed": report.get('invalid_emails_removed', 0),
                                "Invalid Phones Removed": report.get('invalid_phones_removed', 0),
                                "Invalid Segments Removed": report.get('invalid_segments_removed', 0),
                                "Missing Required Fields": report.get('missing_required_fields', {})
                            })
                    
                    # Download section
                    st.markdown("### Download Cleaned Data")
                    st.markdown(get_table_download_link(cleaned_df), unsafe_allow_html=True)
                    
                    # Show complete cleaned data
                    if st.checkbox("Show full cleaned data"):
                        st.dataframe(cleaned_df)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()