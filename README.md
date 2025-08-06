# 🎯 Comprehensive AI Resume Screening Tool

An intelligent, automated resume screening system that uses advanced natural language processing and machine learning techniques to analyze and rank resumes against job descriptions.

## ✨ Features

### 🧠 Intelligent Keyword Extraction
- **Automatic Analysis**: Scans job descriptions to automatically extract relevant keywords
- **Pattern Recognition**: Identifies technical skills, experience levels, education requirements, and industry-specific terms
- **Weighted Categorization**: Assigns intelligent weights based on keyword importance (technical skills get higher weights)
- **Comprehensive Coverage**: Supports programming languages, frameworks, databases, cloud platforms, DevOps tools, and more

### ⚖️ Advanced Scoring System
- **Multi-Component Analysis**: 
  - **Keywords Matching**: Direct skill matching with fuzzy logic and frequency analysis
  - **Semantic Similarity**: TF-IDF vectorization with cosine similarity
  - **Education Matching**: Degree and certification verification
- **Fuzzy Matching**: Handles variations, abbreviations, and typos (e.g., "JS" matches "JavaScript")
- **Frequency Requirements**: High-priority keywords require multiple mentions for full credit
- **Customizable Weights**: Adjust the importance of different scoring components

### 📊 Comprehensive Analysis & Visualization
- **Interactive Dashboard**: Multiple visualization tabs showing score breakdowns and distributions
- **Detailed Reports**: Per-resume analysis with keyword matching details
- **Export Functionality**: Download results as CSV with comprehensive data
- **Real-time Preview**: See keyword extraction and analysis setup before processing

### 🔧 Technical Capabilities
- **Multiple File Formats**: Supports PDF and Word documents (.pdf, .docx, .doc)
- **Batch Processing**: Analyze multiple resumes simultaneously
- **Error Handling**: Robust error handling for corrupted or unsupported files
- **Performance Optimization**: Efficient text processing and analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd resume-screener
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run resume_screener.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 📋 Required Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
PyPDF2>=3.0.0
python-docx>=0.8.11
scikit-learn>=1.3.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.21.0
plotly>=5.15.0
```

## 🎮 How to Use

### 1. **Job Configuration**
- Enter the job title and paste the complete job description
- The system automatically extracts relevant keywords from the description
- Optionally add manual keywords with custom weights (1-10 scale)

### 2. **Scoring Configuration**
- Adjust weights for different scoring components:
  - **Keywords Weight**: Direct skill matching importance
  - **Similarity Weight**: Semantic similarity importance  
  - **Education Weight**: Educational requirements importance
- Set the selection cutoff percentage (recommended: 70-80%)

### 3. **Upload & Analyze**
- Upload one or more resume files (PDF or Word format)
- Click "Analyze Resumes" to start processing
- View real-time progress as files are processed

### 4. **Review Results**
- **Summary Dashboard**: Overall statistics and metrics
- **Interactive Visualizations**: Score breakdowns, distributions, and performance charts
- **Detailed Analysis**: Per-resume keyword matching and education analysis
- **Export Results**: Download comprehensive analysis as CSV

## 🔍 Analysis Components

### Keyword Matching (Default: 50% weight)
- **Automatic Extraction**: Identifies technical skills, tools, experience levels
- **Intelligent Weighting**: 
  - Technical skills (Python, React, AWS): Weight 8-9
  - Experience patterns (5+ years, senior): Weight 6-7
  - Education terms: Weight 4-5
  - Soft skills: Weight 3
- **Fuzzy Matching**: Handles variations and abbreviations
- **Frequency Analysis**: High-priority keywords need multiple mentions

### Semantic Similarity (Default: 30% weight)
- Uses TF-IDF vectorization and cosine similarity
- Captures overall job-resume alignment beyond specific keywords
- Considers context and related terms

### Education Matching (Default: 20% weight)
- Extracts degree requirements from job descriptions
- Matches against resume education information
- Supports various degree formats and abbreviations

## 🎯 Scoring Logic

### Keyword Frequency Requirements
- **High-Priority Keywords (Weight ≥6)**:
  - 4+ mentions: 100% credit
  - 3 mentions: 75% credit
  - 2 mentions: 50% credit
  - 1 mention: 25% credit
- **Regular Keywords**: Single mention gives full credit

### Match Quality Thresholds
- **Exact Match**: 100% score
- **High Quality (≥90%)**: Full weight
- **Good Quality (≥70%)**: 95% weight
- **Acceptable (≥60%)**: 90% weight
- **Below 60%**: Not counted

## 📈 Understanding Results

### Score Breakdown
- **Individual Scores**: Each component (keywords, similarity, education) scored 0-100%
- **Weighted Overall Score**: Combined score based on your weight configuration
- **Status**: Selected/Rejected based on cutoff threshold

### Detailed Analysis
- **Keywords Found**: Shows matched terms with confidence scores and frequency
- **Keywords Missing**: Important terms not found in the resume
- **Education Analysis**: Comparison of job requirements vs. resume education
- **Frequency Multipliers**: Shows how frequency affects high-priority keyword scoring

## ⚙️ Customization Options

### Manual Keywords
- Add specific requirements not captured by automatic extraction
- Set custom weights (1-10) based on importance
- Boost critical skills for your specific role

### Scoring Weights
- Adjust component weights based on role requirements:
  - **Technical roles**: Increase keyword weight
  - **Senior positions**: Balance all components
  - **Entry-level**: Increase education weight

### Selection Cutoff
- **Conservative (80-90%)**: For highly competitive positions
- **Balanced (70-80%)**: For most standard roles
- **Liberal (60-70%)**: For hard-to-fill positions

## 🚨 Troubleshooting

### Common Issues

**1. File Upload Errors**
- Ensure files are in PDF or Word format
- Check that files aren't password-protected or corrupted
- Try converting files to a different format if issues persist

**2. Poor Keyword Extraction**
- Make sure job description is comprehensive and detailed
- Add manual keywords for specific requirements
- Check that technical terms are spelled correctly

**3. Unexpected Scores**
- Verify scoring weights add up correctly (automatically normalized)
- Review the keyword preview to ensure relevant terms are captured
- Check frequency requirements for high-priority keywords

**4. Performance Issues**
- Large PDF files may take longer to process
- Consider processing resumes in smaller batches
- Ensure adequate system memory for large files

### Getting Help
- Check the "Analysis Preview" section to verify keyword extraction
- Review the detailed per-resume analysis for scoring explanations
- Adjust weights and thresholds based on your specific needs

## 🔬 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Text Processing**: PyPDF2 and python-docx for document parsing
- **NLP**: scikit-learn for TF-IDF vectorization and similarity
- **Fuzzy Matching**: fuzzywuzzy with Levenshtein distance
- **Visualization**: Plotly for interactive charts

### Performance
- Typical processing time: 2-5 seconds per resume
- Memory usage: ~50-100MB for standard document sets
- Batch processing: Supports dozens of resumes simultaneously

### Security
- All processing happens locally - no data sent to external services
- Files are processed in memory only
- No permanent storage of uploaded documents

## 📊 Example Use Cases

### Technology Roles
- Software engineers, data scientists, DevOps engineers
- Automatically detects programming languages, frameworks, tools
- Weighs technical skills heavily in scoring

### Business Roles
- Project managers, analysts, consultants
- Focuses on experience levels, soft skills, education
- Balanced scoring across all components

### Specialized Positions
- Add manual keywords for niche requirements
- Adjust weights based on role criticality
- Set appropriate cutoff thresholds

## 🤝 Contributing

This tool is designed to be extensible and customizable. Potential improvements:
- Additional file format support
- Enhanced NLP models
- Industry-specific keyword databases
- Integration with ATS systems

## 📄 License

This project is provided as-is for educational and business use. Please ensure compliance with local regulations regarding automated hiring tools and bias prevention.

---

**Built with ❤️ using Python, Streamlit, and modern NLP techniques** 