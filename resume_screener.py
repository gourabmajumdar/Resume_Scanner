import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import re
from typing import List, Dict, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Document processing imports
try:
    import PyPDF2
except ImportError:
    st.error("Please install PyPDF2: pip install PyPDF2")

try:
    from docx import Document
except ImportError:
    st.error("Please install python-docx: pip install python-docx")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("Please install scikit-learn: pip install scikit-learn")

try:
    from fuzzywuzzy import fuzz, process
except ImportError:
    st.error("Please install fuzzywuzzy: pip install fuzzywuzzy python-Levenshtein")

class ComprehensiveResumeAnalyzer:
    def __init__(self):
        self.job_keywords = []
        self.job_description = ""
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree', 'diploma', 'certificate',
            'bs', 'ba', 'ms', 'ma', 'mba', 'computer science', 'engineering', 
            'information technology', 'software engineering', 'data science',
            'mathematics', 'statistics', 'physics', 'chemistry', 'biology'
        ]
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text based on file type."""
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_extension in ['docx', 'doc']:
            return self.extract_text_from_docx(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.\-]', ' ', text)  # Keep +, #, ., - for technologies
        return text.strip()
    
    def extract_keywords_from_job_description(self, job_desc: str) -> List[str]:
        """Basic keyword extraction from job description."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'required', 'experience', 'years', 'work', 'working', 'knowledge'
            # ADD THESE NEW ONES TO FILTER OUT GARBAGE:
            'such', 'like', 'large', 'strong', 'excellent', 'highly', 'provide', 
            'similar', 'role', 'ability', 'focus', 'key', 'create', 'complex',
            'outstanding', 'technical', 'critical', 'methods', 'foundation',
            'related', 'desirable', 'preferred', 'equivalent', 'attention',
            'qualifications', 'plus', 'field', 'trends', 'insights', 'reports',
            'tools', 'skills', 'building', 'analyze', 'platforms', 'datasets'
        }
        
        processed_text = self.preprocess_text(job_desc)
        words = processed_text.split()
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        unique_keywords = list(set(keywords))
        
        return unique_keywords
    
    def extract_advanced_keywords_from_job_description(self, job_desc: str) -> Dict[str, float]:
        """Advanced keyword extraction with categorization and intelligent weighting."""
        
        job_desc_lower = job_desc.lower()
        extracted_keywords = {}
        
        # Technical skills patterns (highest weight)
        technical_patterns = {
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|scala|kotlin)\b': 9,
            # Web frameworks
            r'\b(react|angular|vue|django|flask|spring|express|laravel|rails)\b': 8,
            # Databases
            r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra|oracle)\b': 8,
            # Cloud platforms
            r'\b(aws|azure|gcp|google cloud|amazon web services|snowflake|microsoft azure)\b': 8,
            # DevOps tools
            r'\b(docker|kubernetes|jenkins|gitlab|github|terraform|gitops|maven|ant|ansible)\b': 7,
            # Data Analyst
            r'\b(power bi|tableau|excel|powerpoint|word)\b': 9,
            # Data science/ML
            r'\b(machine learning|ml|ai|artificial intelligence|predictive modeling|deep learning|tensorflow|pytorch|pandas|numpy|scikit-learn)\b': 9,
            # Mobile development
            r'\b(android|ios|react native|flutter|swift|kotlin|xamarin)\b': 8,
            # Release Management
            r'\b(puppet|ansible|chef|salt|release management|build|scrum|sprint|release|release management|build master|ansible)\b': 9,
            # Testing
            r'\b(junit|pytest|selenium|playwright|cypress|jest|testing|qa|A/B testing|bdd|cucumber|mocha|jasmine|postman|rest|soap|restful|rest api|restful api|jmeter|loadrunner|automation|manual testing|regression testing|smoke testing|unit testing|quality assurance)\b': 8,
        }
        
        # Experience level patterns (high weight)
        experience_patterns = {
            r'\b(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)\b': 7,
            r'\b(senior|lead|principal|architect|manager|director)\b': 6,
            r'\b(junior|entry.level|graduate|intern|trainee)\b': 5,
            r'\b(mid.level|intermediate|associate)\b': 5,
        }
        
        # Education patterns (medium weight)
        education_patterns = {
            r'\b(bachelor|master|phd|doctorate|degree|computer science|engineering)\b': 5,
            r'\b(certification|certified|certificate|diploma)\b': 4,
            r'\b(bs|ba|ms|ma|mba|phd)\b': 5,
        }
        
        # Industry/Domain patterns (medium weight)
        domain_patterns = {
            r'\b(fintech|healthcare|e-commerce|saas|startup|enterprise)\b': 4,
            r'\b(agile|scrum|devops|ci/cd|microservices|api)\b': 5,
            r'\b(frontend|backend|full.stack|fullstack)\b': 6,
        }
        
        # Soft skills patterns (lower weight)
        soft_skills_patterns = {
            r'\b(leadership|communication|teamwork|problem-solving|analytical)\b': 3,
            r'\b(creative|innovative|collaborative|thinking|outstanding|adaptable)\b': 3,
        }
        
        # Combine all patterns
        all_patterns = {
            **technical_patterns, 
            **experience_patterns, 
            **education_patterns, 
            **domain_patterns,
            **soft_skills_patterns
        }
        
        # Extract keywords using patterns
        for pattern, weight in all_patterns.items():
            matches = re.findall(pattern, job_desc_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(filter(None, match)).strip()
                if match and len(match) > 1:
                    # Use the highest weight if keyword appears in multiple patterns
                    extracted_keywords[match] = max(extracted_keywords.get(match, 0), weight)
        
        # Basic word extraction for anything missed (lowest weight)
        #basic_keywords = self.extract_keywords_from_job_description(job_desc)
        #for keyword in basic_keywords:
        #    if keyword not in extracted_keywords and len(keyword) > 2:
        #        extracted_keywords[keyword] = 2  # Default weight for basic extraction
        
        return extracted_keywords
    
    def create_comprehensive_keyword_set(self, job_description: str, manual_keywords: Dict[str, float]) -> Dict[str, float]:
        """Create comprehensive keyword set combining job description analysis and manual input."""
        
        # Step 1: Always extract keywords from job description
        auto_extracted = self.extract_advanced_keywords_from_job_description(job_description)
        
        # Step 2: Start with auto-extracted keywords
        comprehensive_keywords = auto_extracted.copy()
        
        # Step 3: Add/Override with manual keywords (higher priority)
        for keyword, weight in manual_keywords.items():
            keyword_lower = keyword.lower().strip()
            if keyword_lower in comprehensive_keywords:
                # If keyword exists in both, boost it (manual input shows it's important)
                comprehensive_keywords[keyword_lower] = max(weight, comprehensive_keywords[keyword_lower] + 1)
            else:
                # New manual keyword
                comprehensive_keywords[keyword_lower] = weight
        
        # Step 4: Remove very low-value keywords to focus on important ones
        comprehensive_keywords = {k: v for k, v in comprehensive_keywords.items() if v >= 2}
        
        return comprehensive_keywords
    
    def extract_education_requirements(self, job_desc: str) -> List[str]:
        """Extract education requirements from job description."""
        job_desc_lower = job_desc.lower()
        found_education = []
        
        # Look for education keywords
        for edu_keyword in self.education_keywords:
            if edu_keyword in job_desc_lower:
                found_education.append(edu_keyword)
        
        # Look for specific degree patterns
        degree_patterns = [
            r'bachelor[\'s]*\s+(?:degree\s+)?(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'master[\'s]*\s+(?:degree\s+)?(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'phd\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'bs\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'ms\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, job_desc_lower)
            found_education.extend(matches)
        
        return list(set(found_education))
    
    def extract_education_from_resume(self, resume_text: str) -> List[str]:
        """Extract education information from resume."""
        resume_lower = resume_text.lower()
        found_education = []
        
        # Look for education keywords
        for edu_keyword in self.education_keywords:
            if edu_keyword in resume_lower:
                found_education.append(edu_keyword)
        
        # Look for specific degree patterns
        degree_patterns = [
            r'bachelor[\'s]*\s+(?:degree\s+)?(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'master[\'s]*\s+(?:degree\s+)?(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'phd\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'bs\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'ms\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
            r'mba',
            r'certificate\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, resume_lower)
            found_education.extend(matches)
        
        return list(set(found_education))
    
    def fuzzy_keyword_match(self, keyword: str, text: str, threshold: int = 80) -> Dict:
        """Find fuzzy matches for keywords in text with improved logic."""
        keyword = keyword.lower().strip()
        words = text.split()
        
        # Check for exact match first
        if keyword in text:
            return {'matched_text': keyword, 'score': 100}
        
        # Handle common abbreviations and variations
        abbreviations = {
            'javascript': ['js', 'javascript', 'java script'],
            'machine learning': ['ml', 'machine learning', 'machine-learning'],
            'artificial intelligence': ['ai', 'artificial intelligence'],
            'react.js': ['react', 'reactjs', 'react js'],
            'node.js': ['node', 'nodejs', 'node js'],
            'postgresql': ['postgres', 'postgresql', 'postgre'],
            'mongodb': ['mongo', 'mongodb', 'mongo db'],
        }
        
        # Check abbreviations
        for full_term, variants in abbreviations.items():
            if keyword in variants or any(variant in keyword for variant in variants):
                for variant in variants:
                    if variant in text:
                        return {'matched_text': variant, 'score': 95}
        
        # For multi-word keywords, use sliding window approach
        if len(keyword.split()) > 1:
            keyword_words = keyword.split()
            window_size = len(keyword_words)
            
            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i + window_size])
                score = fuzz.ratio(keyword, window)
                
                if score >= threshold:
                    return {'matched_text': window, 'score': score}
        
        # Single word fuzzy matching
        best_match = process.extractOne(keyword, words, scorer=fuzz.ratio)
        
        if best_match and best_match[1] >= threshold:
            return {'matched_text': best_match[0], 'score': best_match[1]}
        
        return None
    
    def fuzzy_keyword_match_with_frequency(self, keyword: str, text: str, threshold: int = 80) -> Dict:
        """Find fuzzy matches with frequency counting for keywords in text."""
        keyword = keyword.lower().strip()
        words = text.split()
        
        matches = []
        match_scores = []
        
        # Check for exact matches first
        exact_count = text.lower().count(keyword.lower())
        if exact_count > 0:
            return {
                'found': {'matched_text': keyword, 'score': 100},
                'frequency': exact_count,
                'score': 100
            }
        
        # Handle common abbreviations and variations
        abbreviations = {
            'javascript': ['js', 'javascript', 'java script'],
            'machine learning': ['ml', 'machine learning', 'machine-learning'],
            'artificial intelligence': ['ai', 'artificial intelligence'],
            'react.js': ['react', 'reactjs', 'react js'],
            'node.js': ['node', 'nodejs', 'node js'],
            'postgresql': ['postgres', 'postgresql', 'postgre'],
            'mongodb': ['mongo', 'mongodb', 'mongo db'],
        }
        
        # Check abbreviations with frequency
        for full_term, variants in abbreviations.items():
            if keyword in variants or any(variant in keyword for variant in variants):
                total_frequency = 0
                best_match = None
                for variant in variants:
                    variant_count = text.lower().count(variant.lower())
                    if variant_count > 0:
                        total_frequency += variant_count
                        if not best_match:
                            best_match = {'matched_text': variant, 'score': 95}
                
                if total_frequency > 0:
                    return {
                        'found': best_match,
                        'frequency': total_frequency,
                        'score': 95
                    }
        
        # For multi-word keywords, use sliding window approach
        if len(keyword.split()) > 1:
            keyword_words = keyword.split()
            window_size = len(keyword_words)
            frequency = 0
            best_match = None
            
            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i + window_size])
                score = fuzz.ratio(keyword, window)
                
                if score >= threshold:
                    frequency += 1
                    if not best_match or score > best_match['score']:
                        best_match = {'matched_text': window, 'score': score}
            
            if frequency > 0:
                return {
                    'found': best_match,
                    'frequency': frequency,
                    'score': best_match['score']
                }
        
        # Single word fuzzy matching with frequency
        word_matches = []
        for word in words:
            score = fuzz.ratio(keyword, word)
            if score >= threshold:
                word_matches.append((word, score))
        
        if word_matches:
            # Get the best match and count similar matches
            best_word, best_score = max(word_matches, key=lambda x: x[1])
            frequency = len([w for w, s in word_matches if s >= threshold * 0.9])  # Count matches within 90% of threshold
            
            return {
                'found': {'matched_text': best_word, 'score': best_score},
                'frequency': frequency,
                'score': best_score
            }
        
        return None

    def calculate_weighted_keyword_score(self, resume_text: str, weighted_keywords: Dict[str, float]) -> Tuple[float, Dict]:
        """Calculate weighted keyword matching score with frequency requirements for important keywords.""" 
        resume_text = self.preprocess_text(resume_text)
        
        total_weight = 0
        matched_weight = 0
        keyword_details = {}
        
        for keyword, weight in weighted_keywords.items():
            total_weight += weight
            
            # Use enhanced fuzzy matching with frequency counting
            found_results = self.fuzzy_keyword_match_with_frequency(keyword, resume_text)
            print(f"DEBUG: {keyword} -> frequency: {found_results['frequency'] if found_results else 'None'}")

            keyword_details[keyword] = {
                'weight': weight,
                'found': found_results['found'] if found_results else None,
                'match_score': found_results['score'] if found_results else 0,
                'frequency': found_results['frequency'] if found_results else 0,
                'contributed_weight': 0
            }
            
            if found_results and found_results['score'] > 50:
                frequency = found_results['frequency']
                
                # Enhanced scoring logic with stricter frequency requirements
                if weight >= 6:  # High-importance keywords need multiple mentions
                    if frequency > 3:
                        frequency_multiplier = 1.0   # 100% credit for 4+ mentions
                    elif frequency == 3:
                        frequency_multiplier = 0.75  # 75% credit for 3 mentions
                    elif frequency == 2:
                        frequency_multiplier = 0.5   # 50% credit for 2 mentions
                    else:  # frequency == 1
                        frequency_multiplier = 0.25  # 25% credit for single mention
                else:  # Regular keywords - single mention is fine
                    frequency_multiplier = 1.0
                
                # Base contribution based on fuzzy match quality
                if found_results['score'] >= 70:
                    base_contribution = weight  # Full weight
                elif found_results['score'] >= 60:
                    base_contribution = weight * 0.95  # 95% weight
                else:
                    base_contribution = weight * 0.9   # 90% weight
                
                # Apply frequency multiplier
                contribution = base_contribution * frequency_multiplier
                matched_weight += contribution
                keyword_details[keyword]['contributed_weight'] = contribution
                keyword_details[keyword]['frequency_multiplier'] = frequency_multiplier
    
        if total_weight == 0:
            return 0.0, keyword_details
        
        score = (matched_weight / total_weight) * 100
        return score, keyword_details

    def calculate_education_match_score(self, job_education: List[str], resume_education: List[str]) -> float:
        """Calculate education matching score with improved logic."""
        if not job_education:
            return 100.0  # No education requirements
        
        if not resume_education:
            return 0.0  # No education found in resume
        
        matched_education = 0
        total_education = len(job_education)
        
        for job_edu in job_education:
            for resume_edu in resume_education:
                # Use fuzzy matching for education
                similarity = fuzz.partial_ratio(job_edu.lower(), resume_edu.lower())
                if similarity > 70:  # 70% threshold for education matching
                    matched_education += 1
                    break
        
        return (matched_education / total_education) * 100
    
    def calculate_similarity_score(self, resume_text: str, job_description: str) -> float:
        """Calculate semantic similarity using TF-IDF and cosine similarity."""
        try:
            resume_clean = self.preprocess_text(resume_text)
            job_clean = self.preprocess_text(job_description)
            
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000,
                ngram_range=(1, 2)  # Include both unigrams and bigrams
            )
            tfidf_matrix = vectorizer.fit_transform([job_clean, resume_clean])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity * 100
        except Exception as e:
            st.warning(f"Could not calculate similarity score: {str(e)}")
            return 0.0
    
    def analyze_resume(self, resume_text: str, job_description: str, 
                      manual_weighted_keywords: Dict[str, float], 
                      scoring_weights: Dict[str, float]) -> Dict:
        """Perform comprehensive resume analysis with all enhancements."""
        
        # Step 1: Create comprehensive keyword set (always includes job description analysis)
        comprehensive_keywords = self.create_comprehensive_keyword_set(
            job_description, manual_weighted_keywords
        )
        
        # Step 2: Extract education requirements
        job_education = self.extract_education_requirements(job_description)
        resume_education = self.extract_education_from_resume(resume_text)
        
        # Step 3: Calculate all scores
        keyword_score, keyword_details = self.calculate_weighted_keyword_score(
            resume_text, comprehensive_keywords
        )
        similarity_score = self.calculate_similarity_score(resume_text, job_description)
        education_score = self.calculate_education_match_score(job_education, resume_education)
        
        # Step 4: Calculate overall score using custom weights
        overall_score = (
            keyword_score * scoring_weights.get('keywords', 0.5) +
            similarity_score * scoring_weights.get('similarity', 0.3) +
            education_score * scoring_weights.get('education', 0.2)
        )
        
        # Step 5: Determine status based on cutoff
        cutoff = scoring_weights.get('cutoff', 90)
        status = "Selected" if overall_score >= cutoff else "Rejected"
        
        # Step 6: Prepare detailed analysis
        auto_extracted = self.extract_advanced_keywords_from_job_description(job_description)
        
        return {
            'keyword_score': round(keyword_score, 2),
            'similarity_score': round(similarity_score, 2),
            'education_score': round(education_score, 2),
            'overall_score': round(overall_score, 2),
            'status': status,
            'keyword_details': keyword_details,
            'comprehensive_keywords': comprehensive_keywords,
            'auto_extracted_keywords': auto_extracted,
            'auto_extracted_count': len(auto_extracted),
            'manual_keywords_count': len(manual_weighted_keywords),
            'total_keywords_used': len(comprehensive_keywords),
            'job_education': job_education,
            'resume_education': resume_education
        }

def main():
    st.set_page_config(
        page_title="Comprehensive Resume Screening Tool",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .score-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .selected {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .rejected {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .keyword-found {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .keyword-missing {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .analysis-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎯 Comprehensive AI Resume Screening Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Enhanced with intelligent keyword extraction, fuzzy matching, and comprehensive analysis**")
    
    analyzer = ComprehensiveResumeAnalyzer()
    
    # Sidebar for configuration
    st.sidebar.header("📋 Job Configuration")
    
    job_title = st.sidebar.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
    
    job_description = st.sidebar.text_area(
        "Job Description",
        height=250,
        placeholder="Paste the complete job description here...",
        help="The system will automatically extract keywords from this description and combine them with your manual keywords."
    )
    
    # Weighted Keywords Section
    st.sidebar.subheader("⚖️ Manual Keywords (Optional)")
    st.sidebar.write("Add specific keywords with importance weights (1-10)")
    st.sidebar.info("💡 The system automatically extracts keywords from the job description. Use this section to add specific requirements or boost important skills.")
    
    # Initialize session state for keywords
    if 'weighted_keywords' not in st.session_state:
        st.session_state.weighted_keywords = {}
    
    # Add new keyword interface
    with st.sidebar.expander("Add New Keyword", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            new_keyword = st.text_input("Keyword", key="new_keyword", placeholder="e.g., Python")
        with col2:
            keyword_weight = st.number_input("Weight", min_value=1, max_value=10, value=5, key="keyword_weight")
        
        if st.button("Add Keyword", type="primary"):
            if new_keyword.strip():
                st.session_state.weighted_keywords[new_keyword.strip().lower()] = keyword_weight
                st.rerun()
    
    # Display current manual keywords
    if st.session_state.weighted_keywords:
        st.sidebar.write("**Current Manual Keywords:**")
        for keyword, weight in st.session_state.weighted_keywords.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"• {keyword} (weight: {weight})")
            with col2:
                if st.button("❌", key=f"remove_{keyword}"):
                    del st.session_state.weighted_keywords[keyword]
                    st.rerun()
    
    # Clear all keywords button
    if st.session_state.weighted_keywords:
        if st.sidebar.button("🗑️ Clear All Manual Keywords"):
            st.session_state.weighted_keywords = {}
            st.rerun()
    
    # Scoring Configuration
    st.sidebar.subheader("📊 Scoring Configuration")
    
    keyword_weight = st.sidebar.slider("Keywords Weight", 0.0, 1.0, 0.5, 0.05)
    similarity_weight = st.sidebar.slider("Similarity Weight", 0.0, 1.0, 0.3, 0.05)
    education_weight = st.sidebar.slider("Education Weight", 0.0, 1.0, 0.2, 0.05)
    
    # Normalize weights to sum to 1
    total_weight = keyword_weight + similarity_weight + education_weight
    if total_weight > 0:
        keyword_weight /= total_weight
        similarity_weight /= total_weight
        education_weight /= total_weight
    
    cutoff_score = st.sidebar.slider("Selection Cutoff (%)", 60, 100, 70, 1)
    
    scoring_weights = {
        'keywords': keyword_weight,
        'similarity': similarity_weight,
        'education': education_weight,
        'cutoff': cutoff_score
    }
    
    # Display normalized weights
    st.sidebar.write("**Normalized Weights:**")
    st.sidebar.write(f"Keywords: {keyword_weight:.1%}")
    st.sidebar.write(f"Similarity: {similarity_weight:.1%}")
    st.sidebar.write(f"Education: {education_weight:.1%}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload multiple resume files in PDF or Word format"
        )
        
        # Show preview of analysis setup
        if job_description:
            with st.expander("🔍 Analysis Preview", expanded=False):
                auto_keywords = analyzer.extract_advanced_keywords_from_job_description(job_description)
                comprehensive_keywords = analyzer.create_comprehensive_keyword_set(job_description, st.session_state.weighted_keywords)
                
                col1_preview, col2_preview = st.columns(2)
                with col1_preview:
                    st.write(f"**Auto-extracted keywords:** {len(auto_keywords)}")
                    if auto_keywords:
                        top_auto = sorted(auto_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
                        for keyword, weight in top_auto:
                            st.write(f"• {keyword} ({weight})")
                
                with col2_preview:
                    st.write(f"**Manual keywords:** {len(st.session_state.weighted_keywords)}")
                    st.write(f"**Total keywords for analysis:** {len(comprehensive_keywords)}")
                
                # NEW SECTION: Show all keywords that will be used
                st.markdown("---")
                st.markdown("#### 📋 **Complete Keyword List for Analysis**")
                
                # Categorize keywords by weight
                high_weight_keywords = []
                medium_weight_keywords = []
                low_weight_keywords = []
                
                for keyword, weight in comprehensive_keywords.items():
                    if weight >= 7:
                        high_weight_keywords.append((keyword, weight))
                    elif weight >= 4:
                        medium_weight_keywords.append((keyword, weight))
                    else:
                        low_weight_keywords.append((keyword, weight))
                
                # Display keywords by category
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if high_weight_keywords:
                        st.markdown("**🔥 High Priority (7-10)**")
                        for keyword, weight in sorted(high_weight_keywords, key=lambda x: x[1], reverse=True):
                            st.write(f"• **{keyword}** ({weight})")
                
                with col2:
                    if medium_weight_keywords:
                        st.markdown("**⚖️ Medium Priority (4-6)**")
                        for keyword, weight in sorted(medium_weight_keywords, key=lambda x: x[1], reverse=True):
                            st.write(f"• {keyword} ({weight})")
                
                with col3:
                    if low_weight_keywords:
                        st.markdown("**📝 Low Priority (2-3)**")
                        for keyword, weight in sorted(low_weight_keywords, key=lambda x: x[1], reverse=True):
                            st.write(f"• {keyword} ({weight})")
                
                # Show frequency requirements
                high_weight_count = len(high_weight_keywords)
                if high_weight_count > 0:
                    st.info(f"🎯 **Frequency Requirements:** {high_weight_count} high-priority keywords require multiple mentions for full credit (4+ mentions = 100%, 3 = 75%, 2 = 50%, 1 = 25%)")
                
        if st.button("🔍 Analyze Resumes", type="primary", use_container_width=True):
            if not job_description.strip():
                st.error("Please enter a job description first!")
                return
            
            if not uploaded_files:
                st.error("Please upload at least one resume!")
                return
            
            # Show analysis setup
            comprehensive_keywords = analyzer.create_comprehensive_keyword_set(
                job_description, st.session_state.weighted_keywords
            )
            
            st.success(f"✅ Analysis starting with {len(comprehensive_keywords)} total keywords")
            
            # Initialize results
            results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze each resume
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"📄 Analyzing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                
                # Extract text from resume
                resume_text = analyzer.extract_text_from_file(uploaded_file)
                
                if resume_text:
                    # Analyze resume
                    analysis = analyzer.analyze_resume(
                        resume_text, job_description, 
                        st.session_state.weighted_keywords, scoring_weights
                    )
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'keyword_score': analysis['keyword_score'],
                        'similarity_score': analysis['similarity_score'],
                        'education_score': analysis['education_score'],
                        'overall_score': analysis['overall_score'],
                        'status': analysis['status'],
                        'analysis_details': analysis
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Analysis complete!")
            
            # Display results
            if results:
                st.header("📊 Comprehensive Analysis Results")
                
                # Create DataFrame for summary
                df = pd.DataFrame([
                    {
                        'filename': r['filename'],
                        'keyword_score': r['keyword_score'],
                        'similarity_score': r['similarity_score'],
                        'education_score': r['education_score'],
                        'overall_score': r['overall_score'],
                        'status': r['status']
                    } for r in results
                ])
                
                # Summary statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Resumes", len(df))
                
                with col2:
                    selected_count = len(df[df['status'] == 'Selected'])
                    st.metric("Selected", selected_count, delta=f"{selected_count/len(df)*100:.1f}%")
                
                with col3:
                    rejected_count = len(df[df['status'] == 'Rejected'])
                    st.metric("Rejected", rejected_count)
                
                with col4:
                    avg_score = df['overall_score'].mean()
                    st.metric("Avg Score", f"{avg_score:.1f}%")
                
                with col5:
                    max_score = df['overall_score'].max()
                    st.metric("Highest Score", f"{max_score:.1f}%")
                
                # Enhanced Visualization
                st.subheader("📈 Score Analysis Dashboard")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📊 Score Breakdown", "📈 Distribution", "🎯 Performance"])
                
                with tab1:
                    # Stacked bar chart showing score components
                    fig = go.Figure(data=[
                        go.Bar(name='Keywords', x=df['filename'], 
                              y=df['keyword_score'] * scoring_weights['keywords'],
                              marker_color='#1f77b4'),
                        go.Bar(name='Similarity', x=df['filename'], 
                              y=df['similarity_score'] * scoring_weights['similarity'],
                              marker_color='#ff7f0e'),
                        go.Bar(name='Education', x=df['filename'], 
                              y=df['education_score'] * scoring_weights['education'],
                              marker_color='#2ca02c')
                    ])
                    
                    fig.update_layout(
                        barmode='stack',
                        title='Weighted Score Components by Resume',
                        xaxis_tickangle=-45,
                        yaxis_title='Weighted Score Contribution',
                        height=500
                    )
                    
                    # Add cutoff line
                    fig.add_hline(y=cutoff_score, line_dash="dash", line_color="red", 
                                 annotation_text=f"Selection Cutoff: {cutoff_score}%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Score distribution histogram
                    fig = px.histogram(df, x='overall_score', nbins=10, 
                                     title='Overall Score Distribution',
                                     labels={'overall_score': 'Overall Score (%)', 'count': 'Number of Resumes'})
                    fig.add_vline(x=cutoff_score, line_dash="dash", line_color="red",
                                 annotation_text=f"Cutoff: {cutoff_score}%")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Scatter plot: Keywords vs Similarity
                    fig = px.scatter(df, x='keyword_score', y='similarity_score', 
                                   size='education_score', color='status',
                                   hover_data=['filename', 'overall_score'],
                                   title='Keywords vs Similarity Performance',
                                   color_discrete_map={'Selected': '#28a745', 'Rejected': '#dc3545'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Results Section
                st.subheader("📋 Detailed Resume Analysis")
                
                # Sort by overall score (descending)
                df_sorted = df.sort_values('overall_score', ascending=False)
                
                for idx, (_, row) in enumerate(df_sorted.iterrows()):
                    # Find the corresponding detailed analysis
                    result_detail = next(r for r in results if r['filename'] == row['filename'])
                    analysis_details = result_detail['analysis_details']
                    
                    status_class = "selected" if row['status'] == 'Selected' else "rejected"
                    status_emoji = "✅" if row['status'] == 'Selected' else "❌"
                    
                    with st.expander(f"{status_emoji} {row['filename']} - {row['overall_score']}% ({row['status']})", expanded=False):
                        
                        # Overview section
                        st.markdown(f"""
                        <div class="analysis-section">
                        <h4>📊 Score Overview</h4>
                        <p><strong>Overall Score:</strong> {row['overall_score']}% 
                           <span style="color: {'green' if row['status'] == 'Selected' else 'red'}; font-weight: bold;">({row['status']})</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Score breakdown
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                f"Keywords Score", 
                                f"{row['keyword_score']}%",
                                help=f"Weight: {scoring_weights['keywords']:.1%}"
                            )
                        
                        with col2:
                            st.metric(
                                f"Similarity Score", 
                                f"{row['similarity_score']}%",
                                help=f"Weight: {scoring_weights['similarity']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                f"Education Score", 
                                f"{row['education_score']}%",
                                help=f"Weight: {scoring_weights['education']:.1%}"
                            )
                        
                        # Detailed keyword analysis
                        st.markdown("#### 🔍 Keyword Matching Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Keywords Found:**")
                            found_keywords = []
                            for keyword, details in analysis_details['keyword_details'].items():
                                if details['found'] and details['match_score'] > 75:
                                    found_keywords.append((keyword, details))
                            
                            if found_keywords:
                                for keyword, details in sorted(found_keywords, key=lambda x: x[1]['weight'], reverse=True):
                                    match_info = details['found']
                                    st.markdown(f"""
                                    <div class="keyword-found">
                                        ✅ <strong>{keyword}</strong> (weight: {details['weight']})  
                                        <br>Found: "{match_info['matched_text']}" ({match_info['score']}% match)
                                        <br>Frequency: {details['frequency']} mention{'s' if details['frequency'] != 1 else ''}
                                        {f"<br><span style='color: {'green' if details.get('frequency_multiplier', 1.0) == 1.0 else 'orange'};'>Frequency multiplier: {details.get('frequency_multiplier', 1.0):.0%}</span>" if details['weight'] >= 6 else ""}
                                        <br>Final contribution: {details['contributed_weight']:.1f} points
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.write("No keywords found with sufficient match score")
                        
                        with col2:
                            st.write("**Keywords Missing:**")
                            missing_keywords = []
                            for keyword, details in analysis_details['keyword_details'].items():
                                if not details['found'] or details['match_score'] <= 75:
                                    missing_keywords.append((keyword, details))
                            
                            if missing_keywords:
                                for keyword, details in sorted(missing_keywords, key=lambda x: x[1]['weight'], reverse=True)[:10]:
                                    st.markdown(f"""
                                    <div class="keyword-missing">
                                        ❌ <strong>{keyword}</strong> (weight: {details['weight']})
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Education analysis
                        if analysis_details['job_education'] or analysis_details['resume_education']:
                            st.markdown("#### 🎓 Education Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Job Requirements:**")
                                if analysis_details['job_education']:
                                    for edu in analysis_details['job_education']:
                                        st.write(f"• {edu}")
                                else:
                                    st.write("No specific education requirements found")
                            
                            with col2:
                                st.write("**Resume Education:**")
                                if analysis_details['resume_education']:
                                    for edu in analysis_details['resume_education']:
                                        st.write(f"• {edu}")
                                else:
                                    st.write("No education information found")
                        
                        # Analysis summary
                        st.markdown("#### 📈 Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Auto Keywords Used", analysis_details['auto_extracted_count'])
                        
                        with col2:
                            st.metric("Manual Keywords Used", analysis_details['manual_keywords_count'])
                        
                        with col3:
                            st.metric("Total Keywords Analyzed", analysis_details['total_keywords_used'])
                
                # Download enhanced results
                st.subheader("📥 Export Results")
                
                # Prepare detailed CSV data
                detailed_csv_data = []
                for result in results:
                    analysis = result['analysis_details']
                    found_keywords = [k for k, d in analysis['keyword_details'].items() 
                                    if d['found'] and d['match_score'] > 75]
                    missing_keywords = [k for k, d in analysis['keyword_details'].items() 
                                      if not d['found'] or d['match_score'] <= 75]
                    
                    detailed_csv_data.append({
                        'filename': result['filename'],
                        'overall_score': result['overall_score'],
                        'status': result['status'],
                        'keyword_score': result['keyword_score'],
                        'similarity_score': result['similarity_score'],
                        'education_score': result['education_score'],
                        'keywords_found': '; '.join(found_keywords),
                        'keywords_missing': '; '.join(missing_keywords[:10]),  # Limit to avoid very long text
                        'total_keywords_analyzed': analysis['total_keywords_used'],
                        'auto_extracted_count': analysis['auto_extracted_count'],
                        'manual_keywords_count': analysis['manual_keywords_count']
                    })
                
                detailed_df = pd.DataFrame(detailed_csv_data)
                csv = detailed_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Detailed Results as CSV",
                    data=csv,
                    file_name=f"comprehensive_resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.header("ℹ️ How It Works")
        st.markdown("""
        ### 🚀 **Enhanced Features:**
        
        **1. 🧠 Intelligent Keyword Extraction**
        - Automatically scans job description
        - Identifies technical skills, experience levels, education
        - Categorizes and weights keywords intelligently
        
        **2. ⚖️ Weighted Scoring System**
        - Manual keywords can boost important skills
        - Auto-extracted keywords ensure comprehensive coverage
        - Fuzzy matching handles variations and typos
        
        **3. 📊 Multi-Component Analysis**
        - **Keywords**: Direct skill matching with fuzzy logic
        - **Similarity**: Semantic understanding using TF-IDF
        - **Education**: Degree and certification matching
        
        **4. 🎯 Comprehensive Coverage**
        - Always analyzes the full job description
        - Combines auto-extraction with manual input
        - No important keywords are missed
        """)
        
        # Analysis breakdown section
        if job_description:
            st.subheader("🔍 Current Analysis Setup")
            
            # Show keyword analysis
            auto_keywords = analyzer.extract_advanced_keywords_from_job_description(job_description)
            comprehensive_keywords = analyzer.create_comprehensive_keyword_set(
                job_description, st.session_state.weighted_keywords
            )
            
            st.write(f"**Job Description Analysis:**")
            st.write(f"• Auto-extracted keywords: {len(auto_keywords)}")
            st.write(f"• Manual keywords: {len(st.session_state.weighted_keywords)}")
            st.write(f"• Total keywords for analysis: {len(comprehensive_keywords)}")
            
            # Show top keywords
            if auto_keywords:
                st.write("**Top Auto-Extracted Keywords:**")
                top_keywords = sorted(auto_keywords.items(), key=lambda x: x[1], reverse=True)[:8]
                keyword_text = ", ".join([f"{k} ({v})" for k, v in top_keywords])
                st.text(keyword_text)
            
            # Show education requirements
            education_reqs = analyzer.extract_education_requirements(job_description)
            if education_reqs:
                st.write("**Education Requirements Found:**")
                for edu in education_reqs[:5]:
                    st.write(f"• {edu}")
            
            # Current scoring weights
            st.subheader("⚖️ Scoring Configuration")
            st.write(f"Keywords: {scoring_weights['keywords']:.1%}")
            st.write(f"Similarity: {scoring_weights['similarity']:.1%}")
            st.write(f"Education: {scoring_weights['education']:.1%}")
            st.write(f"Selection Cutoff: {scoring_weights['cutoff']}%")

if __name__ == "__main__":
    main()