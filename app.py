import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Job Classification System",
    page_icon="💼",
    layout="wide"
)

# Title
st.title("💼 Job Classification System")
st.markdown("---")

# Initialize session state
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'clustered_df' not in st.session_state:
    st.session_state.clustered_df = None

class JobScraper:
    def _init_(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
    def scrape_jobs(self, search_terms, max_pages=2):
        """Scrape jobs from job websites"""
        all_jobs = []
        
        # Sample job data for demo (replace with actual scraping)
        sample_jobs = [
            {
                'title': 'Python Developer',
                'company': 'Tech Corp',
                'location': 'Bangalore',
                'skills': 'Python, Django, SQL, AWS',
                'experience': '2-4 years',
                'description': 'Develop web applications using Python and Django framework'
            },
            {
                'title': 'Data Scientist',
                'company': 'Analytics Inc',
                'location': 'Mumbai',
                'skills': 'Python, Machine Learning, Pandas, NumPy, TensorFlow',
                'experience': '3-5 years',
                'description': 'Build machine learning models and analyze data'
            },
            {
                'title': 'React Developer',
                'company': 'Frontend Solutions',
                'location': 'Delhi',
                'skills': 'React, JavaScript, HTML, CSS, Node.js',
                'experience': '1-3 years',
                'description': 'Create responsive web applications using React'
            },
            {
                'title': 'Java Developer',
                'company': 'Enterprise Systems',
                'location': 'Pune',
                'skills': 'Java, Spring Boot, MySQL, Microservices',
                'experience': '2-5 years',
                'description': 'Develop enterprise applications using Java'
            },
            {
                'title': 'DevOps Engineer',
                'company': 'Cloud Solutions',
                'location': 'Hyderabad',
                'skills': 'AWS, Docker, Kubernetes, Jenkins, Linux',
                'experience': '3-6 years',
                'description': 'Manage cloud infrastructure and deployment pipelines'
            },
            {
                'title': 'Full Stack Developer',
                'company': 'Startup Hub',
                'location': 'Bangalore',
                'skills': 'React, Node.js, MongoDB, Express, JavaScript',
                'experience': '2-4 years',
                'description': 'Build end-to-end web applications'
            },
            {
                'title': 'ML Engineer',
                'company': 'AI Labs',
                'location': 'Chennai',
                'skills': 'Python, TensorFlow, PyTorch, MLOps, Docker',
                'experience': '3-5 years',
                'description': 'Deploy and maintain machine learning models'
            },
            {
                'title': 'Backend Developer',
                'company': 'API Services',
                'location': 'Noida',
                'skills': 'Python, FastAPI, PostgreSQL, Redis, AWS',
                'experience': '2-4 years',
                'description': 'Build scalable backend services and APIs'
            }
        ]
        
        # Simulate scraping with progress bar
        progress_bar = st.progress(0)
        for i, job in enumerate(sample_jobs):
            progress_bar.progress((i + 1) / len(sample_jobs))
            job['job_id'] = f"job_{i+1}"
            job['scraped_at'] = datetime.now().isoformat()
            all_jobs.append(job)
            time.sleep(0.1)  # Simulate scraping delay
        
        return pd.DataFrame(all_jobs)

class SkillsProcessor:
    def _init_(self):
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'react', 'node.js'],
            'database': ['sql', 'mysql', 'mongodb', 'postgresql', 'redis'],
            'cloud': ['aws', 'azure', 'docker', 'kubernetes'],
            'data_science': ['machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
            'web': ['html', 'css', 'django', 'flask', 'express', 'fastapi']
        }
    
    def process_skills(self, df):
        """Process and normalize skills"""
        df['skills_lower'] = df['skills'].str.lower()
        df['skills_list'] = df['skills_lower'].str.split(',').apply(lambda x: [s.strip() for s in x])
        
        # Create skill category scores
        for category, skills in self.skill_categories.items():
            df[f'{category}_score'] = df['skills_list'].apply(
                lambda x: sum(1 for skill in skills if any(s in skill for s in x))
            )
        
        # Combined text for clustering
        df['combined_text'] = df['title'] + ' ' + df['skills'] + ' ' + df['description']
        
        return df

class JobClusterer:
    def _init_(self):
        self.vectorizer = None
        self.kmeans = None
        self.n_clusters = None
    
    def fit_model(self, df):
        """Train clustering model"""
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = self.vectorizer.fit_transform(df['combined_text'])
        
        # Determine optimal clusters
        silhouette_scores = []
        k_range = range(2, min(8, len(df)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        # Choose best k
        best_k = k_range[np.argmax(silhouette_scores)]
        self.n_clusters = best_k
        
        # Final model
        self.kmeans = KMeans(n_clusters=best_k, random_state=42)
        df['cluster'] = self.kmeans.fit_predict(X)
        
        return df
    
    def analyze_clusters(self, df):
        """Analyze cluster characteristics"""
        cluster_info = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_jobs = df[df['cluster'] == cluster_id]
            
            # Top skills
            all_skills = []
            for skills_list in cluster_jobs['skills_list']:
                all_skills.extend(skills_list)
            
            skill_counts = pd.Series(all_skills).value_counts().head(5)
            
            cluster_info[cluster_id] = {
                'size': len(cluster_jobs),
                'top_skills': skill_counts.to_dict(),
                'top_titles': cluster_jobs['title'].value_counts().head(3).to_dict(),
                'avg_categories': {
                    cat: cluster_jobs[f'{cat}_score'].mean() 
                    for cat in ['programming', 'database', 'cloud', 'data_science', 'web']
                }
            }
        
        return cluster_info

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    
    # Step 1: Scraping
    st.subheader("1. Data Collection")
    search_terms = st.text_input("Search Terms (comma-separated)", 
                                value="python developer, data scientist, react developer")
    max_pages = st.slider("Max Pages", 1, 5, 2)
    
    if st.button("🕷 Scrape Jobs", type="primary"):
        scraper = JobScraper()
        terms_list = [term.strip() for term in search_terms.split(',')]
        
        with st.spinner("Scraping jobs..."):
            st.session_state.jobs_df = scraper.scrape_jobs(terms_list, max_pages)
        
        st.success(f"Scraped {len(st.session_state.jobs_df)} jobs!")
    
    st.markdown("---")
    
    # Step 2: Clustering
    st.subheader("2. Job Classification")
    
    if st.button("🤖 Train Model & Classify", 
                 disabled=st.session_state.jobs_df is None):
        
        processor = SkillsProcessor()
        clusterer = JobClusterer()
        
        with st.spinner("Processing skills and training model..."):
            # Process skills
            processed_df = processor.process_skills(st.session_state.jobs_df.copy())
            
            # Train clustering model
            st.session_state.clustered_df = clusterer.fit_model(processed_df)
            
            # Store clusterer in session state
            st.session_state.clusterer = clusterer
            st.session_state.processor = processor
        
        st.success(f"Created {clusterer.n_clusters} job clusters!")

# Main content
if st.session_state.jobs_df is not None:
    
    # Display raw data
    st.subheader("📊 Scraped Jobs Data")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", len(st.session_state.jobs_df))
    with col2:
        st.metric("Companies", st.session_state.jobs_df['company'].nunique())
    with col3:
        st.metric("Locations", st.session_state.jobs_df['location'].nunique())
    with col4:
        if st.session_state.clustered_df is not None:
            st.metric("Clusters", st.session_state.clustered_df['cluster'].nunique())
    
    # Show sample data
    st.dataframe(st.session_state.jobs_df.head(), use_container_width=True)
    
    # Clustering results
    if st.session_state.clustered_df is not None:
        st.markdown("---")
        st.subheader("🎯 Job Classification Results")
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_counts = st.session_state.clustered_df['cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Jobs'},
                title="Jobs per Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=cluster_counts.values,
                names=[f'Cluster {i}' for i in cluster_counts.index],
                title="Cluster Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster analysis
        cluster_info = st.session_state.clusterer.analyze_clusters(st.session_state.clustered_df)
        
        st.subheader("🔍 Cluster Analysis")
        
        for cluster_id, info in cluster_info.items():
            with st.expander(f"Cluster {cluster_id} ({info['size']} jobs)"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("*Top Skills:*")
                    for skill, count in info['top_skills'].items():
                        st.write(f"- {skill}: {count}")
                
                with col2:
                    st.write("*Top Job Titles:*")
                    for title, count in info['top_titles'].items():
                        st.write(f"- {title}: {count}")
                
                st.write("*Category Scores:*")
                categories = list(info['avg_categories'].keys())
                scores = list(info['avg_categories'].values())
                
                fig = go.Figure(go.Bar(x=categories, y=scores))
                fig.update_layout(
                    title=f"Skill Categories for Cluster {cluster_id}",
                    xaxis_title="Categories",
                    yaxis_title="Average Score"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Jobs by cluster
        st.subheader("📋 Jobs by Cluster")
        selected_cluster = st.selectbox(
            "Select Cluster to View Jobs",
            options=sorted(st.session_state.clustered_df['cluster'].unique())
        )
        
        cluster_jobs = st.session_state.clustered_df[
            st.session_state.clustered_df['cluster'] == selected_cluster
        ][['title', 'company', 'location', 'skills', 'experience']]
        
        st.dataframe(cluster_jobs, use_container_width=True)

else:
    st.info("👈 Use the sidebar to start scraping jobs and building the classification model!")
    
    st.markdown("""
    ## How to use this app:
    
    1. *Scrape Jobs*: Enter search terms and click "Scrape Jobs" to collect job data
    2. *Train Model*: Click "Train Model & Classify" to create job clusters based on skills
    3. *Analyze Results*: Explore the clusters and their characteristics
    
    The system uses K-means clustering on job skills to automatically categorize jobs into similar groups.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤ using Streamlit • Job Classification with K-means Clustering")
