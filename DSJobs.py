import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import plotly.graph_objects as go
from wordcloud import WordCloud
import calmap
from plotly.subplots import make_subplots



# Set up the dashboard title and page config
st.set_page_config(page_title="DS Job Market Analysis", page_icon=":bar_chart:", layout="wide")


# Sidebar toggle for theme
mode = st.sidebar.radio("Select Mode", ["Light", "Dark"])

# Apply custom CSS based on mode
if mode == "Dark":
    dark_mode_style = """
        <style>
        .block-container {
            background-color: #2e2e2e;
            color: white;
        }
        .stButton > button {
            background-color: #444;
            color: white;
        }
        .custom-text {
            color: #f0f0f0;
        }
        </style>
        """
    st.markdown(dark_mode_style, unsafe_allow_html=True)
else:
    light_mode_style = """
        <style>
        .block-container {
            background-color: white;
            color: black;
        }
        .stButton > button {
            background-color: #e0e0e0;
            color: black;
        }
        .custom-text {
            color: #000000;
        }
        </style>
        """
    st.markdown(light_mode_style, unsafe_allow_html=True)





# Display updated DataFrame


# Main Title for the dashboard
st.title("📊 DS Job Market Analysis")

# Create three tabs: Key Insights, EDA, and Job Recommendation Tool
tab1, tab2, tab3, tab4 = st.tabs(["📈 Key Insights", "🔍 EDA", "💼 Job Recommendation",":two_men_holding_hands: About Us" ])

### TAB 1: Key Insights ###
with tab1:
    st.subheader("📈 Key Insights")
    st.write("""
    **Welcome to the Key Insights Section!**
    
    In this section, you will get an overview of the current Data Science job market.
    We will present key metrics such as job availability, demand for skills, and hiring trends.
    """)
    
    # Adding vertical space between the description and columns
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    # 1. Total Data Points 

    df = pd.read_csv('Job.csv') 
    # location issue solved
    df['location_1'] = df['location_1'].replace({'Helsinki Metropolitan Area': 'Helsinki', 'Uusimaa': 'Helsinki', 'Espoo':'Espoo' , 'Vantaa' : 'Vantaa' })





# Streamlit container for enhanced text-based visualization
    total_data_points = len(df)
    with col1:
        st.markdown(f"""
        <div style="background-color: #f0f4f8; padding: 20px; border-radius: 15px; 
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); text-align: center; 
                    width: 90%; margin: auto;">
            <div style="font-size: 40px; color: #007BFF; font-weight: bold;">
                {total_data_points:,}
            </div>
            <p style="font-size: 20px; font-weight: bold; color: #333;">
                Total Job Postings
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    
    
    
# 2. Month-on-Month Jobs Growth (MoM)


    # Calculate MoM jobs growth
    august_count = df[df['published_month'] == 8]['jobUrl'].nunique()
    september_count = df[df['published_month'] == 9]['jobUrl'].nunique()
    mom_jobs_growth = (september_count - august_count) * 100 / max(august_count, 1)

    # Choose colors and icons based on positive or negative growth
    if mom_jobs_growth >= 0:
        growth_color = "#28a745"  # Green for growth
        icon_svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="{growth_color}" viewBox="0 0 24 24">
            <path d="M12 2l-10 18h20z"/>
        </svg>
        """
    else:
        growth_color = "#dc3545"  # Red for decline
        icon_svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="{growth_color}" viewBox="0 0 24 24">
            <path d="M12 22l10-18h-20z"/>
        </svg>
        """

    with col2:
        st.markdown(f"""
        <div style="background-color: #f0f4f8; padding: 20px; border-radius: 15px; 
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); text-align: center; 
                    width: 90%; margin: auto;">
            <div style="font-size: 40px; font-weight: bold;">
                {icon_svg} {mom_jobs_growth:.2f}% 
            </div>
            <p style="font-size: 20px; font-weight: bold; color: #333;">September Job Growth</p>
            <p style="font-size: 16px; color: #6c757d;">(August vs September)</p>
        </div>
        """, unsafe_allow_html=True)

    
       
    st.markdown("<br><br>", unsafe_allow_html=True)   
    
    
    # Pic
    
    image_path = "KPI.png"  # Replace this with the correct path to your image
    image = Image.open(image_path)

    # Display image in the second column
    with col3:
        st.image(image, width=250)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    
    
    
    
    
    
    
    
     
    
    col1, col2 = st.columns([1, 1])    
    
        
        
        
        
        
    # Average  job posts count per week#   

            

    # Convert the 'publishedAt' column to datetime format using the correct format
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], format='%Y-%m-%d')

    # Calculate average job posts per week
    weekly_avg = df.resample('W', on='publishedAt').size().reset_index(name='Count')
    weekly_avg['Week'] = weekly_avg['publishedAt'].dt.strftime('Week %U - %Y')

    # Create a line chart to visualize the weekly averages
    fig = go.Figure()

    # Add weekly average line
    fig.add_trace(go.Scatter(x=weekly_avg['Week'], y=weekly_avg['Count'],
                            mode='lines+markers', name='Weekly Average',
                            line=dict(color='orange', width=3, dash='dot'),
                            marker=dict(size=6)))

    # Customize layout for better visual impact
    fig.update_layout(
        title='Job Posts Count per Week',
        xaxis_title='Week',
        yaxis_title='Number of Job Postings',
        legend_title='Average Type',
        xaxis_tickangle=-45,
        xaxis=dict(type='category'),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12)
    )

    # Display the chart in Streamlit

    st.plotly_chart(fig, use_container_width=True)
        
        
        
        
        
        
        
        
        
        


        
        # 3. Top 5 companies
        
        
    top_5_companies = df['companyName'].value_counts().nlargest(5)
    fig2 = go.Figure(go.Pie(labels=top_5_companies.index, values=top_5_companies.values, 
                                hoverinfo="label+percent", textinfo="label+value", marker=dict(colors=px.colors.qualitative.Bold)))
    fig2.update_layout(title_text="Top 5 Companies by Job Postings", height=300, 
                        margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

        
        
        
    # 4. Top 5 Job Titles
        
    top_5_titles_df = df['title'].value_counts().nlargest(5).reset_index()
    top_5_titles_df.columns = ['title', 'count']
    fig4 = px.treemap(top_5_titles_df, path=['title'], values='count', 
                        title="Top 5 Job Titles by Job Postings")
    fig4.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)', 
                        paper_bgcolor='rgba(0, 0, 0, 0)')
    with col1:
        st.plotly_chart(fig4, use_container_width=True)
        
 
 
    # Display the chart in Streamlit
    

    # Define the categorize_sector function
    def categorize_sector(sector):
        # Check if the sector is NaN
        if pd.isna(sector):
            return "Unknown"

        # Defining the five main categories
        it_consulting = [
            "Software Development", "IT Services and IT Consulting", "Technology, Information and Internet",
            "Information Technology & Services", "IT System Custom Software Development", "Telecommunications",
            "Computer Games", "Data Infrastructure and Analytics", "Cloud Computing", "IT System Testing and Evaluation"
        ]
        
        healthcare_pharma = [
            "Pharmaceutical Manufacturing", "Hospitals and Health Care", "Medical Equipment Manufacturing", 
            "Medical Device"
        ]
        
        manufacturing_engineering = [
            "Industrial Machinery Manufacturing", "Engineering Services", "Mechanical Or Industrial Engineering", 
            "Computer Hardware Manufacturing", "Semiconductor Manufacturing", "Oil and Gas", 
            "Maritime Transportation", "Appliances, Electrical, and Electronics Manufacturing", "Utility System Construction"
        ]
        
        finance_banking = [
            "Banking", "Financial Services", "Venture Capital and Private Equity Principals", "Insurance"
        ]
        
        education_misc = [
            "Higher Education", "Non-profit Organizations and Primary and Secondary Education", "Research Services", 
            "Advertising Services", "Entertainment Providers", "Construction", "Utilities", "Staffing and Recruiting",
            "Business Consulting and Services", "Retail", "Government Administration"
        ]

        if any(term in sector for term in it_consulting):
            return "Information Technology & Consulting"
        elif any(term in sector for term in healthcare_pharma):
            return "Healthcare & Pharmaceuticals"
        elif any(term in sector for term in manufacturing_engineering):
            return "Manufacturing & Engineering"
        elif any(term in sector for term in finance_banking):
            return "Financial Services & Banking"
        else:
            return "Education & Miscellaneous"

    # Apply the categorize_sector function to create a new column
    df['main_sector'] = df['sector'].apply(categorize_sector)

    # Group by main sector and count the job posts per sector
    job_posts_per_sector = df['main_sector'].value_counts()

    # Plotting the results using matplotlib and displaying in Streamlit

    fig, ax = plt.subplots(figsize=(10, 6))
    job_posts_per_sector.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Job Posts Count per Main Sector', fontsize=16)
    ax.set_xlabel('Main Sector', fontsize=14)
    ax.set_ylabel('Job Posts Count', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    # Displaying the plot on Streamlit
    st.pyplot(fig)
    
    
   
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 ##### Filteration I for number of job postings 
 
 
     
 
    def categorize_sector(sector):
        if pd.isna(sector):
            return "Unknown"

        it_consulting = [
            "Software Development", "IT Services and IT Consulting", "Technology, Information and Internet",
            "Information Technology & Services", "IT System Custom Software Development", "Telecommunications",
            "Computer Games", "Data Infrastructure and Analytics", "Cloud Computing", "IT System Testing and Evaluation"
        ]

        healthcare_pharma = [
            "Pharmaceutical Manufacturing", "Hospitals and Health Care", "Medical Equipment Manufacturing", 
            "Medical Device"
        ]

        manufacturing_engineering = [
            "Industrial Machinery Manufacturing", "Engineering Services", "Mechanical Or Industrial Engineering", 
            "Computer Hardware Manufacturing", "Semiconductor Manufacturing", "Oil and Gas", 
            "Maritime Transportation", "Appliances, Electrical, and Electronics Manufacturing", "Utility System Construction"
        ]

        finance_banking = [
            "Banking", "Financial Services", "Venture Capital and Private Equity Principals", "Insurance"
        ]

        education_misc = [
            "Higher Education", "Non-profit Organizations and Primary and Secondary Education", "Research Services", 
            "Advertising Services", "Entertainment Providers", "Construction", "Utilities", "Staffing and Recruiting",
            "Business Consulting and Services", "Retail", "Government Administration"
        ]

        if any(term in sector for term in it_consulting):
            return "Information Technology & Consulting"
        elif any(term in sector for term in healthcare_pharma):
            return "Healthcare & Pharmaceuticals"
        elif any(term in sector for term in manufacturing_engineering):
            return "Manufacturing & Engineering"
        elif any(term in sector for term in finance_banking):
            return "Financial Services & Banking"
        else:
            return "Education & Miscellaneous"


    def categorize_job_category(job_category):
        if pd.isna(job_category) or job_category == []:
            return "Unknown"

        engineer_related = ["Engineer", "Architect", "Developer", "Scientist"]
        analyst_related = ["Analyst", "Consultant", "Data Steward", "Product Owner"]
        manager_related = ["Manager", "Internship", "Specialist", "Researcher"]

        if any(term in job_category for term in engineer_related):
            return "Engineering & Development"
        elif any(term in job_category for term in analyst_related):
            return "Analysis & Consulting"
        elif any(term in job_category for term in manager_related):
            return "Management & Research"
        else:
            return "Other"


    # Handle 'publishedAt' column to convert to datetime
    df['published_date'] = pd.to_datetime(df['publishedAt'], dayfirst=True, errors='coerce')
    df['published_month'] = df['published_date'].dt.month

    # Apply the categorization functions
    df['main_sector'] = df['sector'].apply(categorize_sector)
    df['main_job_category'] = df['general_job_category'].apply(categorize_job_category)

    # Filter selection
    sectors = df['main_sector'].dropna().unique().tolist()
    selected_sector = st.selectbox("Select a Sector", options=["All"] + sectors)

    job_categories = df['main_job_category'].dropna().unique().tolist()
    selected_job_category = st.selectbox("Select a General Job Category", options=["All"] + job_categories)

    # Apply the selected filters
    filtered_df = df.copy()

    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df['main_sector'] == selected_sector]

    if selected_job_category != "All":
        filtered_df = filtered_df[filtered_df['main_job_category'] == selected_job_category]

    # Handling the case when no data is available after filtering
    if filtered_df.empty:
        st.write("No data available for the selected filters. Please try different options.")
    else:
        # Calculate job posts count per month and week for the filtered data
        job_posts_month = filtered_df.groupby(['published_month'])['jobUrl'].nunique().reset_index()
        job_posts_week = filtered_df.groupby(['published_date'])['jobUrl'].nunique().reset_index()

        # Create line chart for monthly data
        fig_monthly = px.line(job_posts_month, 
                            x='published_month', 
                            y='jobUrl', 
                            title=f'Job Posts Count per Month (Filtered for Sector: {selected_sector} and Job Category: {selected_job_category})',
                            labels={'published_month': 'Month', 'jobUrl': 'Job Posts Count'},
                            markers=True)

        # Create line chart for weekly data
        fig_weekly = px.line(job_posts_week, 
                            x='published_date', 
                            y='jobUrl', 
                            title=f'Job Posts Count per Week (Filtered for Sector: {selected_sector} and Job Category: {selected_job_category})',
                            labels={'published_date': 'Week', 'jobUrl': 'Job Posts Count'},
                            markers=True)

        # Displaying the figures
    
        st.subheader("Job posting counts per month & week by Sector and General Job Category.")
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.plotly_chart(fig_weekly, use_container_width=True)


    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
    # 5. Uusimaa Subdivisions
    image_path = "uu1.png"  # Replace this with the correct path to your image
    image = Image.open(image_path)

    # Display image in the second column
    with col1:
        st.image(image, caption="Uusimaa Job Postings by Subdivision", width=800)

    st.markdown("<br><br>", unsafe_allow_html=True)



        
    # Applications Count vs Job Displayed Time for Top 5 Job Titles 

    df['job_displayedTime'] = df['job_displayedTime'].str.replace('w', '', regex=False)  # Remove 'w'
    df['job_displayedTime'] = df['job_displayedTime'].str.replace('+', '', regex=False)  # Remove '+'

    df['job_displayedTime_numeric'] = pd.to_numeric(df['job_displayedTime'], errors='coerce')

    df['applicationsCount'] = pd.to_numeric(df['applicationsCount'], errors='coerce')

    top_5_titles = df.groupby('title')['applicationsCount'].sum().nlargest(5).index

    filtered_df = df[df['title'].isin(top_5_titles)]

    fig = px.bar(filtered_df, 
                x='job_displayedTime_numeric', 
                y='applicationsCount', 
                color='title',  # Add color for different job titles and use it as a legend
                labels={'job_displayedTime_numeric': 'Job Displayed Time (Weeks)', 'applicationsCount': 'Applications Count', 'title': 'Job Title'},
                title="Applications Count vs Job Displayed Time for Top 5 Job Titles")

    # Adjust layout, add title, and display the chart
    fig.update_layout(
        margin=dict(t=60, b=60), 
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        paper_bgcolor='rgba(0, 0, 0, 0)', 
        showlegend=True  # Show the legend
    )


   
    ######## Filterations 2 - skills based on company, job and location ############ 
    

    def flatten_and_count(column_data):
        all_values = column_data.dropna().str.split(',').apply(lambda x: [item.strip() for item in x]).explode()
        value_counts = all_values.value_counts()
        return value_counts

    # Get the top 5 companies based on job postings

    st.subheader("Distribution of job requiremets based on company, job title and location ")   
    
    all_companies = df['companyName'].dropna().unique().tolist()
    top_5_companies = df['companyName'].value_counts().nlargest(5).index.tolist()
    selected_company = st.selectbox("Select Company (Optional)", options=["All"] + top_5_companies)

    # Filter job titles based on the selected company
    if selected_company != "All":
        company_specific_titles = df[df['companyName'] == selected_company]['title'].dropna().unique().tolist()
    else:
        company_specific_titles = df['title'].dropna().unique().tolist()

    selected_title = st.selectbox("Select Job Title (Optional)", options=["All"] + company_specific_titles)

    # Filter locations
    unique_locations = df['location_1'].dropna().unique().tolist()
    selected_location = st.selectbox("Select Location (Optional)", options=["All"] + unique_locations)

    # Apply filters to the DataFrame
    filtered_df = df.copy()

    if selected_company != "All":
        filtered_df = filtered_df[filtered_df['companyName'] == selected_company]

    if selected_title != "All":
        filtered_df = filtered_df[filtered_df['title'] == selected_title]

    if selected_location != "All":
        filtered_df = filtered_df[filtered_df['location_1'] == selected_location]

    if filtered_df.empty:
        st.write("No data available for the selected filters. Please try different options.")
    else:
        st.write(f"### Data for {selected_company if selected_company != 'All' else 'All Companies'} - {selected_title if selected_title != 'All' else 'All Titles'} in {selected_location if selected_location != 'All' else 'All Locations'}")
        
        col1, col2 = st.columns(2)


        
        # 1. Visualize Soft Skills
        with col1:
            if 'Soft Skills' in filtered_df.columns:
                value_counts_soft = flatten_and_count(filtered_df['Soft Skills'])
                if not value_counts_soft.empty:
                    fig_soft_skills = px.bar(value_counts_soft, x=value_counts_soft.values, y=value_counts_soft.index, 
                                            title="Top Soft Skills", 
                                            orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                    st.plotly_chart(fig_soft_skills, use_container_width=True)
                else:
                    st.write("No Soft Skills data available for the selected filters.")
            else:
                st.write("Soft Skills column not found in the dataset.")
                

        # 2. Visualize Experience Level
        with col2:
            if 'experienceLevel' in filtered_df.columns:
                if filtered_df['experienceLevel'].notna().sum() > 0:
                    fig_experience = px.pie(filtered_df, names='experienceLevel', 
                                            title="Experience Level Distribution",
                                            labels={'experienceLevel': 'Experience Level'})
                    st.plotly_chart(fig_experience, use_container_width=True)
                else:
                    st.write("No Experience Level data available for the selected filters.")
            else:
                st.write("Experience Level column not found in the dataset.")
                

        # 3. Visualize Data Base Applications
        with col1:
            if 'Data Base Applications' in filtered_df.columns:
                value_counts_db_apps = flatten_and_count(filtered_df['Data Base Applications'])
                if not value_counts_db_apps.empty:
                    fig_db_apps = px.bar(value_counts_db_apps, x=value_counts_db_apps.values, y=value_counts_db_apps.index, 
                                        title="Top Data Base Applications", 
                                        orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                    st.plotly_chart(fig_db_apps, use_container_width=True)
                else:
                    st.write("No Data Base Applications data available for the selected filters.")
            else:
                st.write("Data Base Applications column not found in the dataset.")
                

        ### Visualize Data Engineering and Cloud Computing ###
        
        # 4. Visualize Data Engineering
        with col2:
            if 'Data Engineering' in filtered_df.columns:
                value_counts_data_eng = flatten_and_count(filtered_df['Data Engineering'])
                if not value_counts_data_eng.empty:
                    fig_data_eng = px.bar(value_counts_data_eng, x=value_counts_data_eng.values, y=value_counts_data_eng.index, 
                                        title="Top Data Engineering Skills", 
                                        orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                    st.plotly_chart(fig_data_eng, use_container_width=True)
                else:
                    st.write("No Data Engineering data available for the selected filters.")
            else:
                st.write("Data Engineering column not found in the dataset.")

        # 5. Visualize Cloud Computing
        
        with col1:
            if 'Cloud Computing' in filtered_df.columns:
                value_counts_cloud = flatten_and_count(filtered_df['Cloud Computing'])
                if not value_counts_cloud.empty:
                    fig_cloud = px.bar(value_counts_cloud, x=value_counts_cloud.values, y=value_counts_cloud.index, 
                                    title="Top Cloud Computing Skills", 
                                    orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                    st.plotly_chart(fig_cloud, use_container_width=True)
                else:
                    st.write("No Cloud Computing data available for the selected filters.")
            else:
                st.write("Cloud Computing column not found in the dataset.")
        
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
      

### TAB 2: EDA (Exploratory Data Analysis) ###



with tab2:
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.write("""
    **Welcome to the EDA Section!**
    
    In this section, we will visualize the key variables from the dataset to help understand the job market dynamics.
    """)

    # Dataset Overview and Variable Overview are already displayed
    st.write("### Dataset Overview")
    st.write(df.head())
   
    
    

    # Variable Overview table (types, unique values, and missing values)
    column_descriptions = {
    'applicationsCount': 'Total number of applications received',
    'applyType': 'Type of application method used',
    'applyUrl': 'URL to apply for the job',
    'companyId': 'Unique identifier for the company',
    'companyName': 'Name of the company',
    'companyUrl': 'URL of the company website',
    'contractType': 'Type of contract offered',
    'description': 'Description of the job role',
    'experienceLevel': 'Required experience level for the job',
    'jobUrl': 'URL of the job posting',
    'location': 'Location of the job',
    'postedTime': 'Time when the job was posted',
    'posterFullName': 'Full name of the person who posted the job',
    'posterProfileUrl': 'URL to the profile of the job poster',
    'publishedAt': 'Date when the job was published',
    'salary': 'Salary offered for the job',
    'sector': 'Sector or industry of the job',
    'title': 'Job title',
    'workType': 'Type of work',
    'work_arrangement': 'Work arrangement',
    'benefits': 'Benefits offered with the job',
    'id': 'Unique identifier for the job posting',
    'Language Posted': 'Language in which the job was posted',
    'required skills': 'List of skills required for the job',
    'Data Base Applications': 'Database skills required for the job',
    'Data Engineering': 'Data engineering skills required for the job',
    'Cloud Computing': 'Cloud computing skills required for the job',
    'Soft Skills': 'Soft skills required for the job',
    'applicationsCount_interval': 'Interval range for the number of applications received',
    'company_level': 'Level or scale of the company ',
    'location_1': 'Primary location information for the job',
    'location_2': 'Secondary location information for the job',
    'displayed_weeks': 'Number of weeks the job was displayed',
    'job_displayedTime': 'Time for which the job was displayed (e.g., in weeks)',
    'published_month': 'Month when the job was published',
    'general_job_category': 'General category or classification of the job',
    'job_title_desc': 'Description of the job title',
    'job_title_desc_list': 'List of descriptions associated with the job title',
    'worktype_desc': 'Detailed description of the type of work',
    'job_displayedTime_numeric': 'Numeric representation of the displayed time for the job (e.g., in weeks)'
}

    # Create the DataFrame for the EDA
    eda_info = pd.DataFrame({
        "Description": [column_descriptions.get(col, "No description available") for col in df.columns],
        "Data Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Missing Values": df.isnull().sum()
    })

    # Display the Variable Overview
    st.write("### Variable Overview")
    st.dataframe(eda_info)

    
    
    
    
    
    st.write("## Univariate Visualization")
    
    col1, col2 = st.columns(2)
       

    # 1. Applications Count (Numerical)
    with col1:
        st.write("#### Applications Count")
        fig_applications = px.histogram(df, x='applicationsCount', nbins=20, 
                                        labels={'applicationsCount': 'Applications Count', 'y': 'Frequency'})
        fig_applications.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig_applications, use_container_width=True)
        
        
        

    # 2. Apply Type (Categorical)
    with col2:
        applytype_map2 = {'EXTERNAL': 'External', 'EASY_APPLY': 'Easy Apply'}

    
        df['applyType'] = df['applyType'].map(applytype_map2)
        
        st.write("#### Apply Type")
        fig_apply_type = px.bar(df['applyType'].value_counts(), x=df['applyType'].value_counts().index, 
                                y=df['applyType'].value_counts().values,
                                labels={'x': 'Apply Type', 'y': 'Number of Applications'})
        fig_apply_type.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig_apply_type, use_container_width=True)
        
        
        
        

    # 3. Company Name (Categorical - Top 10)
    with col1:
        df = df.rename(columns={'companyName': 'Company Name'})
        st.write("#### Top 10 Companies by Job Postings")
        
        top_companies = df['Company Name'].value_counts().nlargest(10)
        fig_company = px.bar(top_companies, x=top_companies.index, y=top_companies.values,
                             labels={'x': 'Company Name', 'y': 'Number of Job Postings'})
        fig_company.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig_company, use_container_width=True)
        
        
        

    # 4. Contract Type (Categorical)
    with col2:
        
        st.write("#### Contract Type")
        fig_contract = px.pie(df, names='contractType',
                            labels={'contractType': 'Contract Type'})

        
        fig_contract.update_traces(textinfo='percent+label',
                                textfont=dict(color='white'))  # Set the font color of the labels to black

        
        fig_contract.update_layout(margin=dict(t=30, b=10), 
                                plot_bgcolor='rgba(0, 0, 0, 0)', 
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                font=dict(color='white'),  # Change the font color in the layout (including legend)
                                legend=dict(font=dict(color='white')))  # Set the legend font color to black

        st.plotly_chart(fig_contract, use_container_width=True)
        
        
        

    # 5. Experience Level (Categorical)
    
    with col1:
        st.write("#### Experience Level")
        fig_experience = px.bar(df['experienceLevel'].value_counts(), x=df['experienceLevel'].value_counts().index, 
                                y=df['experienceLevel'].value_counts().values,
                                labels={'x': 'Experience Level', 'y': 'Number of Applications'})
        fig_experience.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig_experience, use_container_width=True)
        
        
        

    # 6. Sector (Categorical)
    
    with col2:
        company_level_counts = df['company_level'].value_counts()

        
        fig_donut = go.Figure(go.Pie(
            labels=company_level_counts.index, 
            values=company_level_counts.values, 
            hole=0.4,  
            marker=dict(colors=px.colors.qualitative.Dark2),  
            hoverinfo="label+percent",  
            textinfo="value+label",  
            textfont=dict(color='white')  
        ))

        
        fig_donut.update_layout(
            annotations=[dict(text='Company Levels', x=0.5, y=0.5, font_size=20, font=dict(color='black'), showarrow=False)],  # Text inside the hole
            margin=dict(t=30, b=10),
            showlegend=True,
            font=dict(color='black'),  
            legend=dict(font=dict(color='black')),  
            uniformtext_minsize=12,  
            uniformtext_mode='hide',  
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'   
        )

        st.write("#### Company Level Distribution")
        st.plotly_chart(fig_donut, use_container_width=True)

            
        
        

    # 7. Title (Categorical)
    with col1:
        st.write("#### Job Title")
        fig_title = px.bar(df['title'].value_counts().nlargest(10), x=df['title'].value_counts().nlargest(10).index, 
                           y=df['title'].value_counts().nlargest(10).values,
                           labels={'x': 'Job Title', 'y': 'Number of Jobs'})
        fig_title.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig_title, use_container_width=True)

    
    # 8. Work Type (Treemap)
            
    all_work_types = df['workType'].dropna().str.replace('and', '').str.split('[, ]')  
    flat_work_types = [item.strip() for sublist in all_work_types for item in sublist if item.strip()]  
   
    word_freq_df = pd.DataFrame(flat_work_types, columns=['Work Type'])
    word_freq_df['Count'] = word_freq_df.groupby('Work Type')['Work Type'].transform('count')
    word_freq_df = word_freq_df.drop_duplicates().reset_index(drop=True)
  
    fig_treemap = px.treemap(word_freq_df, path=['Work Type'], values='Count', 
                            color='Count', color_continuous_scale='RdBu')
   
    st.write("#### Work Type Treemap")
    st.plotly_chart(fig_treemap, use_container_width=True)


    # 9. Work Arrangement (Categorical)
    
    with col1:
        
        work_arrangement_map = {'On site': 'On-site','On-site': 'On-site','Hybrid': 'Hybrid','Remote': 'Remote'}
        
        
        df['work_arrangement'] = df['work_arrangement'].map(work_arrangement_map)
        
        st.write("#### Work Arrangement")
        fig_workarrangement = px.bar(df['work_arrangement'].value_counts(), x=df['work_arrangement'].value_counts().index, 
                                     y=df['work_arrangement'].value_counts().values,
                                     labels={'x': 'Work Arrangement', 'y': 'Number of Jobs'})
        fig_workarrangement.update_layout(margin=dict(t=30, b=10), plot_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig_workarrangement, use_container_width=True)
        
    
    
    
   
    #10. Soft skills
    
    soft_skills_map = {'strategic leadership': 'Strategic Leadership', 'organizational skills': 'Organizational Skills','interpersonal communication': 'Interpersonal Communication','time management': 'Time Management','adaptability': 'Adaptability','analytical skills': 'Analytical Skills','empathy': 'Empathy','written communication': 'Written Communication','attention to detail': 'Attention To Detail','collaboration skills': '','leadership': 'Leadership','flexibility': 'Flexibility','communication': 'Communication',}
        
        
    df['Soft Skills'] = df['Soft Skills'].map(soft_skills_map)
    
    with col2:
            def flatten_and_count(column_data):
                all_values = column_data.dropna().str.split(',').apply(lambda x: [item.strip() for item in x]).explode()
                all_values = all_values[all_values.str.lower() != 'unknown']
                value_counts = all_values.value_counts()
                return value_counts
            
        
            st.write("### Soft Skill Requirements")
            if 'Soft Skills' in df.columns:
                value_counts_soft = flatten_and_count(df['Soft Skills'])
                if not value_counts_soft.empty:
                    fig_soft_skills = px.bar(value_counts_soft, x=value_counts_soft.values, y=value_counts_soft.index, 
                                            orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                    fig_soft_skills.update_layout(margin=dict(t=40, b=40), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                    st.plotly_chart(fig_soft_skills, use_container_width=True)
                else:
                    st.write("No data available for Soft Skills.")
            else:
                st.write("Column 'Soft Skills' not found in the dataset.")

        
   
    # 11. Language Posted (Categorical)
    
    
    with col2:
        st.write("#### Language Posted")
        df.rename(columns={'language_posted': 'Language Posted'}, inplace=True)

        fig_language = px.pie(df, names='Language Posted', 
                            labels={'Language Posted': 'Language'},
                            color_discrete_sequence=px.colors.qualitative.Dark2)  

        
        fig_language.update_traces(
            textinfo='percent+label',
            textfont=dict(color='black')  
        )

        
        fig_language.update_layout(
            margin=dict(t=30, b=10),
            plot_bgcolor='rgba(0, 0, 0, 0)',  
            paper_bgcolor='rgba(0, 0, 0, 0)',  
            font=dict(color='black'),  
            legend=dict(font=dict(color='black')),  
        )

        
        st.plotly_chart(fig_language, use_container_width=True)


    
   
    # 12. Required Skills (Word Cloud)
    
    all_skills = df['required skills'].dropna().str.split(',')  
    flat_skills = [item.strip() for sublist in all_skills for item in sublist if item.strip()]    
    word_freq = pd.Series(flat_skills).value_counts().to_dict()
   
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)
    
    st.write("#### Basic Skill Requirements")
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    st.pyplot(plt)
    
    
       
    # Helper function to flatten, count occurrences, and remove unwanted values like "unknown"
    def flatten_and_count(column_data):
        all_values = column_data.dropna().str.split(',').apply(lambda x: [item.strip() for item in x]).explode()
        all_values = all_values[all_values.str.lower() != 'unknown']
        value_counts = all_values.value_counts()
        return value_counts

    with col1:
        st.write("### Data Base Application Requirements")
        if 'Data Base Applications' in df.columns:
            value_counts_db_apps = flatten_and_count(df['Data Base Applications'])
            if not value_counts_db_apps.empty:
                fig_db_apps = px.bar(value_counts_db_apps, x=value_counts_db_apps.values, y=value_counts_db_apps.index, 
                                    title="Top Data Base Applications", 
                                    orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                fig_db_apps.update_layout(margin=dict(t=40, b=40), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                st.plotly_chart(fig_db_apps, use_container_width=True)
            else:
                st.write("No data available for Data Base Applications.")
        else:
            st.write("Column 'Data Base Applications' not found in the dataset.")
            
            
            

    ### 13. Data Engineering Visualization
    with col2:
        st.write("### Data Engineering Requirements")
        if 'Data Engineering' in df.columns:
            value_counts_data_eng = flatten_and_count(df['Data Engineering'])
            if not value_counts_data_eng.empty:
                fig_data_eng = px.bar(value_counts_data_eng, x=value_counts_data_eng.values, y=value_counts_data_eng.index,  
                                    orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                fig_data_eng.update_layout(margin=dict(t=40, b=40), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                st.plotly_chart(fig_data_eng, use_container_width=True)
            else:
                st.write("No data available for Data Engineering.")
        else:
            st.write("Column 'Data Engineering' not found in the dataset.")
            
            
            
    ### 14. Cloud Computing Visualization 
    with col1:
        st.write("### Cloud Computing Requirements")
        if 'Cloud Computing' in df.columns:
            value_counts_cloud = flatten_and_count(df['Cloud Computing'])
            if not value_counts_cloud.empty:
                fig_cloud = px.bar(value_counts_cloud, x=value_counts_cloud.values, y=value_counts_cloud.index, 
                                orientation='h', labels={'y': 'Skill', 'x': 'Count'}, height=400)
                fig_cloud.update_layout(margin=dict(t=40, b=40), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                st.plotly_chart(fig_cloud, use_container_width=True)
            else:
                st.write("No data available for Cloud Computing.")
        else:
            st.write("Column 'Cloud Computing' not found in the dataset.")


    
    st.write("## Bivariate Visualization")
    
    col1, col2 = st.columns(2)
    
    
    
    
    # 1. Applications Count vs. Experience Level (Box Plot)
    with col1:
        st.write("### Applications Count vs. Experience Level")
        fig1 = px.box(df, x='experienceLevel', y='applicationsCount',  
                    labels={'experienceLevel': 'Experience Level', 'applicationsCount': 'Applications Count'})
        st.plotly_chart(fig1, use_container_width=True)




    # 2. Contract Type vs. Work Arrangement (Stacked Bar Chart)
    with col2:
        
           
        st.write("### Contract Type vs. Work Arrangement")
        fig2 = px.histogram(df, x='work_arrangement', color='contractType', 
                            labels={'work_arrangement': 'Work Arrangement', 'contractType': 'Contract Type'}, 
                            barmode='stack')
        st.plotly_chart(fig2, use_container_width=True)
        
        
        
    # 3. Company Level vs. Work Arrangement (Sorted Grouped Bar Chart)
    with col1:
        st.write("### Company Level vs. Work Arrangement")
        
        
        sorted_df = df.sort_values(by=['company_level', 'work_arrangement'])

        fig3 = px.bar(sorted_df, x='company_level', color='work_arrangement', 
                    title="Company Level by Work Arrangement", 
                    labels={'company_level': 'Company Level', 'work_arrangement': 'Work Arrangement'},
                    barmode='group')
        
        fig3.update_layout(margin=dict(t=30, b=10), xaxis={'categoryorder': 'total ascending'})

        st.plotly_chart(fig3, use_container_width=True)


        
        
    # 4. Experience Level vs. Title (Stacked Bar Chart)
    with col2:
        st.write("### Experience Level vs. Job Title")
        fig5 = px.histogram(df, x='title', color='experienceLevel', 
                            labels={'title': 'Job Title', 'experienceLevel': 'Experience Level'},
                            barmode='stack')
        st.plotly_chart(fig5, use_container_width=True)


   
  
    # 5. Language Posted vs. Applications Count (Box Plot)
    with col1:
        st.write("### Language Posted vs. Applications Count")
        fig7 = px.box(df, x='Language Posted', y='applicationsCount', 
                    labels={'Language Posted': 'Language', 'applicationsCount': 'Applications Count'})
        st.plotly_chart(fig7, use_container_width=True)



    # 6. Company Level vs. Applications Count (Violin Plot)
    with col2:
        st.write("### Company Level vs. Applications Count")
        fig8 = px.violin(df, x='company_level', y='applicationsCount', 
                        labels={'company_level': 'Company Level', 'applicationsCount': 'Applications Count'})
        st.plotly_chart(fig8, use_container_width=True)



    # 7. Contract Type vs. Applications Count (Violin Plot)
    with col1:
        st.write("### Contract Type vs. Applications Count")
        fig9 = px.violin(df, x='contractType', y='applicationsCount', 
                        labels={'contractType': 'Contract Type', 'applicationsCount': 'Applications Count'})
        st.plotly_chart(fig9, use_container_width=True)
        
        
        
    # 8. Experience Level vs. Contract Type (Grouped Bar Chart)
    with col2:
        st.write("### Experience Level vs. Contract Type")
        fig12 = px.bar(df, x='contractType', color='experienceLevel', 
                    labels={'contractType': 'Contract Type', 'experienceLevel': 'Experience Level'},
                    barmode='group')
        st.plotly_chart(fig12, use_container_width=True)    
        
        

    # 9. Data Base Applications vs. Data Engineering (Grouped Bar Chart)
    with col1:

        def create_wordcloud(column_data):
            all_values = column_data.dropna().astype(str).str.split(',').apply(lambda x: [item.strip() for item in x if item.strip().lower() != 'unknown']).explode()
            
            all_text = ' '.join(all_values.dropna())  # Drop any remaining NaNs just in case
            
            wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='viridis').generate(all_text)
            return wordcloud

        with col1:
            st.write("### Data Base Applications Requirements")
            if 'Data Base Applications' in df.columns:
                wordcloud_db_apps = create_wordcloud(df['Data Base Applications'])
                # Display the word cloud
                plt.figure(figsize=(6, 4))
                plt.imshow(wordcloud_db_apps, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

        with col2:
            st.write("### Data Engineering Requirements")
            if 'Data Engineering' in df.columns:
                wordcloud_data_eng = create_wordcloud(df['Data Engineering'])
                # Display the word cloud
                plt.figure(figsize=(6, 4))
                plt.imshow(wordcloud_data_eng, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)


           

    # 10. Experience Level vs. Soft Skills
    
    with col2:
        st.write("### Experience Level vs. Soft Skills")
        
        # Helper function to flatten and count occurrences
        def flatten_and_count_by_experience(df, experience_col, skill_col):
            # Flatten the comma-separated values for soft skills, remove leading/trailing spaces
            skill_data = df[[experience_col, skill_col]].dropna()
            skill_data[skill_col] = skill_data[skill_col].astype(str).str.split(',').apply(lambda x: [item.strip() for item in x if item.strip().lower() != 'unknown'])
            
            exploded_data = skill_data.explode(skill_col)
            
            skill_counts = exploded_data.groupby([experience_col, skill_col]).size().reset_index(name='Count')
            return skill_counts

        skill_counts_exp_soft = flatten_and_count_by_experience(df, 'experienceLevel', 'Soft Skills')

        fig_exp_soft_skills = px.bar(skill_counts_exp_soft, x='Soft Skills', y='Count', color='experienceLevel',
                                    labels={'Soft Skills': 'Soft Skills', 'Count': 'Count', 'experienceLevel': 'Experience Level'},
                                    barmode='group', height=500)

        fig_exp_soft_skills.update_layout(margin=dict(t=40, b=40), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

        st.plotly_chart(fig_exp_soft_skills, use_container_width=True)
                
        
        
    
    # 11. Experience Level vs. Contract Type 
    with col1:
        st.write("### Experience Level vs. Contract Type")
        fig12 = px.bar(df, x='contractType', color='experienceLevel', 
                    labels={'contractType': 'Contract Type', 'experienceLevel': 'Experience Level'},
                    barmode='group')
        st.plotly_chart(fig12, use_container_width=True)






    # 12. Work Arrangement vs. Applications Count (Violin Plot)
    with col2:
        st.write("### Work Arrangement vs. Applications Count")
        fig13 = px.violin(df, x='work_arrangement', y='applicationsCount', 
                        labels={'work_arrangement': 'Work Arrangement', 'applicationsCount': 'Applications Count'})
        st.plotly_chart(fig13, use_container_width=True)



    # 13. Top 10 Job Titles vs. Company Level (Grouped Bar Chart)
    with col1:
        st.write("### Top 10 Job Titles vs. Company Level")
        top_10_titles = df['title'].value_counts().nlargest(10).index
        filtered_df = df[df['title'].isin(top_10_titles)]
        fig14 = px.histogram(filtered_df, x='title', color='company_level', 
                            labels={'title': 'Job Title', 'company_level': 'Company Level'},
                            barmode='group')
        st.plotly_chart(fig14, use_container_width=True)



    # 14. Applications Count vs. Title (Scatter Plot)
    
    with col1:
        st.write("### Applications Count vs. Top 5 Job Titles")

        df['applicationsCount'] = df['applicationsCount'].replace("Over 200", "200")

        df['applicationsCount'] = pd.to_numeric(df['applicationsCount'], errors='coerce')

        top_5_titles = df.groupby('title')['applicationsCount'].sum().nlargest(5).index

        filtered_df = df[df['title'].isin(top_5_titles)]

        fig15 = px.box(filtered_df, x='title', y='applicationsCount', 
                    labels={'title': 'Job Title', 'applicationsCount': 'Applications Count'}, 
                    points="all",  # Show all points for better visualization
                    color='title')  # Add color for each title

        st.plotly_chart(fig15, use_container_width=True)


    
    st.write("### Applications Count vs. Job Titles and Job Displayed time")

    df['job_displayedTime'] = df['job_displayedTime'].str.replace('w', '', regex=False)  # Remove 'w'
    df['job_displayedTime'] = df['job_displayedTime'].str.replace('+', '', regex=False)  # Remove '+'

    df['job_displayedTime_numeric'] = pd.to_numeric(df['job_displayedTime'], errors='coerce')

    df['applicationsCount'] = pd.to_numeric(df['applicationsCount'], errors='coerce')

    top_5_titles = df.groupby('title')['applicationsCount'].sum().nlargest(5).index

    filtered_df = df[df['title'].isin(top_5_titles)]

    fig = px.bar(filtered_df, 
                x='job_displayedTime_numeric', 
                y='applicationsCount', 
                color='title',  
                labels={'job_displayedTime_numeric': 'Job Displayed Time (Weeks)', 'applicationsCount': 'Applications Count', 'title': 'Job Title'})

    fig.update_layout(
        margin=dict(t=60, b=60), 
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        paper_bgcolor='rgba(0, 0, 0, 0)', 
        showlegend=True  # Show the legend
    )

    st.plotly_chart(fig, use_container_width=True)






























### TAB 3: Job Recommendation ###
with tab3:
    st.subheader("💼 Job Recommendation")
    st.write("""
    **Welcome to the DS related Job Recommendation Tool!**  
    Based on the information you provide, we will match your resume data with the most compatible job opportunities available. 
    Our tool analyzes your technical and soft skills, along with your experience level, to recommend positions that best align with your qualifications.
    """)
    input_file_path = 'user_input.csv'
    output_file_path = 'job_recommendations.txt'
    missing_skills_output_path = 'missing_skills_output.txt'

    # Load the dataset
    file_path = 'Job.csv'
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found at path: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)

    # Prepare the dataset by filling NaN values and creating a combined skills column
    df['combined_skills'] = (
    df['required skills'].fillna('') + ' ' +
    df['Data Engineering'].fillna('') + ' ' +
    df['Cloud Computing'].fillna('') + ' ' +
    df['Soft Skills'].fillna('')
    )


# Create tabs for "Input Data" and "Results"
    tab1, tab2 = st.tabs(["Input Data", "Results"])

# Input Data Section (Tab 1)
    with tab1:
        st.subheader("Enter Your Information")

        col1, col2 = st.columns(2)

        with col1:
            # Dropdown for Technical Skills
            st.write("### Technical Skills")
            technical_skills = {
                "Programming": ["Python", "Java", "C++", "R"],
                "Database": ["MySQL", "PostgreSQL", "MongoDB", "SQLite"],
                "Data Engineering": ["Apache Spark", "Apache Kafka", "Airflow", "Hadoop"],
                "Cloud Computing": ["AWS", "Azure", "Google Cloud", "IBM Cloud"]
            }

            programming_lang = st.multiselect("Programming", technical_skills["Programming"])
            database = st.multiselect("Database", technical_skills["Database"])
            data_engineering = st.multiselect("Data Engineering", technical_skills["Data Engineering"])
            cloud_computing = st.multiselect("Cloud Computing", technical_skills["Cloud Computing"])

            # Dropdown for Soft Skills
            st.write("### Soft Skills")
            soft_skills = ["Communication", "Teamwork", "Leadership", "Problem Solving"]
            selected_soft_skills = st.multiselect("Soft Skills", soft_skills)

            # Experience Level
            st.write("### Experience Level")
            experience_levels = ["Junior", "Mid", "Senior", "Lead"]
            experience_level = st.selectbox("Experience Level", experience_levels)

        with col2:
            # Dropdown for Education Level
            st.write("### Education Level")
            education_levels = ["Bachelor's", "Master's", "PhD", "Other"]
            education_level = st.selectbox("Education Level", education_levels)

            # Dropdown for Language Skills
            st.write("### Language Skills")
            language_skills = ["English", "Finnish", "Spanish", "German"]
            language_skill = st.multiselect("Language Skills", language_skills)

            # Text Input for Project Experience
            st.write("### Project Experience (200 words max)")
            project_experience = st.text_area("Describe your project experience", max_chars=200)

            # Preferences Section
            st.write("### Your Preferences")
            
        col3, col4 = st.columns(2)

        with col3:
            work_arrangements = ["Remote", "On-site", "Hybrid"]
            work_arrangement = st.multiselect("Work Arrangement", work_arrangements)

            work_sectors = ["Technology", "Finance", "Healthcare", "Education"]
            work_sector = st.multiselect("Work Sector", work_sectors)

        with col4:
            work_types = ["Full-time", "Part-time", "Contract", "Internship"]
            work_type = st.multiselect("Work Type", work_types)

            contract_types = ["Permanent", "Temporary", "Freelance"]
            contract_type = st.multiselect("Contract Type", contract_types)

            company_levels = ["Startup", "Small Company", "Medium Company", "Large Company"]
            company_level = st.multiselect("Company Level", company_levels)

        # Button to Save Preferences
        if st.button("Save Preferences"):
            # Validate input before saving
            if not project_experience or not selected_soft_skills or not experience_level or not work_arrangement or not work_sector:
                st.error("Please fill in all fields before saving!")
            else:
                user_input = {
                    "project_experience": project_experience,
                    "programming_lang": ', '.join(programming_lang),
                    "database": ', '.join(database),
                    "data_engineering": ', '.join(data_engineering),
                    "cloud_computing": ', '.join(cloud_computing),
                    "language_skills": ', '.join(language_skill),
                    "soft_skills": ', '.join(selected_soft_skills),
                    "experience_level": experience_level,
                    "education_level": education_level,
                    "work_arrangement": ', '.join(work_arrangement),
                    "work_sector": ', '.join(work_sector),
                    "work_type": ', '.join(work_type),
                    "contract_type": ', '.join(contract_type),
                    "company_level": ', '.join(company_level)
                }
                # Save input to a CSV file
                input_df = pd.DataFrame([input_file_path])
                input_df.to_csv(input_file_path, index=False)
                st.success("Preferences saved successfully!")

        # Button to Proceed
        if st.button("Proceed"):
            # Check if user input is saved
            if not os.path.exists(input_file_path):
                st.error("Please save your preferences first!")
            else:
                # Simulate loading with a progress bar
                st.write("Processing your information...")
                progress_bar = st.progress(0)
                for percent_complete in range(101):
                    time.sleep(0.05)  # Simulating loading time
                    progress_bar.progress(percent_complete)

                # Read the input file to use as cv_input
                input_df = pd.read_csv(input_file_path)
                
                cv_input = {
                    'project_description': input_df['project_experience'][0],
                    'soft_skills': input_df['soft_skills'][0],
                    'experienceLevel': input_df['experience_level'][0],
                    'contractType': input_df['contract_type'][0],
                    'work_arrangement': input_df['work_arrangement'][0],
                    'sector': input_df['work_sector'][0],
                    'workType': input_df['work_type'][0],
                    'company_level': input_df['company_level'][0],
                    'programming_lang': input_df['programming_lang'][0],
                    'database': input_df['database'][0],
                    'data_engineering': input_df['data_engineering'][0],
                    'cloud_computing': input_df['cloud_computing'][0],
                    'education_level': input_df['education_level'][0],
                    'language_skills': input_df['language_skill'][0]
                }

                # Processing logic for CV input
                cv_combined_input = (
                    cv_input['project_description'] + ' ' +
                    cv_input['soft_skills'] + ' ' +
                    cv_input['experienceLevel'] + ' ' +
                    cv_input['contractType'] + ' ' +
                    cv_input['work_arrangement'] + ' ' +
                    cv_input['sector'] + ' ' +
                    cv_input['workType'] + ' ' +
                    cv_input['company_level'] + ' ' +
                    cv_input['programming_lang'] + ' ' +
                    cv_input['database'] + ' ' +
                    cv_input['data_engineering'] + ' ' +
                    cv_input['cloud_computing'] + ' ' +
                    cv_input['education_level'] + ' ' +
                    cv_input['language_skills']
                ).lower()

                cv_skills_set = set(skill.strip().lower() for skill in cv_input['soft_skills'].split(', '))

                # Using TF-IDF Vectorizer
                vectorizer = TfidfVectorizer(stop_words='english')

                # Filter jobs based on sector preference
                filtered_df = df[
                    df['sector'].str.contains('|'.join(cv_input['sector'].split(', ')), case=False, na=False)
                ]

                # Calculate cosine similarity
                if not filtered_df.empty:
                    job_descriptions_tfidf = vectorizer.fit_transform(filtered_df['description'].fillna(''))
                    cv_input_tfidf = vectorizer.transform([cv_combined_input])
                    cosine_similarities = cosine_similarity(cv_input_tfidf, job_descriptions_tfidf).flatten()

                    # Add the cosine similarity scores to the filtered DataFrame
                    filtered_df['similarity'] = cosine_similarities

                    # Sort and take the top 10 based on similarity
                    top_10_jobs = filtered_df.sort_values(by='similarity', ascending=False).head(10)

                    # Convert similarity scores to percentage
                    top_10_jobs['similarity'] = top_10_jobs['similarity'] * 100

                    # Prepare result for saving
                    result = top_10_jobs[['title', 'companyName', 'workType', 'general_job_category', 'sector', 'jobUrl', 'work_arrangement', 'similarity', 'job_title_desc']]
                    result = result.rename(columns={'similarity': 'similarity (%)'})

                    # Save the results to a text file
                    result.to_csv(output_file_path, sep='\t', index=False)
                    st.success(f"Results saved to {output_file_path}")

                    ################ Updated Missing Skills by Sector ################
                    # Suggest missing skills classified by sector
                    missing_skills_by_sector = {}

                    for index, row in top_10_jobs.iterrows():
                        # Extract the required skills and categorize them
                        job_sector = row['sector']

                        required_skills = row['required skills'].split(', ')
                        data_engineering_skills = row['Data Engineering'].split(', ')
                        cloud_computing_skills = row['Cloud Computing'].split(', ')
                        soft_skills = row['Soft Skills'].split(', ')

                        # Initialize sector in the dictionary if not present
                        if job_sector not in missing_skills_by_sector:
                            missing_skills_by_sector[job_sector] = {
                                'Required Skills': set(),
                                'Data Engineering': set(),
                                'Cloud Computing': set(),
                                'Soft Skills': set()
                            }

                        # Calculate missing skills for each job
                        required_missing = set(required_skills) - cv_skills_set
                        data_eng_missing = set(data_engineering_skills) - cv_skills_set
                        cloud_comp_missing = set(cloud_computing_skills) - cv_skills_set
                        soft_skills_missing = set(soft_skills) - cv_skills_set

                        # Add missing skills to the corresponding sector category
                        missing_skills_by_sector[job_sector]['Required Skills'].update(required_missing)
                        missing_skills_by_sector[job_sector]['Data Engineering'].update(data_eng_missing)
                        missing_skills_by_sector[job_sector]['Cloud Computing'].update(cloud_comp_missing)
                        missing_skills_by_sector[job_sector]['Soft Skills'].update(soft_skills_missing)

                    # Save the missing skills data to a text file
                    with open(missing_skills_output_path, 'w') as f:
                        for sector, skill_types in missing_skills_by_sector.items():
                            f.write(f"Sector: {sector}\n")
                            for skill_type, missing_skills in skill_types.items():
                                f.write(f"{skill_type}: {', '.join(missing_skills)}\n")
                            f.write("\n")

                    st.success(f"Missing skills saved to {missing_skills_output_path}")

                else:
                    st.warning("No jobs found in the selected sector.")






    # Results Section (Tab 2)
    # Results Section (Tab 2)
    with tab2:
        st.subheader("Your Job Recommendations")

        # Check if output file exists
        if os.path.exists(output_file_path):
            st.write("### Top 10 Recommended Jobs:")

            # Load the output file and display it as a table
            result_df = pd.read_csv(output_file_path, sep='\t')
            st.table(result_df)

            # Calculate Average Similarity for Gauge Meter
            average_similarity = result_df['similarity (%)'].mean()

            # Display Average Similarity as a metric
            st.metric(label="Average CV Potential Score", value=f"{average_similarity:.2f}%")

            # Plot a gauge chart to display the score percentage
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=average_similarity,
                title={'text': "CV Potential Score", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': average_similarity}
                }
            ))

            fig.update_layout(
                paper_bgcolor="black",
                font={'color': "white", 'family': "Arial"},
                height=400, width=600
            )

            # Display the gauge chart in Streamlit
            st.plotly_chart(fig)

            # Download button to download the table as CSV
            st.download_button(
                label="Download recommendations as CSV",
                data=result_df.to_csv(index=False),
                file_name="job_recommendations.csv",
                mime="text/csv"
            )

        else:
            st.warning("No output available. Please proceed from Tab 1 to generate recommendations.")

        ################### Missing Skills Output Section ######################

        # Check if the missing skills file exists
        if os.path.exists(missing_skills_output_path):
            st.write("### Missing Skills by Sector")

            # Helper function to parse the missing skills file
            def parse_missing_skills_file(file_path):
                missing_skills_by_sector = {}
                current_sector = None
                current_skill_type = None

                with open(file_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith("Sector:"):
                            current_sector = line.split("Sector: ")[1].strip()
                            missing_skills_by_sector[current_sector] = {
                                'Required Skills': [],
                                'Data Engineering': [],
                                'Cloud Computing': [],
                                'Soft Skills': []
                            }
                        elif line.startswith("Required Skills:"):
                            current_skill_type = 'Required Skills'
                            missing_skills_by_sector[current_sector]['Required Skills'] = line.split(": ")[1].strip().split(", ")
                        elif line.startswith("Data Engineering:"):
                            current_skill_type = 'Data Engineering'
                            missing_skills_by_sector[current_sector]['Data Engineering'] = line.split(": ")[1].strip().split(", ")
                        elif line.startswith("Cloud Computing:"):
                            current_skill_type = 'Cloud Computing'
                            missing_skills_by_sector[current_sector]['Cloud Computing'] = line.split(": ")[1].strip().split(", ")
                        elif line.startswith("Soft Skills:"):
                            current_skill_type = 'Soft Skills'
                            missing_skills_by_sector[current_sector]['Soft Skills'] = line.split(": ")[1].strip().split(", ")

                return missing_skills_by_sector

            # Parse the missing skills file
            missing_skills_by_sector = parse_missing_skills_file(missing_skills_output_path)

            # Get all sectors and add an "All Sectors" option
            all_sectors = list(missing_skills_by_sector.keys())
            all_sectors.append("All Sectors")

            # Sector filter (drop-down menu)
            selected_sector = st.selectbox("Select Sector to View Required Skills:", all_sectors)

            # Display missing skills immediately upon sector selection
            if selected_sector != "All Sectors":
                st.subheader(f"Skills for Sector: {selected_sector}")

                # Display skills for each category
                for skill_type in ['Required Skills', 'Data Engineering', 'Cloud Computing', 'Soft Skills']:
                    skills = missing_skills_by_sector[selected_sector][skill_type]
                    with st.expander(skill_type, expanded=True):  # Set expanded to True
                        if skills:
                            st.write(", ".join(skills))
                        else:
                            st.write(f"All required {skill_type.lower()} are present.")

            else:
                st.write("### Required Skills Across All Sectors")

                # Initialize sets to store common skills
                common_skills = {
                    'Required Skills': set(),
                    'Data Engineering': set(),
                    'Cloud Computing': set(),
                    'Soft Skills': set()
                }

                # Iterate over each sector and accumulate skills into the common set
                for sector, skills in missing_skills_by_sector.items():
                    for skill_type, skill_list in skills.items():
                        common_skills[skill_type].update(skill_list)

                # Display common skills across all sectors
                for skill_type, skills in common_skills.items():
                    with st.expander(f"{skill_type} (All Sectors)", expanded=True):  # Set expanded to True
                        if skills:
                            st.write(", ".join(skills))
                        else:
                            st.write(f"No missing {skill_type.lower()} across all sectors.")

            # Download button for the missing skills output file
            with open(missing_skills_output_path, "r") as file:
                st.download_button(
                    label="Download Missing Skills File",
                    data=file.read(),
                    file_name="Required_skills_output.txt",
                    mime="text/plain"
                )

        else:
            st.warning("No missing skills output file found. Please proceed from Tab 1 to generate the missing skills report.")

        # Button to refresh (delete output file) at the bottom
        st.markdown("---")  # Separator line for better UI
        if st.button("Refresh"):
            os.remove(output_file_path)
            st.success("Output file deleted. Please proceed again from Tab 1.")
   












### TAB 4: About Us ###
with tab4:
    st.subheader("👥 About Us")
    st.write("""
    **Meet the Team Behind the DS Job Market Analysis Tool**

    1. 	:female-technologist:[Gayathri Senanayake](https://www.linkedin.com/in/kavya-atapattu-9057501b3)
    2. 	:female-technologist:[Dilusha Senarathna](https://www.linkedin.com/in/kavya-atapattu-9057501b3)
    3. 	:female-technologist:[Kavya Atapattu](https://www.linkedin.com/in/kavya-atapattu-9057501b3)

    We are passionate about Data Science and committed to helping you navigate the job market more effectively. Feel free to connect with us on LinkedIn!
    """)
