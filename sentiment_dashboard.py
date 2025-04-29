import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import numpy as np

# ✅ Safe Page Configuration
st.set_page_config(page_title="Sentiment Insights", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:bold;
    color:#1E90FF;
}
.metric-container {
    background-color:#F0F8FF;
    border-radius:10px;
    padding:10px;
    margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

# ✅ Load and preprocess data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

        df['Sentiment'] = df['Sentiment'].astype(str)
        df['Platform'] = df['Platform'].astype(str)
        df['Country'] = df['Country'].astype(str)
        df['Hashtags'] = df['Hashtags'].astype(str)
        df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce').fillna(0).astype(int)
        df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce').fillna(0)
        df['Retweets'] = pd.to_numeric(df['Retweets'], errors='coerce').fillna(0)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load dataset
file_path = "C:/Users/madha/Downloads/sentimentdataset.csv"
df = load_data(file_path)

if df is not None and not df.empty:
    st.sidebar.header("🔍 Dashboard Filters")

    selected_platforms = st.sidebar.multiselect(
        "Select Platforms",
        options=df['Platform'].unique()
    )
    selected_sentiments = st.sidebar.multiselect(
        "Select Sentiments",
        options=df['Sentiment'].unique()
    )
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=df['Country'].unique()
    )

    # ❗ If nothing is selected, assume everything is selected
    filtered_df = df.copy()

    if selected_platforms:
        filtered_df = filtered_df[filtered_df['Platform'].isin(selected_platforms)]

    if selected_sentiments:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]

    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    st.title("📊 Sentiment Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Posts", len(filtered_df), "🔢")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Likes", f"{filtered_df['Likes'].mean():.2f}", "👍")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Retweets", f"{filtered_df['Retweets'].mean():.2f}", "🔁")
        st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment Insights", "Platform & Engagement", "Temporal Analysis", "Advanced Correlations"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_df['Sentiment'].value_counts()
            if not sentiment_counts.empty:
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Breakdown",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.warning("No sentiment data to display.")

        with col2:
            st.subheader("Engagement by Sentiment")
            if not filtered_df.empty:
                fig_box = px.box(
                    filtered_df,
                    x="Sentiment",
                    y="Likes",
                    color="Sentiment",
                    title="Likes Distribution Across Sentiments",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("No data for box plot.")

    with tab2:
        st.subheader("Platform Performance")
        platform_sentiment = filtered_df.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0)

        if not platform_sentiment.empty:
            fig_platform = px.bar(
                platform_sentiment,
                title="Sentiment Distribution Across Platforms",
                labels={'value': 'Number of Posts', 'index': 'Platform'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_platform, use_container_width=True)
        else:
            st.warning("No platform data to display.")

        st.subheader("Retweets vs Likes")
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df,
                x="Retweets",
                y="Likes",
                color="Sentiment",
                size="Likes",
                hover_data=['Platform', 'Country'],
                title="Engagement Correlation",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No data for scatter plot.")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Peak Posting Hours")
            hour_counts = filtered_df['Hour'].value_counts().sort_index()
            if not hour_counts.empty:
                fig_hours = px.line(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    title="Posting Activity by Hour",
                    labels={'x': 'Hour', 'y': 'Number of Posts'},
                    color_discrete_sequence=['#1E90FF']
                )
                st.plotly_chart(fig_hours, use_container_width=True)
            else:
                st.warning("No hourly data available.")

        with col2:
            st.subheader("Top Countries")
            country_counts = filtered_df['Country'].value_counts().head(10)
            if not country_counts.empty:
                fig_countries = px.bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    title="Top 10 Countries by Post Volume",
                    labels={'x': 'Country', 'y': 'Number of Posts'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_countries, use_container_width=True)
            else:
                st.warning("No country data to display.")

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Hashtags")
            hashtag_list = ' '.join(filtered_df['Hashtags'].dropna()).split()
            hashtag_counts = Counter(hashtag_list).most_common(10)

            if hashtag_counts:
                hashtags, counts = zip(*hashtag_counts)
                fig_hashtags = px.bar(
                    x=hashtags,
                    y=counts,
                    title="Top 10 Hashtags",
                    labels={'x': 'Hashtag', 'y': 'Frequency'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_hashtags, use_container_width=True)
            else:
                st.warning("No hashtags found.")

        with col2:
            st.subheader("Correlation Heatmap")
            numeric_cols = ['Likes', 'Retweets', 'Hour']
            if not filtered_df[numeric_cols].empty:
                corr_matrix = filtered_df[numeric_cols].corr()
                fig_heatmap = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Not enough data for correlation heatmap.")

    st.markdown("---")
    st.markdown("📊 Sentiment Analysis Dashboard | Powered by Streamlit & Plotly")

else:
    st.error("Failed to load the dataset or dataset is empty.")
# streamlit run sentiment_dashboard.py