import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import shap
import ast
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']


print("[1/4] Loading cleaned dataset...")
df = pd.read_csv('cleaned_data.csv')

# Extract feature columns for services and barriers
svc_cols = [c for c in df.columns if c.startswith('svc_')]
bar_cols = [c for c in df.columns if c.startswith('bar_')]

# =====================================================================
# Data Foundation & Demographics
# =====================================================================
print("[2/4] Generating charts for Speaker 1 (Demographics & Personas)...")

# --- Chart 1: Demographics (Age and Cancer Status) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1A: Age Distribution
sns.countplot(y='age_group_filled', data=df, order=df['age_group_filled'].value_counts().index, ax=axes[0], palette='Blues_r')
axes[0].set_title('Chart 1A: Client Age Distribution', fontsize=14, weight='bold')
axes[0].set_xlabel('Number of Clients')
axes[0].set_ylabel('Age Group')

# 1B: Cancer Status Distribution
sns.countplot(y='cancer_status_filled', data=df, order=df['cancer_status_filled'].value_counts().index, ax=axes[1], palette='Greens_r')
axes[1].set_title('Chart 1B: Client Cancer Status', fontsize=14, weight='bold')
axes[1].set_xlabel('Number of Clients')
axes[1].set_ylabel('Status')

plt.tight_layout()
plt.savefig('1_Demographics.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 2: Unsupervised Learning - Client Persona Clustering ---
# Fill NaNs with 0 for clustering
X_cluster = df[svc_cols].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# Calculate the mean service usage for each cluster
cluster_summary = df.groupby('Cluster')[svc_cols].mean()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary.T, cmap='Blues')
plt.title('Chart 2: Client Persona Clustering by Service Preferences', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('2_Cluster_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()


# =====================================================================
# Exceptional Performance & Hidden Bottlenecks
# =====================================================================
print("[3/4] Generating charts for Speaker 2 (NPS Trends, SHAP & Bottlenecks)...")

# --- Chart 3: Bulletproof Satisfaction - YoY NPS Trend ---
nps_yearly = df.groupby('year')['nps_score_filled'].mean()

fig, ax1 = plt.subplots(figsize=(10, 6))
color = '#2ca02c' # Green for positive satisfaction
ax1.plot(nps_yearly.index, nps_yearly.values, marker='o', markersize=10, color=color, linewidth=3)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Average NPS Score (out of 10)', color=color, fontsize=12, weight='bold')
ax1.set_ylim(8.5, 10.2)
ax1.set_xticks(nps_yearly.index)

# Annotate values on the plot
for i, v in enumerate(nps_yearly.values):
    ax1.text(nps_yearly.index[i], v + 0.08, f"{v:.2f}", ha='center', color=color, weight='bold', fontsize=12)

plt.title('Chart 3: Bulletproof Satisfaction - YoY NPS Trend', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('3_YoY_NPS_Trend.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 4: Supervised Learning - Core Drivers of NPS (SHAP) ---
features = svc_cols + bar_cols
X_shap = df[features].fillna(0)
y_shap = df['nps_score_filled'].fillna(df['nps_score_filled'].median())

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_shap, y_shap)

# Extract SHAP values for model explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_shap)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, show=False, max_display=12)
plt.title('Chart 4: Core Drivers of NPS (Random Forest & SHAP)', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('4_SHAP_Drivers.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 5: The Shifting Bottleneck (Barriers YoY) ---
barriers_yoy = df.groupby('year')[bar_cols].mean() * 100

plt.figure(figsize=(10, 6))
plt.plot(barriers_yoy.index, barriers_yoy['bar_long_wait_for_services'], marker='s', markersize=8, linewidth=3, color='#ff7f0e', label='Long Wait for Services')
plt.plot(barriers_yoy.index, barriers_yoy['bar_inconvenient_service_times'], marker='o', markersize=8, linewidth=3, color='#d62728', label='Inconvenient Times')

plt.title('Chart 5: The Shifting Bottleneck (2022-2025)', fontsize=16, weight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('% of Clients Reporting Barrier', fontsize=12)
plt.xticks(barriers_yoy.index)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('5_YoY_Barriers_Trend.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 6: Cross-Analysis - The Working-Age Blind Spot ---
time_barrier_by_age = df.groupby('age_group_filled')['bar_inconvenient_service_times'].mean() * 100
age_order = ['35-44', '45-54', '55-64', '65-74', '75 or older']
time_barrier_by_age = time_barrier_by_age.reindex(age_order)

plt.figure(figsize=(10, 6))
sns.barplot(x=time_barrier_by_age.index, y=time_barrier_by_age.values, palette='magma_r')
plt.title('Chart 6: The Working-Age Blind Spot (Inconvenient Time by Age)', fontsize=16, weight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('% Reporting Inconvenient Service Times', fontsize=12)

# Annotate percentages on bars
for index, value in enumerate(time_barrier_by_age.values):
    plt.text(index, value + 0.5, f"{value:.1f}%", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('6_Age_vs_TimeBarrier.png', dpi=300, bbox_inches='tight')
plt.close()


# =====================================================================
# Social Impact, Advanced NLP & Strategic Bundling
# =====================================================================
print("[4/4] Generating charts for Speaker 3 (Social Impact, Advanced NLP & Bundling)...")

# --- Chart 7: Quantifying Social Impact ROI ---
def parse_impacts(val):
    try:
        return ast.literal_eval(val) # Safely parse stringified lists
    except:
        return []

df['impacts_list'] = df['impacts_reported'].apply(parse_impacts)
all_impacts = [impact for sublist in df['impacts_list'] for impact in sublist]
impact_counts = pd.Series(all_impacts).value_counts().head(6)

# Truncate overly long impact descriptions for better layout
short_labels = [label[:45] + '...' if len(label) > 45 else label for label in impact_counts.index]

plt.figure(figsize=(12, 6))
sns.barplot(x=impact_counts.values, y=short_labels, palette='Purples_r')
plt.title('Chart 7: Social Impact - Top Real-World Benefits', fontsize=15, weight='bold')
plt.xlabel('Number of Clients Reporting the Benefit')
plt.tight_layout()
plt.savefig('7_Social_Impact.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 8: NLP Voice of Customer (Word Cloud with Domain Filters) ---
text_data = ' '.join(df['feedback_text'].dropna().astype(str).tolist())

# Filter out domain-specific noise words to reveal true sentiment
custom_stopwords = set(STOPWORDS)
noise_words = [
    'dempsey', 'center', 'cancer', 'service', 'services',
    'people', 'one', 'us', 'client', 'clients', 'much',
    'really', 'time', 'would', 'also', 'even'
]
custom_stopwords.update(noise_words)

wordcloud = WordCloud(
    width=1200, height=600,
    background_color='white',
    colormap='viridis',
    max_words=80,
    stopwords=custom_stopwords
).generate(text_data)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Chart 8: Voice of Customer - Feedback Word Cloud', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('8_WordCloud_Filtered.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 9: Market Basket Analysis (Association Rules) ---
top_services = df[svc_cols].sum().sort_values(ascending=False).head(10).index
df_top = df[top_services]

# Calculate Co-occurrence and Confidence Matrix
co_occurrence = df_top.T.dot(df_top)
occurrences = np.diag(co_occurrence)
confidence = co_occurrence.divide(occurrences, axis=0)
np.fill_diagonal(confidence.values, np.nan) # Remove self-references

# Clean up service labels
labels = [c.replace('svc_', '').replace('_', ' ').title() for c in top_services]
confidence.columns = labels
confidence.index = labels

plt.figure(figsize=(12, 9))
sns.heatmap(confidence, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=0.75,
            cbar_kws={'label': 'Probability (Confidence)'}, linewidths=.5, square=True)
plt.title('Chart 9: Market Basket Analysis (Cross-Selling Probabilities)', fontsize=16, weight='bold', pad=20)
plt.ylabel('Base Service (If they use...)', fontsize=12, weight='bold')
plt.xlabel('Associated Service (...how likely are they to also use this?)', fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig('9_Market_Basket_Analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Chart 10: Advanced NLP - VADER Sentiment vs NPS Score ---
df_text = df.dropna(subset=['feedback_text']).copy()
sia = SentimentIntensityAnalyzer()

# Calculate Compound Semantic Score (-1 to +1)
df_text['sentiment_score'] = df_text['feedback_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='nps_category', y='sentiment_score', data=df_text,
            order=['Promoter', 'Passive', 'Detractor'],
            palette=['#2ca02c', '#ff7f0e', '#d62728'])
sns.stripplot(x='nps_category', y='sentiment_score', data=df_text,
              order=['Promoter', 'Passive', 'Detractor'],
              color='black', alpha=0.3, jitter=True)

plt.title('Chart 10: Advanced NLP Sentiment Analysis by NPS Category', fontsize=16, weight='bold')
plt.xlabel('NPS Category (Survey Score)', fontsize=12)
plt.ylabel('Semantic Sentiment Score (from -1 to +1)', fontsize=12)
plt.axhline(0, ls='--', color='gray', alpha=0.7) # Baseline zero

plt.tight_layout()
plt.savefig('10_NLP_Sentiment_Boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

constructive_critics = df_text[(df_text['nps_category'] == 'Promoter') & (df_text['sentiment_score'] < 0)]
print("\n=== ADVANCED BUSINESS INSIGHTS ===")
print(f"Discovered {len(constructive_critics)} 'Constructive Critics' (Promoters who expressed negative sentiment in text):")
if len(constructive_critics) > 0:
    for text in constructive_critics['feedback_text'].head(3):
        print(f" - \"{text[:150]}...\"")