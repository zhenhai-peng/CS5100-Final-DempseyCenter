# CS5100-Final-DempseyCenter

# Dempsey Center Client Survey Analysis (2022–2025)

This repository contains the data cleaning, feature engineering, and analytical pipeline for a longitudinal study of Dempsey Center annual client surveys from 2022 to 2025.

The project applies data engineering, clustering, machine learning, and NLP techniques to identify hidden service barriers, understand client service preferences, and generate operational recommendations for nonprofit service optimization.

## Project Highlights

- Integrated and standardized multi-year survey data with schema drift
- Built a cleaning pipeline for inconsistent service names, barriers, and demographics
- Performed client persona discovery using K-Means clustering
- Analyzed key satisfaction drivers using Random Forest and SHAP
- Applied NLP techniques to open-ended feedback, including sentiment analysis and word cloud generation
- Conducted association analysis to identify potential service bundling opportunities

## Main Insight

Although overall satisfaction remains consistently high, the analysis shows that working-age clients face a disproportionately high scheduling barrier. This suggests that improving service accessibility and scheduling flexibility may create more impact than simply adding new services.

## Repository Structure

├── data_clean.py              # Data cleaning and feature engineering pipeline
├── analysis.py                # Main analysis and visualization script
├── cleaned_data.csv           # Final cleaned dataset used for modeling
├── cleaned_data.pkl           # Serialized cleaned dataset
