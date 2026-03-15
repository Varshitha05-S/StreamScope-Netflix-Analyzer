# StreamScope: Netflix Content Strategy Analyzer

A data visualization and analytics project that analyzes Netflix’s content catalog using the Kaggle Netflix Movies and TV Shows dataset.

## Objective
To uncover trends in Netflix’s content strategy such as:
- Content growth over time
- Movies vs TV Shows distribution
- Genre and rating popularity
- Country-wise content contribution
- Advanced analytics using clustering and classification

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn
- Streamlit (Dashboard)

## Project Structure
app/        -> Streamlit dashboard code  
data/       -> Raw and cleaned datasets  
notebooks/  -> Data cleaning, EDA, feature engineering, modeling  
reports/    -> Screenshots, insights, and summary report  

## Current Progress
- Repository initialized
- Virtual environment created
- Required libraries installed
- Folder structure created

## Dataset
Netflix Movies and TV Shows dataset (Kaggle)

## Next Steps
- Download dataset
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- ML: Clustering + Classification
- Dashboard development and deployment
## Milestone 2 – EDA & Feature Engineering Completed

Implemented:

- Netflix content growth analysis over time
- Distribution of Movies vs TV Shows
- Top Genres analysis
- Country-level content contribution
- Rating distribution insights
- Content Length Category feature:
    - Short Movie
    - Medium Movie
    - Long Movie
    - Limited Series
    - Multi-Season
    - Long Running Series
- Interactive dashboard with filtering:
    - Year
    - Genre
    - Country
    - Rating
    - Content Type

Note:
The dataset does not explicitly provide a column to distinguish Netflix Originals from Licensed content.
Therefore, this feature was not derived to maintain data integrity.git push origin milestone-2-dashboard