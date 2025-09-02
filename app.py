# Exploratory Data Analysis (EDA) Toolkit for Prostate Cancer Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (expects 'pros.csv' in the same directory)
def load_data(file_path='pros.csv'):
    df = pd.read_csv(file_path)
    return df

# Clean data: fill NA, drop missing RACE
def clean_data(df):
    df['VOL'] = df['VOL'].fillna(df['VOL'].mean())
    df.dropna(subset=['RACE'], inplace=True)
    return df

# Display basic statistics and info
def basic_info(df):
    print("--- Data Info ---")
    print(df.info())
    print("\n--- Description ---")
    print(df.describe())
    print("\n--- Duplicates ---")
    print(f"{df.duplicated().sum()} duplicate rows")
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

# Plot correlation heatmap
def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Univariate distribution plots
def plot_distributions(df):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['AGE'], bins=10, kde=True)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='RACE', data=df)
    plt.title('Distribution of Race')
    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.show()

    df[['RACE']].boxplot()
    df[['AGE']].boxplot()

# Bivariate analysis plots
def bivariate_analysis(df):
    avg_tumor_vol = df.groupby(['AGE', 'RACE'])['VOL'].mean().reset_index()
    sns.lineplot(x='AGE', y='VOL', hue='RACE', data=avg_tumor_vol)
    plt.title('Average Tumor Volume by Age and Race')
    plt.xlabel('Age')
    plt.ylabel('Average Tumor Volume')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='AGE', y='GLEASON', hue='RACE', data=df)
    plt.title('Gleason Score by Age and Race')
    plt.xlabel('Age')
    plt.ylabel('Gleason Score')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='AGE', y='PSA', hue='RACE', data=df)
    plt.title('PSA Levels by Age and Race')
    plt.xlabel('Age')
    plt.ylabel('PSA Levels')
    plt.show()

# Categorical relationship plot
def plot_categorical_relationship(df):
    contingency_table = pd.crosstab(df['DCAPS'], df['CAPSULE'])
    print("--- Contingency Table ---")
    print(contingency_table)
    contingency_table.plot(kind='bar', stacked=True)
    plt.title('Relationship between DCAPS and CAPSULE')
    plt.xlabel('DCAPS')
    plt.ylabel('Count')
    plt.legend(title='CAPSULE')
    plt.show()

# Dual-axis plot: CAPSULE vs PSA by Gleason
def plot_gleason_effects():
    gleason_scores = [0, 4, 5, 6, 7, 8, 9]
    capsule_proportion = [0.0, 0.0, 0.089552, 0.273381, 0.570312, 0.800000, 0.923077]
    psa_values = [3.9, 14.3, 8.179104, 8.804964, 19.776484, 34.433333, 38.223077]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Gleason Score')
    ax1.set_ylabel('Capsule Proportion', color='blue')
    ax1.plot(gleason_scores, capsule_proportion, color='blue', marker='o', label='Capsule Proportion')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(gleason_scores)

    ax2 = ax1.twinx()
    ax2.set_ylabel('PSA (Mean)', color='orange')
    ax2.plot(gleason_scores, psa_values, color='orange', marker='s', linestyle='--', label='PSA (Mean)')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Capsule Proportion and PSA by Gleason Score')
    fig.tight_layout()
    plt.show()

# Example execution pipeline (can be wrapped in main)
df = load_data()
df = clean_data(df)
basic_info(df)
plot_correlation(df)
plot_distributions(df)
bivariate_analysis(df)
plot_categorical_relationship(df)
plot_gleason_effects()
