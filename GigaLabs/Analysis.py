# import survey data from raw file in folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import datetime
import time
import math
import nltk
import itertools
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")


# impot xlsx file and read data from it
df = pd.read_excel("raw.xlsx")
df.head()


nltk.download("punkt")
nltk.download("stopwords")


# QUESTION 1 HEAT MAP FOR SIMILARITY BETWEEN OPEN ENDED QUESTIONS

questions = [
    "What’s one word you associate with Spaghetti?",
    "Why would you buy this spaghetti in your own words?",
    "Why would you NOT buy this spaghetti in your own words?",
]

# Initialize an empty list to store dictionaries
data_list = []

# Iterate over each unique identifier
unique_identifiers = df["Unique Identifier"].unique()
for identifier in unique_identifiers:
    # Get a list of unique pairs of concepts
    concept_pairs = list(itertools.combinations(df["Product name"].unique(), 2))

    # Iterate over each pair of concepts
    for concept1, concept2 in concept_pairs:
        for question in questions:
            # Get responses for the current unique identifier, concept pair, and question
            responses_concept1 = df[
                (df["Unique Identifier"] == identifier)
                & (df["Product name"] == concept1)
            ][question].values
            responses_concept2 = df[
                (df["Unique Identifier"] == identifier)
                & (df["Product name"] == concept2)
            ][question].values

            # Check if there are responses for both concepts
            if len(responses_concept1) > 0 and len(responses_concept2) > 0:
                # Preprocess the responses
                preprocessed_text1 = " ".join(
                    str(response)
                    for response in responses_concept1
                    if not pd.isna(response)
                )
                preprocessed_text2 = " ".join(
                    str(response)
                    for response in responses_concept2
                    if not pd.isna(response)
                )

                # Check if the documents are not empty
                if preprocessed_text1 and preprocessed_text2:
                    # Calculate cosine similarity
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(
                        [preprocessed_text1, preprocessed_text2]
                    )
                    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][
                        0
                    ]

                    # Append the data to the list
                    data_list.append(
                        {
                            "Unique Identifier": identifier,
                            "Concept Pair": f"{concept1} - {concept2}",
                            "Question": question,
                            "Cosine Similarity": similarity,
                        }
                    )

# Convert the list of dictionaries to a DataFrame
similarity_data = pd.DataFrame(data_list)

# Pivot the DataFrame for visualization
similarity_data_long = similarity_data.pivot_table(
    index=["Unique Identifier", "Concept Pair", "Question"], values="Cosine Similarity"
).reset_index()

# Plot a heatmap
heatmap = sns.heatmap(
    similarity_data_long.pivot_table(
        index="Unique Identifier",
        columns=["Concept Pair", "Question"],
        values="Cosine Similarity",
    ),
    cmap="coolwarm",
    annot=True,
)
# Show the plot
plt.show()

# This is graph for different Spaaghetti brands
if df.empty:
    print("DataFrame is empty. No data to plot.")
else:
    # Extracting the columns related to spaghetti brands
    spaghetti_brands_columns = df[
        [
            "Which spaghetti brands have you purchased in the last 3 months? - Barilla",
            "Which spaghetti brands have you purchased in the last 3 months? - Store brand (e.g. Signature Select, 365 by Whole Foods, good + gather)",
            "Which spaghetti brands have you purchased in the last 3 months? - De Cecco",
            "Which spaghetti brands have you purchased in the last 3 months? - Rummo",
            "Which spaghetti brands have you purchased in the last 3 months? - Bionaturae",
            "Which spaghetti brands have you purchased in the last 3 months? - Rao’s Homemade",
            "Which spaghetti brands have you purchased in the last 3 months? - Montebello",
            "Which spaghetti brands have you purchased in the last 3 months? - Ronzoni",
            "Which spaghetti brands have you purchased in the last 3 months? - Banza",
            "Which spaghetti brands have you purchased in the last 3 months? - Tolerant",
            "Which spaghetti brands have you purchased in the last 3 months? - I don’t know / don’t pay attention to brand",
            "Which spaghetti brands have you purchased in the last 3 months? - Other",
        ]
    ]

# Summing up the counts for each brand
brand_counts = spaghetti_brands_columns.sum()

# Check if there are any columns to plot
if not brand_counts.empty and brand_counts.max() > 0:
    # Sort the DataFrame by the sum of each row (brand count) in descending order
    sorted_brand_counts = brand_counts.sort_values(ascending=False)

    # Plotting the bar graph using the sorted DataFrame
    plt.figure(figsize=(10, 6))
    sorted_brand_counts.plot(kind="bar", color="skyblue")
    plt.title("Spaghetti Brands Purchased in the Last 3 Months (Descending Order)")
    plt.xlabel("Spaghetti Brands")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.show()
else:
    print("No spaghetti brand columns found in the DataFrame.")


retailer_columns = df.filter(like="Which retailers do you buy spaghetti pasta from?")

# Exclude the "Other Value" column
retailer_columns = retailer_columns.drop(
    columns="Which retailers do you buy spaghetti pasta from? - Other Value",
    errors="ignore",
)

# Summing up the counts for each retailer
retailer_counts = retailer_columns.sum()

# Check if there are any columns to plot
if not retailer_counts.empty and retailer_counts.max() > 0:
    # Sort the DataFrame by the sum of each row (retailer count) in descending order
    sorted_retailer_counts = retailer_counts.sort_values(ascending=False)

    # Plotting the bar graph using the sorted DataFrame
    plt.figure(figsize=(12, 6))
    sorted_retailer_counts.plot(kind="bar", color="orange")
    plt.title("Retailers from Which Spaghetti Pasta is Purchased (Descending Order)")
    plt.xlabel("Retailers")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.show()
else:
    print("No retailer columns found in the DataFrame.")


# QUESTION 2 AND QUESTION 3  CHECKING CORRELATION AND CONTRADICTION

concept1_data = pd.read_excel("raw.xlsx", sheet_name="Concept 1")
concept2_data = pd.read_excel("raw.xlsx", sheet_name="Concept 2")
concept3_data = pd.read_excel("raw.xlsx", sheet_name="Concept 3")


def analyze_concept(concept_data, concept_number):
    print(f"\nAnalysis for Concept {concept_number}:\n")

    # Check correlation between numerical ratings and open-ended responses for each specific user
    unique_users = concept_data["Unique Identifier"].unique()
    for user_id in unique_users:
        user_data = concept_data[concept_data["Unique Identifier"] == user_id]
        print(f"\nAnalysis for User {user_id}:")

        # Your analysis code for the specific user
        overall_appeal = user_data[
            "How appealing do you find this spaghetti OVERALL? - value"
        ].values[0]
        purchase_likelihood = user_data[
            "How likely, or unlikely, would you be to purchase this Spaghetti product in the future? - value"
        ].values[0]
        reasons_to_buy = user_data[
            "Why would you buy this spaghetti in your own words?"
        ].values[0]
        reasons_not_to_buy = user_data[
            "Why would you NOT buy this spaghetti in your own words?"
        ].values[0]
        word_association = user_data[
            "What’s one word you associate with Spaghetti?"
        ].values[0]

        print(f"Overall Appeal: {overall_appeal}")
        print(f"Purchase Likelihood: {purchase_likelihood}")
        print(f"Reasons to Buy: {reasons_to_buy}")
        print(f"Reasons Not to Buy: {reasons_not_to_buy}")
        print(f"Word Association: {word_association}")

        premium_association = user_data[
            "Which of these words would you associate with this spaghetti? - Premium"
        ].values[0]
        cheap_association = user_data[
            "Which of these words would you associate with this spaghetti? - Cheap"
        ].values[0]
        artisan_association = user_data[
            "Which of these words would you associate with this spaghetti? - Artisan"
        ].values[0]
        convenient_association = user_data[
            "Which of these words would you associate with this spaghetti? - Convenient"
        ].values[0]
        flavorful_association = user_data[
            "Which of these words would you associate with this spaghetti? - Flavorful"
        ].values[0]
        nutritious_association = user_data[
            "Which of these words would you associate with this spaghetti? - Nutritious"
        ].values[0]
        authentic_association = user_data[
            "Which of these words would you associate with this spaghetti? - Authentic"
        ].values[0]
        delicious_association = user_data[
            "Which of these words would you associate with this spaghetti? - Delicious"
        ].values[0]
        low_quality_association = user_data[
            "Which of these words would you associate with this spaghetti? - Low-quality"
        ].values[0]

        if premium_association != 0:
            print(f"Premium Association: {premium_association}")
        if cheap_association != 0:
            print(f"Cheap Association: {cheap_association}")
        if artisan_association != 0:
            print(f"Artisan Association: {artisan_association}")
        if convenient_association != 0:
            print(f"Convenient Association: {convenient_association}")
        if flavorful_association != 0:
            print(f"Flavorful Association: {flavorful_association}")
        if nutritious_association != 0:
            print(f"Nutritious Association: {nutritious_association}")
        if authentic_association != 0:
            print(f"Authentic Association: {authentic_association}")
        if delicious_association != 0:
            print(f"Delicious Association: {delicious_association}")
        if low_quality_association != 0:
            print(f"Low-quality Association: {low_quality_association}")

        # Additional columns related to willingness to pay, frequency of purchase, salt preference, package preference, and purchase frequency
        willingness_to_pay = user_data[
            "How much would you be willing to pay for this spaghetti? - label"
        ].values[0]
        purchase_frequency = user_data[
            "How often would you buy this spaghetti over your current go-to? - label"
        ].values[0]
        salt_preference = user_data[
            "How much, if any, salt do you put in your water while cooking pasta? - value"
        ].values[0]
        package_preference = user_data[
            "Spaghetti is sold in a variety of packages, which do you most prefer? - value"
        ].values[0]
        purchase_frequency_spaghetti = user_data[
            "How often are you buying the following pasta? - Spaghetti"
        ].values[0]

        print(f"Willingness to Pay: {willingness_to_pay}")
        print(f"Purchase Frequency: {purchase_frequency}")
        print(f"Salt Preference: {salt_preference}")
        print(f"Package Preference: {package_preference}")
        print(f"Purchase Frequency for Spaghetti: {purchase_frequency_spaghetti}")

        # Compare open-ended answer with rating question
        if overall_appeal > 3 and reasons_to_buy.lower() not in [
            "positive",
            "good",
            "satisfied",
        ]:
            print("Warning: Open-ended response does not match positive rating.")

        if overall_appeal < 3 and reasons_not_to_buy.lower() not in [
            "negative",
            "bad",
            "unsatisfied",
        ]:
            print("Warning: Open-ended response does not match negative rating.")

        # Check for contradictory responses
        if overall_appeal > 3 and purchase_likelihood < 3:
            print("Contradictory responses: High appeal but low purchase likelihood.")
        elif overall_appeal < 3 and purchase_likelihood > 3:
            print("Contradictory responses: Low appeal but high purchase likelihood.")


# Perform analysis for each concept turn by turn
analyze_concept(concept1_data, 1)

# ----------------------------------Uncomment the following lines to analyze the other concepts-----------------------------


# analyze_concept(concept2_data, 2)


# analyze_concept(concept3_data, 3)
