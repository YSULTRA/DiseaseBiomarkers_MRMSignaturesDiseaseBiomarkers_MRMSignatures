import pandas as pd
import json
from pandas import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import networkx as nx
from scipy.stats import chi2_contingency
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings("ignore")

# Set plot style for publication-quality visuals
plt.style.use("seaborn-v0_8")  # Updated style name
sns.set_palette("deep")

# Create output directory for saving plots
output_dir = "eda_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading and Preprocessing ---
def load_json(file_path):
    """Load JSON file and return normalized DataFrame."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # Normalize nested Biomarkers and meta fields
        df = json_normalize(
            data,
            record_path=["Biomarkers"],
            meta=["_id", "DiseaseName", "SourceMaterial", "Organism", "TechniqueUsed", "PMID"],
            record_prefix="Biomarker.",
            errors="ignore"
        )
        
        # Clean data
        # Handle list-type DiseaseName and remove duplicates within each record
        df["DiseaseName"] = df["DiseaseName"].apply(
            lambda x: ", ".join(set(x)) if isinstance(x, list) else x
        )
        # Convert lists in TechniqueUsed to strings
        df["TechniqueUsed"] = df["TechniqueUsed"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        # Remove leading/trailing whitespace
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].str.strip() if df[col].dtype == "object" else df[col]
        
        return df
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

# Load JSON file (replace with your file path)
file_path = "MasterDB.MasterCollection23APR (1).json"
df = load_json(file_path)

if df is None:
    print("Exiting due to data loading error.")
    exit()

# Basic dataset info
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Unique Values ---")
for col in ["Biomarker.ProteinName", "DiseaseName", "SourceMaterial", "Organism"]:
    print(f"{col}: {df[col].nunique()} unique values")

# Save missing value heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Dataset")
plt.savefig(os.path.join(output_dir, "missing_values_heatmap.png"))
plt.close()

# --- 2. Univariate Analysis ---
def univariate_analysis():
    """Perform univariate analysis on key columns."""
    print("\n--- Univariate Analysis ---")
    
    # a. Protein Names
    protein_counts = df["Biomarker.ProteinName"].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=protein_counts.values, y=protein_counts.index)
    plt.title("Top 10 Most Frequent Proteins")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "top_proteins_bar.png"))
    plt.close()
    
    # Missing UniProtID
    missing_uniprot = df["Biomarker.UniProtID"].isnull().value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(missing_uniprot, labels=["With UniProtID", "Missing UniProtID"], autopct="%1.1f%%")
    plt.title("UniProtID Availability")
    plt.savefig(os.path.join(output_dir, "uniprot_missing_pie.png"))
    plt.close()
    
    # b. Diseases
    disease_counts = df["DiseaseName"].str.split(", ").explode().value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=disease_counts.values, y=disease_counts.index)
    plt.title("Top 10 Diseases")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "top_diseases_bar.png"))
    plt.close()
    
    # Disease Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(df["DiseaseName"].dropna().astype(str))
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Disease Name Word Cloud")
    plt.savefig(os.path.join(output_dir, "disease_wordcloud.png"))
    plt.close()
    
    # c. Source Material
    source_counts = df["SourceMaterial"].str.split(", ").explode().value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=source_counts.values, y=source_counts.index)
    plt.title("Top 10 Source Materials")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "top_source_materials_bar.png"))
    plt.close()
    
    # d. Techniques Used
    technique_counts = df["TechniqueUsed"].str.split(", ").explode().value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=technique_counts.values, y=technique_counts.index)
    plt.title("Top 10 Techniques Used")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "top_techniques_bar.png"))
    plt.close()
    
    # e. Organism
    organism_counts = df["Organism"].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=organism_counts.values, y=organism_counts.index)
    plt.title("Organism Distribution")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "organism_bar.png"))
    plt.close()

univariate_analysis()

# --- 3. Bivariate Analysis ---
def bivariate_analysis():
    """Perform bivariate analysis to explore relationships."""
    print("\n--- Bivariate Analysis ---")
    
    # a. Protein-Disease Associations
    protein_disease = df.groupby(["Biomarker.ProteinName", "DiseaseName"]).size().unstack(fill_value=0)
    protein_disease_top = protein_disease.loc[protein_disease.sum(axis=1).nlargest(10).index,
                                            protein_disease.sum().nlargest(10).index]
    plt.figure(figsize=(12, 10))
    sns.heatmap(protein_disease_top, cmap="Blues", annot=True, fmt="d")
    plt.title("Top Protein-Disease Associations")
    plt.savefig(os.path.join(output_dir, "protein_disease_heatmap.png"))
    plt.close()
    
    # b. Protein-Source Material
    protein_source = df.groupby(["Biomarker.ProteinName", "SourceMaterial"]).size().unstack(fill_value=0)
    protein_source_top = protein_source.loc[protein_source.sum(axis=1).nlargest(10).index,
                                          protein_source.sum().nlargest(10).index]
    plt.figure(figsize=(12, 10))
    sns.heatmap(protein_source_top, cmap="Greens", annot=True, fmt="d")
    plt.title("Top Protein-Source Material Associations")
    plt.savefig(os.path.join(output_dir, "protein_source_heatmap.png"))
    plt.close()
    
    # c. Disease-Technique
    disease_technique = df.explode("TechniqueUsed").groupby(["DiseaseName", "TechniqueUsed"]).size().unstack(fill_value=0)
    disease_technique_top = disease_technique.loc[disease_technique.sum(axis=1).nlargest(10).index,
                                                disease_technique.sum().nlargest(10).index]
    plt.figure(figsize=(12, 10))
    sns.heatmap(disease_technique_top, cmap="Reds", annot=True, fmt="d")
    plt.title("Top Disease-Technique Associations")
    plt.savefig(os.path.join(output_dir, "disease_technique_heatmap.png"))
    plt.close()

bivariate_analysis()

# --- 4. Text Analysis ---
def text_analysis():
    """Perform text analysis on protein and disease names."""
    print("\n--- Text Analysis ---")
    
    # a. Protein Name Keywords
    def extract_keywords(text):
        words = re.findall(r'\b\w+\b', str(text).lower())
        return [w for w in words if w not in ["protein", "human", "type"]]
    
    protein_keywords = Counter()
    df["Biomarker.ProteinName"].dropna().apply(lambda x: protein_keywords.update(extract_keywords(x)))
    top_keywords = dict(protein_keywords.most_common(10))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(top_keywords.values()), y=list(top_keywords.keys()))
    plt.title("Top 10 Protein Name Keywords")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "protein_keywords_bar.png"))
    plt.close()
    
    # b. Disease Clustering
    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    disease_matrix = tfidf.fit_transform(df["DiseaseName"].dropna().astype(str))
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(disease_matrix)
    
    df["DiseaseCluster"] = pd.Series(clusters, index=df.index[:len(clusters)])
    print("\nDisease Clusters:")
    for cluster in range(5):
        cluster_diseases = df[df["DiseaseCluster"] == cluster]["DiseaseName"].unique()[:5]
        print(f"Cluster {cluster}: {', '.join(cluster_diseases)}")

text_analysis()

# --- 5. Network Analysis ---
def network_analysis():
    """Perform network analysis for protein-disease relationships."""
    print("\n--- Network Analysis ---")
    
    G = nx.Graph()
    for _, row in df.iterrows():
        protein = row["Biomarker.ProteinName"]
        diseases = str(row["DiseaseName"]).split(", ")
        for disease in diseases:
            G.add_edge(protein, disease)
    
    # Filter to top nodes for visualization
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:20]
    top_node_names = [node for node, _ in top_nodes]
    G_sub = G.subgraph(top_node_names)
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G_sub)
    nx.draw(G_sub, pos, with_labels=True, node_size=500, font_size=8, node_color="lightblue")
    plt.title("Protein-Disease Network (Top 20 Nodes)")
    plt.savefig(os.path.join(output_dir, "protein_disease_network.png"))
    plt.close()

network_analysis()

# --- 6. Statistical Analysis ---
def statistical_analysis():
    """Perform statistical tests for associations."""
    print("\n--- Statistical Analysis ---")
    
    # Chi-square test for Disease-Source Material association
    contingency_table = pd.crosstab(
        df["DiseaseName"].str.split(", ").explode(),
        df["SourceMaterial"].str.split(", ").explode()
    )
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test (Disease-Source Material): p-value = {p:.4f}")
    
    # Correlation between protein frequency and technique usage
    protein_freq = df["Biomarker.ProteinName"].value_counts()
    technique_freq = df["TechniqueUsed"].str.split(", ").explode().value_counts()
    print("\nTop 5 Proteins and Techniques:")
    print(f"Proteins: {protein_freq.head(5).to_dict()}")
    print(f"Techniques: {technique_freq.head(5).to_dict()}")

statistical_analysis()

# --- 7. CRP-Specific Analysis ---
def crp_analysis():
    """Analyze C-reactive protein (CRP) specific data."""
    print("\n--- C-reactive Protein (CRP) Analysis ---")
    
    crp_df = df[df["Biomarker.ProteinName"].str.contains("C-reactive protein", case=False, na=False)]
    print(f"Number of CRP records: {len(crp_df)}")
    
    # Diseases associated with CRP
    crp_diseases = crp_df["DiseaseName"].str.split(", ").explode().value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=crp_diseases.values, y=crp_diseases.index)
    plt.title("Top 10 Diseases Associated with C-reactive Protein")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "crp_diseases_bar.png"))
    plt.close()
    
    # Source materials for CRP
    crp_sources = crp_df["SourceMaterial"].str.split(", ").explode().value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(crp_sources, labels=crp_sources.index, autopct="%1.1f%%")
    plt.title("Source Materials for C-reactive Protein")
    plt.savefig(os.path.join(output_dir, "crp_sources_pie.png"))
    plt.close()
    
    # Techniques for CRP
    crp_techniques = crp_df["TechniqueUsed"].str.split(", ").explode().value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=crp_techniques.values, y=crp_techniques.index)
    plt.title("Top 10 Techniques for C-reactive Protein")
    plt.xlabel("Count")
    plt.savefig(os.path.join(output_dir, "crp_techniques_bar.png"))
    plt.close()

crp_analysis()

# --- 8. Top 10 Diseases by Biomarkers ---
def plot_top_disease_biomarkers():
    """Create a graph of top 10 diseases and their associated biomarkers."""
    # Explode DiseaseName to handle multiple diseases per record
    df_exploded = df[["DiseaseName", "Biomarker.ProteinName"]].copy()
    df_exploded["DiseaseName"] = df_exploded["DiseaseName"].str.split(", ")
    df_exploded = df_exploded.explode("DiseaseName")
    
    # Get the top 10 diseases by frequency
    top_diseases = df_exploded["DiseaseName"].value_counts().head(10).index
    
    # Filter for top 10 diseases
    df_top = df_exploded[df_exploded["DiseaseName"].isin(top_diseases)]
    
    # Group by DiseaseName and count unique biomarkers
    disease_biomarker_counts = df_top.groupby("DiseaseName")["Biomarker.ProteinName"].nunique().sort_values(ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=disease_biomarker_counts.index, y=disease_biomarker_counts.values)
    plt.title("Top 10 Diseases by Number of Unique Biomarkers")
    plt.xlabel("Disease Name")
    plt.ylabel("Number of Unique Biomarkers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_diseases_biomarkers_bar.png"))
    plt.close()

plot_top_disease_biomarkers()

# --- 9. Create CSV with Disease and Associated Biomarkers ---
def create_disease_biomarker_csv():
    """Create a CSV with Disease and Associated Biomarkers."""
    # Explode DiseaseName to handle multiple diseases per record
    df_exploded = df[["DiseaseName", "Biomarker.ProteinName"]].copy()
    df_exploded["DiseaseName"] = df_exploded["DiseaseName"].str.split(", ")
    df_exploded = df_exploded.explode("DiseaseName")
    
    # Rename columns for clarity
    df_exploded = df_exploded.rename(columns={
        "DiseaseName": "Disease",
        "Biomarker.ProteinName": "Associated_Biomarker"
    })
    
    # Drop any rows with missing values
    df_exploded = df_exploded.dropna()
    
    # Save to CSV
    output_csv_path = os.path.join(output_dir, "disease_biomarker_mapping.csv")
    df_exploded.to_csv(output_csv_path, index=False)
    print(f"Disease-Biomarker mapping saved to {output_csv_path}")

create_disease_biomarker_csv()

# --- 10. Summary Report ---
def generate_summary():
    """Generate a summary report of findings."""
    print("\n--- EDA Summary ---")
    print(f"Total Records: {len(df)}")
    print(f"Unique Proteins: {df['Biomarker.ProteinName'].nunique()}")
    print(f"Unique Diseases: {df['DiseaseName'].str.split(', ').explode().nunique()}")
    print(f"Unique Source Materials: {df['SourceMaterial'].str.split(', ').explode().nunique()}")
    print(f"Unique Techniques: {df['TechniqueUsed'].str.split(', ').explode().nunique()}")
    print(f"Missing UniProtID: {df['Biomarker.UniProtID'].isnull().sum()} ({df['Biomarker.UniProtID'].isnull().mean()*100:.1f}%)")
    print("\nKey Insights:")
    print("- C-reactive protein is a frequently studied biomarker, associated with multiple diseases.")
    print("- Serum and plasma are the most common source materials.")
    print("- ELISA and immunohistochemistry are widely used techniques.")
    print("- Network analysis reveals central proteins linking multiple diseases.")
    print(f"\nAll plots saved in: {output_dir}")

generate_summary()

# Save dataset summary to CSV
df.to_csv(os.path.join(output_dir, "processed_dataset.csv"), index=False)
print(f"Processed dataset saved to {os.path.join(output_dir, 'processed_dataset.csv')}")