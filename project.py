
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import sys

def setup_environment():
    """Ensure all required packages are installed"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install pandas numpy matplotlib scikit-learn seaborn")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    print("✅ Directories created")

def generate_sample_data():
    """Generate sample customer data if no file exists"""
    try:
        df = pd.read_csv('data/mall_customers.csv')
        print("✅ Loaded existing customer data")
        return df
    except FileNotFoundError:
        print("📊 Creating sample customer data...")
        np.random.seed(42)
        
        # Create realistic customer data
        n_customers = 300
        data = {
            'CustomerID': range(1, n_customers + 1),
            'Age': np.random.normal(45, 15, n_customers).astype(int),
            'Annual Income (k$)': np.random.normal(60, 20, n_customers).astype(int),
            'Spending Score (1-100)': np.random.normal(50, 25, n_customers).astype(int)
        }
        
        # Ensure values are within reasonable ranges
        data['Age'] = np.clip(data['Age'], 18, 70)
        data['Annual Income (k$)'] = np.clip(data['Annual Income (k$)'], 15, 140)
        data['Spending Score (1-100)'] = np.clip(data['Spending Score (1-100)'], 1, 100)
        
        df = pd.DataFrame(data)
        df.to_csv('data/mall_customers.csv', index=False)
        print("✅ Sample data created and saved to data/mall_customers.csv")
        return df

def explore_data(df):
    """Explore and understand the data"""
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nDescriptive statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age distribution
    axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # Income distribution
    axes[0, 1].hist(df['Annual Income (k$)'], bins=20, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Annual Income Distribution')
    axes[0, 1].set_xlabel('Annual Income (k$)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Spending score distribution
    axes[1, 0].hist(df['Spending Score (1-100)'], bins=20, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Spending Score Distribution')
    axes[1, 0].set_xlabel('Spending Score (1-100)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Income vs Spending
    axes[1, 1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], alpha=0.6)
    axes[1, 1].set_title('Income vs Spending Score')
    axes[1, 1].set_xlabel('Annual Income (k$)')
    axes[1, 1].set_ylabel('Spending Score (1-100)')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/data_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_clustering(df):
    """Perform K-Means clustering analysis"""
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS")
    print("="*50)
    
    # Select features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters using elbow method
    print("Finding optimal number of clusters...")
    wcss = []
    silhouette_scores = []
    cluster_range = range(2, 11)
    
    for i in cluster_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
        if i > 1:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"Clusters: {i}, WCSS: {kmeans.inertia_:.2f}, Silhouette: {score:.3f}")
    
    # Plot elbow method and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow method
    ax1.plot(cluster_range, wcss, marker='o', linestyle='--', color='blue')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='red')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Choose optimal number of clusters (typically 5 for this type of data)
    optimal_clusters = 5
    print(f"\nUsing {optimal_clusters} clusters for segmentation")
    
    # Apply K-Means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['Cluster'] = clusters
    
    return df, kmeans, scaler

def visualize_results(df, kmeans, scaler):
    """Visualize and analyze the clustering results"""
    print("\n" + "="*50)
    print("CLUSTER VISUALIZATION")
    print("="*50)
    
    # Get cluster centers in original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create a beautiful cluster visualization
    plt.figure(figsize=(12, 8))
    
    # Define colors for each cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#9B59B6']
    
    # Plot each cluster with different color
    for cluster_num in range(5):
        cluster_data = df[df['Cluster'] == cluster_num]
        plt.scatter(cluster_data['Annual Income (k$)'], 
                   cluster_data['Spending Score (1-100)'], 
                   c=colors[cluster_num], 
                   s=100, 
                   alpha=0.7,
                   label=f'Cluster {cluster_num}')
    
    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=1.0, marker='X', 
                label='Cluster Centers', edgecolors='white', linewidth=2)
    
    plt.xlabel('Annual Income (k$)', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.title('Customer Segments - K-Means Clustering', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.style.use('default')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('results/visualizations/customer_segments.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze each cluster
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS")
    print("="*50)
    
    cluster_analysis = df.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
        'Age': ['mean', 'std'],
        'CustomerID': 'count'
    }).round(2)
    
    cluster_analysis.columns = ['Income_Mean', 'Income_Std', 'Income_Min', 'Income_Max',
                               'Spending_Mean', 'Spending_Std', 'Spending_Min', 'Spending_Max',
                               'Age_Mean', 'Age_Std', 'Customer_Count']
    
    print(cluster_analysis)
    
    # Interpret clusters
    print("\n" + "="*50)
    print("CLUSTER INTERPRETATIONS")
    print("="*50)
    
    cluster_profiles = {
        0: "💰 High Income, Low Spending - Conservative Customers",
        1: "⚖️ Medium Income, Medium Spending - Standard Customers", 
        2: "🎯 Low Income, High Spending - Carefree Spenders",
        3: "💡 Low Income, Low Spending - Budget-Conscious",
        4: "⭐ High Income, High Spending - Premium Customers"
    }
    
    for cluster_num, profile in cluster_profiles.items():
        count = len(df[df['Cluster'] == cluster_num])
        print(f"Cluster {cluster_num}: {profile} ({count} customers)")
    
    return cluster_analysis

def save_results(df, cluster_analysis):
    """Save all results to files"""
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    # Save clustered data
    df.to_csv('results/customer_segments.csv', index=False)
    print("✅ Saved clustered data to results/customer_segments.csv")
    
    # Save cluster analysis
    cluster_analysis.to_csv('results/cluster_analysis.csv')
    print("✅ Saved cluster analysis to results/cluster_analysis.csv")
    
    # Save cluster centers
    centers_df = pd.DataFrame({
        'Cluster': range(5),
        'Avg_Income': cluster_analysis['Income_Mean'],
        'Avg_Spending': cluster_analysis['Spending_Mean'],
        'Customer_Count': cluster_analysis['Customer_Count']
    })
    centers_df.to_csv('results/cluster_summary.csv', index=False)
    print("✅ Saved cluster summary to results/cluster_summary.csv")

def main():
    """Main function to run the complete analysis"""
    print("🎯 CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
    print("="*60)
    
    try:
        # Step 1: Setup environment
        setup_environment()
        create_directories()
        
        # Step 2: Load or create data
        df = generate_sample_data()
        
        # Step 3: Explore data
        explore_data(df)
        
        # Step 4: Perform clustering
        df, kmeans, scaler = perform_clustering(df)
        
        # Step 5: Visualize results
        cluster_analysis = visualize_results(df, kmeans, scaler)
        
        # Step 6: Save results
        save_results(df, cluster_analysis)
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the 'results' folder for:")
        print("   - customer_segments.csv (data with cluster labels)")
        print("   - cluster_analysis.csv (detailed cluster statistics)")
        print("   - cluster_summary.csv (cluster overview)")
        print("   - visualizations/ folder (all charts and graphs)")
        print("\n📊 You can now use these segments for targeted marketing!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check your setup and try again.")
        import traceback
        traceback.print_exc()

# Run the main function
if __name__ == "__main__":
    main()

