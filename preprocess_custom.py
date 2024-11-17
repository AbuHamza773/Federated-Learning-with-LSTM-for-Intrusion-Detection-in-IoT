
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_dataset(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handle missing values and clean the dataset."""
    # Replace missing values with the median for numerical columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Drop rows with excessive missing values
    df.dropna(thresh=len(df.columns) - 2, inplace=True)
    return df

def normalize_data(df, columns, method='minmax'):
    """Normalize specified columns in the dataset."""
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Choose 'minmax' or 'standard'.")
    
    df[columns] = scaler.fit_transform(df[columns])
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def balance_data(X, y):
    """Handle class imbalance using SMOTE."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

if __name__ == "__main__":
    # WSN-DS dataset example (from the paper)
    file_path = "data/wsn-ds.csv"
    target_column = "attack_type"
    
    # Load dataset
    df = load_dataset(file_path)
    print(f"Dataset loaded with shape: {df.shape}")

    # Clean data
    df = clean_data(df)
    print(f"Dataset after cleaning: {df.shape}")

    # Normalize data
    numerical_columns = ['RSSI', 'energy_consumed', 'distance_to_cluster_head']
    df = normalize_data(df, numerical_columns)
    print(f"Dataset after normalization:
{df.head()}")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Balance data
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    print(f"Balanced training set: {X_train_balanced.shape}, {y_train_balanced.shape}")
