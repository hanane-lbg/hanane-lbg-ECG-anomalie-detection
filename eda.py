import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path_abnormal = ''
path_normal = ''
abnormal_df = pd.read_csv(path_abnormal)
normal_df = pd.read_csv(path_normal)

abnormal_df['class'] = 1
normal_df['class'] = 0

# --- Function to summarize dataset ---
def ecg_dataset_summary(df, name="Dataset"):
    print(f"\n=== {name} Overview ===")
    print(f"Shape: {df.shape}\n")
    
    print("=== Statistical Summary ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    print(df.isna().sum())
    
    return {
        "shape": df.shape,
        "describe": df.describe(),
        "missing_values": df.isna().sum()
    }

# --- Summarize individual datasets ---
ecg_dataset_summary(normal_df, name="Normal ECG")
ecg_dataset_summary(abnormal_df, name="Abnormal ECG")


print('============ Check identical column names =====================')

def have_identical_columns(df1, df2):
    return df1.columns.equals(df2.columns)

CLASS_NAMES = ["Normal", "Anomaly"]
#Creat a copy for each dataset
normal_df_copy = normal_df.copy()
anomaly_df_copy = abnormal_df.copy()
#rename the columns to be identical
normal_df_copy = normal_df_copy.set_axis(range(1, 189), axis=1)
anomaly_df_copy = anomaly_df_copy.set_axis(range(1, 189), axis=1)
normal_df_copy = normal_df_copy.assign(target = CLASS_NAMES[0])
anomaly_df_copy = anomaly_df_copy.assign(target = CLASS_NAMES[1])




df = pd.concat([normal_df, abnormal_df], ignore_index=True)
print(df.isna().sum().sum())

# --- Plot class distribution ---
def class_distribution(df):
    counts = df['target'].value_counts()
    df['class'].value_counts().plot(kind='bar', color=['green','red'])
    plt.title("ECG Class Distribution")
    plt.xlabel("Class (0=Normal, 1=Abnormal)")
    plt.ylabel("Count")
    plt.show()
    return counts



