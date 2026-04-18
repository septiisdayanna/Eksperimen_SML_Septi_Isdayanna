import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import dump

# ========================================================
# CUSTOM HELPER FUNCTIONS
# ========================================================

def split_blood_pressure(df, col='Blood Pressure'):
    """Split kolom Blood Pressure menjadi Systolic & Diastolic"""
    if col in df.columns:
        bp_split = df[col].str.split('/', expand=True)
        df['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
        df.drop(col, axis=1, inplace=True)
    return df


def clean_structural_data(df, target_column):
    """Pembersihan awal sebelum masuk pipeline"""
    df = df.copy()

    print(f"Data awal: {df.shape}")

    # Hapus duplikat
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplikat dihapus: {before - after}")

    # Drop ID
    if 'Person ID' in df.columns:
        df.drop('Person ID', axis=1, inplace=True)

    # Isi missing target dengan 'Normal'
    if df[target_column].isnull().sum() > 0:
        df[target_column] = df[target_column].fillna('Normal')

    # Normalisasi kategori BMI
    if 'BMI Category' in df.columns:
        df['BMI Category'] = df['BMI Category'].replace({
            'Normal Weight': 'Normal'
        })

    # Split Blood Pressure
    df = split_blood_pressure(df)

    return df


def validate_columns(df, required_cols):
    """Validasi kolom dataset"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing}")


# ========================================================
# MAIN PIPELINE FUNCTION
# ========================================================

def preprocess_data_pipeline(
    data,
    target_column='Sleep Disorder',
    save_path='preprocessor_pipeline.joblib',
    header_raw_path='raw_header.csv',
    header_processed_path='processed_header.csv',
    test_size=0.2,
    random_state=42
):
    print("Mulai preprocessing pipeline...")

    # =========================
    # 1. CLEANING
    # =========================
    df = clean_structural_data(data, target_column)

    # =========================
    # 2. SIMPAN RAW HEADER
    # =========================
    raw_columns = df.drop(columns=[target_column]).columns
    df_header = pd.DataFrame(columns=raw_columns)
    df_header.to_csv(header_raw_path, index=False)

    print(f"Raw header disimpan di: {header_raw_path}")

    # =========================
    # 3. SPLIT FEATURE & TARGET
    # =========================
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Jumlah data setelah cleaning: {df.shape}")
    print(f"Distribusi target:\n{y.value_counts()}\n")

    # =========================
    # 4. AUTO DETECT FEATURES
    # =========================
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    ordinal_features = ['BMI Category'] if 'BMI Category' in X.columns else []
    onehot_features = [col for col in categorical_features if col not in ordinal_features]

    # Validasi kolom
    validate_columns(df, numeric_features + ordinal_features + onehot_features + [target_column])

    # =========================
    # 5. TRAIN TEST SPLIT
    # =========================
    stratify_option = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_option # memastikan proporsi kelas dalam data latih (Train) dan data uji (Test) sama persis dengan proporsi di dataset aslinya.
    )

    # =========================
    # 6. BUILD TRANSFORMERS
    # =========================

    # Numeric
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Ordinal (BMI)
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[['Normal', 'Overweight', 'Obese']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # Nominal
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # =========================
    # 7. COLUMN TRANSFORMER
    # =========================
    transformers = []

    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))

    if ordinal_features:
        transformers.append(('ord', ordinal_transformer, ordinal_features))

    if onehot_features:
        transformers.append(('cat', categorical_transformer, onehot_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # =========================
    # 8. FIT & TRANSFORM
    # =========================
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # =========================
    # 9. FEATURE NAMES
    # =========================
    feature_names = preprocessor.get_feature_names_out()

    # =========================
    # 10. SIMPAN PROCESSED HEADER
    # =========================
    df_processed_header = pd.DataFrame(columns=feature_names)
    df_processed_header.to_csv(header_processed_path, index=False)

    print(f"Processed header disimpan di: {header_processed_path}")

    # =========================
    # 11. SAVE PIPELINE
    # =========================
    dump(preprocessor, save_path)

    print(f"Pipeline berhasil disimpan di: {save_path}")
    print(f"Total fitur setelah transformasi: {len(feature_names)}")

    # =========================
    # 12. RETURN
    # =========================
    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }

# ========================================================
# BLOK EKSEKUSI (Dijalankan oleh GitHub Actions)
# ========================================================
if __name__ == "__main__":
    # 1. BUAT FOLDER OTOMATIS SEBELUM PROSES DIMULAI
    output_folder = 'Preprocessing/Sleep_health_and_lifestyle_dataset_preprocessing'
    os.makedirs(output_folder, exist_ok=True)
    print(f"Folder output siap di: {output_folder}")

    # 2. BACA DATA MENTAH
    raw_data_path = 'data_raw/Sleep_health_and_lifestyle_dataset_raw.csv' 
    df_raw = pd.read_csv(raw_data_path)

    # 3. JALANKAN PIPELINE
    # Kita arahkan semua output file agar masuk ke dalam folder yang baru dibuat
    hasil = preprocess_data_pipeline(
        data=df_raw,
        target_column='Sleep Disorder',
        save_path=f'{output_folder}/preprocessor_pipeline.joblib',
        header_raw_path=f'{output_folder}/raw_header.csv',
        header_processed_path=f'{output_folder}/processed_header.csv'
    )

    # 4. SIMPAN DATASET HASIL PREPROCESSING KE DALAM FOLDER
    print("Menyimpan hasil dataset split ke CSV...")
    
    # Ambil data dari dictionary hasil fungsi
    X_train_df = pd.DataFrame(hasil["X_train"], columns=hasil["feature_names"])
    X_test_df = pd.DataFrame(hasil["X_test"], columns=hasil["feature_names"])
    
    X_train_df.to_csv(f'{output_folder}/X_train_clean.csv', index=False)
    X_test_df.to_csv(f'{output_folder}/X_test_clean.csv', index=False)
    hasil["y_train"].to_csv(f'{output_folder}/y_train.csv', index=False)
    hasil["y_test"].to_csv(f'{output_folder}/y_test.csv', index=False)

    print("Seluruh proses berhasil! Data siap di-download via GitHub Actions.")
