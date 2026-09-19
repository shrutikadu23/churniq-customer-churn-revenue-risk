import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from urllib.parse import quote_plus


# ============================================================
# ChurnIQ — Load Training Data into PostgreSQL
# ============================================================

# ---------- Configuration ----------
CSV_PATH = Path("data/raw/train.csv")

DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churniq"
DB_USER = "postgres"

EXPECTED_ROWS = 69_999
EXPECTED_COLUMNS = 172


# ============================================================
# 1. Validate Source File
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the dataset at: {CSV_PATH}\n"
        "Make sure you are running this script from the ChurnIQ project root."
    )

print("=" * 60)
print("ChurnIQ — PostgreSQL Data Loader")
print("=" * 60)

print("\n[1/5] Loading train.csv...")

df = pd.read_csv(CSV_PATH)

print(f"✓ CSV loaded successfully")
print(f"  Rows:    {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]:,}")


# ============================================================
# 2. Validate Dataset Structure
# ============================================================

print("\n[2/5] Validating dataset structure...")

if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
    raise ValueError(
        f"Unexpected dataset shape: {df.shape}. "
        f"Expected ({EXPECTED_ROWS}, {EXPECTED_COLUMNS})."
    )

required_columns = {"id", "churn_probability"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {sorted(missing_columns)}"
    )

if df.columns.duplicated().any():
    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    raise ValueError(
        f"Duplicate column names found: {duplicated_columns}"
    )

print("✓ Dataset dimensions verified")
print("✓ Required columns verified")
print("✓ Column names are unique")


# ============================================================
# 3. Validate Identifier and Target
# ============================================================

print("\n[3/5] Validating identifiers and target...")

duplicate_ids = df["id"].duplicated().sum()

if duplicate_ids > 0:
    raise ValueError(
        f"Found {duplicate_ids:,} duplicate customer IDs."
    )

if df["id"].isna().any():
    raise ValueError("Customer ID contains missing values.")

target_values = set(df["churn_probability"].dropna().unique())

if not target_values.issubset({0, 1}):
    raise ValueError(
        f"Unexpected target values found: {sorted(target_values)}"
    )

if df["churn_probability"].isna().any():
    raise ValueError("Target variable contains missing values.")

churn_count = int((df["churn_probability"] == 1).sum())
retained_count = int((df["churn_probability"] == 0).sum())

print("✓ Customer IDs are unique")
print("✓ No missing customer IDs")
print("✓ Target contains only 0 and 1")
print("✓ No missing target values")

print(f"  Retained customers: {retained_count:,}")
print(f"  Churned customers:  {churn_count:,}")
print(f"  Churn rate:         {df['churn_probability'].mean():.2%}")


# ============================================================
# 4. Connect to PostgreSQL
# ============================================================

print("\n[4/5] Connecting to PostgreSQL...")

password = input("Enter PostgreSQL password: ")

encoded_password = quote_plus(password)

connection_url = (
    f"postgresql+psycopg2://{DB_USER}:{encoded_password}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_url)

try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("✓ PostgreSQL connection successful")

except Exception as error:
    raise ConnectionError(
        "Could not connect to PostgreSQL. "
        "Check that PostgreSQL is running and your credentials are correct."
    ) from error


# ============================================================
# 5. Load Data into Raw Staging Layer
# ============================================================

print("\n[5/5] Loading data into staging.train_raw...")
print("This may take a little time...")

with engine.begin() as connection:

    # Prevent accidental overwrite of an existing staging table.
    table_exists = connection.execute(
        text("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'staging'
                  AND table_name = 'train_raw'
            )
        """)
    ).scalar()

    if table_exists:
        raise RuntimeError(
            "staging.train_raw already exists.\n"
            "The loader will not overwrite an existing raw staging table."
        )

df.to_sql(
    name="train_raw",
    con=engine,
    schema="staging",
    if_exists="fail",
    index=False,
    chunksize=2_000,
    method="multi"
)

print("✓ Data loaded successfully")


# ============================================================
# 6. Post-Load Verification
# ============================================================

print("\nVerifying PostgreSQL table...")

with engine.connect() as connection:

    row_count = connection.execute(
        text("SELECT COUNT(*) FROM staging.train_raw")
    ).scalar()

    column_count = connection.execute(
        text("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'staging'
              AND table_name = 'train_raw'
        """)
    ).scalar()

    database_churn_count = connection.execute(
        text("""
            SELECT COUNT(*)
            FROM staging.train_raw
            WHERE churn_probability = 1
        """)
    ).scalar()


# ============================================================
# Final Result
# ============================================================

print("\n" + "=" * 60)
print("LOAD VERIFICATION")
print("=" * 60)

print(f"Rows in PostgreSQL:       {row_count:,}")
print(f"Columns in PostgreSQL:    {column_count:,}")
print(f"Churned customers:        {database_churn_count:,}")

if row_count != EXPECTED_ROWS:
    raise ValueError(
        f"Row verification failed: expected {EXPECTED_ROWS:,}, "
        f"found {row_count:,}."
    )

if column_count != EXPECTED_COLUMNS:
    raise ValueError(
        f"Column verification failed: expected {EXPECTED_COLUMNS}, "
        f"found {column_count}."
    )

if database_churn_count != churn_count:
    raise ValueError(
        "Target verification failed between CSV and PostgreSQL."
    )

print("\n✓ Row count verified")
print("✓ Column count verified")
print("✓ Target count verified")
print("✓ staging.train_raw created successfully")
print("✓ Raw CSV remains unchanged")

print("\nNext step: PostgreSQL data validation.")