import pandas as pd
from sqlalchemy import create_engine
import os


host=os.getenv('ENERGY_DB_TDS_HOST')
user=os.getenv('ENERGY_DB_TDS_USER')
password=os.getenv('ENERGY_DB_TDS_PASSWORD')
database=os.getenv('ENERGY_DB_TDS_DATABASE')

#credentials are on bitwarden
# Database connection parameters
DB_CONFIG = {
    "host": f"{host}",
    "user": f"{user}",
    "password": f"{password}",
    "database": f"{database}"
}

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")


def fetch_data():
    """Connects to MySQL and retrieves data."""
    query = "SELECT * FROM job_energy where updated BETWEEN '2025-01-01' AND '2025-03-27' AND cluster='alps-santis';"  # Modify as needed
    df = pd.read_sql(query, engine)
    return df

# Fetch data
# for cluster in clusters:
#     df = fetch_data(cluster=cluster)
#     print(df.columns)
#     df.to_csv(f"{cluster}_2025-03-24.csv")
df = fetch_data()
df=df[df['elapsed']>60]

df.to_csv('raw_data_alps_santis_jan_feb_mar.csv')

