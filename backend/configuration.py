import os


DATABASE_URL = "database" if ( "PRODUCTION" in os.environ ) else "172.18.0.1"
DATABASE_USERNAME = "root"      
DATABASE_PASSWORD = "root"      
DATABASE_NAME     = "db"     

class Configuration:
    # SQLALCHEMY_DATABASE_URI   = "sqlite:///database.db" 
    SQLALCHEMY_DATABASE_URI   = f"mysql+pymysql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_URL}/{DATABASE_NAME}" 
