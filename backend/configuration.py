import os


DATABASE_URL      = "localhost" 
DATABASE_USERNAME = "root"      
DATABASE_PASSWORD = "root"      
DATABASE_NAME     = "database"     

class Configuration:
    # SQLALCHEMY_DATABASE_URI   = "sqlite:///database.db" 
    SQLALCHEMY_DATABASE_URI   = f"mysql+pymysql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_URL}/{DATABASE_NAME}" 
