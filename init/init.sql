
CREATE DATABASE IF NOT EXISTS db;

-- Use the database
USE db;

-- Create the Users table
CREATE TABLE IF NOT EXISTS users (
    userid INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Create the EvalData table
CREATE TABLE IF NOT EXISTS eval_data (
    UploadID INT AUTO_INCREMENT PRIMARY KEY,
    UploadDate DATETIME NOT NULL,
    InputFileName VARCHAR(255) NOT NULL,
    OutputFileName VARCHAR(255) NOT NULL,
    UserID INT,
    FOREIGN KEY (UserID) REFERENCES users(userid)
);

