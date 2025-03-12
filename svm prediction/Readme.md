# Heart Disease Prediction System

This project implements a comprehensive web-based system to predict the risk of heart disease using a machine learning model. 
#### Try it out at [heart-disease-prediction.harshbanka.codes](https://heart-disease-prediction.harshbanka.codes/)
##### NOTE: Server start may take a minute or two.

### Features

* **User Input Form:**  An intuitive web form for users to enter their health data.  
* **Machine Learning Model:**  A trained model that predicts heart disease risk based on the provided input. 
* **Prediction Display:**  Clear presentation of the prediction result to the user.
* **MongoDB Integration:**  Stores user input and prediction results in a MongoDB database.
* **Prediction History:**  A history page to view past prediction records.

### Technologies

* **Python**
* **FastAPI**
* **Jinja2 Templates**
* **MongoDB**
* **pymongo**
* **HTML, CSS, Bootstrap**
* **JavaScript**

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/<project-repository-name>
   ```

2. **Create a Virtual Environment:**
    *  **Recommended:** Use a tool like `venv` or `conda`.
    *  **Example (venv):** 
      ```bash
      python3 -m venv env
      source env/bin/activate 
      ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **MongoDB Setup**
   * **Local Installation:**
      *   Follow the instructions from the official MongoDB website: [https://www.mongodb.com/docs/](https://www.mongodb.com/docs/)
      *   Start the MongoDB server (usually with the `mongod` command).
   * **Cloud Service (MongoDB Atlas):** 
      *   Create a MongoDB Atlas account: [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
      *   Set up a cluster and obtain your database connection string.

5. **Configure MongoDB Connection**
    * **Option 1: Export MONGO_URL in Terminal**
      *   Set the `MONGO_URL` environment variable in your terminal before running the application:

          ```bash
          export MONGO_URL="your_mongo_connection_string"
          ```
   * **Option 2: Update Manually in the File**
      *   If you prefer not to use environment variables, you can manually update the `MONGO_URL` variable in the file with your MongoDB connection string.
      *   Locate `utils/database.py` or a similar file where you manage database connections.
      *   Update the connection function (e.g., `connect_to_mongo()`) with your MongoDB connection details:** 
      ```python
      # ... other code ...
            MONGO_URL = os.environ.get('MONGO_URL') or "mongodb://127.0.0.1:27017/"
            client = pymongo.MongoClient(MONGO_URL)    # Replace with your connection string 
          # ... rest of the code ... 
      ```

6.  **Run the Application:**
   ```bash
   uvicorn app:app --reload 
   ```

### Accessing the System

Open `http://127.0.0.1:8000` in your web browser to use the heart disease prediction system.

## Model Training
The machine learning model used for heart disease prediction is SVM using scikit-learn.

### License
This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.
