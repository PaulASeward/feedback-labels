# Data Visualization of Student Feedbacls
This project is a Dash web application designed to visualize the feedbacks provided by students.

## Installation

Ensure you have Python 3.6 or newer installed. It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash 
pip install -r requirements.txt
```

If this virtual environment is already created and requirements installed into it, we can just activate to use it:

```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Running the Application

To start the Dash server and run the application, execute:
    
```bash
python3 app.py
```

The application will be available at http://127.0.0.1:8050/


## Project Structure
    app.py: The main Dash application.
    select_utils.py: Utility for selecting courses, assignments, and tasks.
    plot_utils.py: Functions for plotting data visualizations.
    dimension_reduction.py: Dimensionality reduction utilities.
    embeddings.py: Processing and handling of embedding vectors.
    data/: Directory containing datasets used by the application.
