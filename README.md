# My Habit-Tracking App / Project

Useful information on installing and operating the app

https://github.com/jgoth9301/habittracker_app

## What is this app about?
This app is a habit tracking tool designed to help users build good habits and break bad ones. It allows users to define tasks they want to complete periodically, like daily workouts or annual appointments, and track their progress over time. By marking tasks as completed, users can build streaks and receive insights about their habits, such as their longest streaks or which habits need improvement. Focused on functionality, the app enables users to manage and analyze habits effectively without distractions, using Python-based object-oriented and functional programming. It’s a straightforward and powerful tool for personal growth and productivity.

## Installation instructions
To set up the habit tracking app, ensure you have Python installed (version 3.8 or higher). Clone the repository, navigate to the project directory, and install required dependencies using pip install -r requirements.txt. Once installed, you can run the app with python main.py.
```shell
pip install -r requirements.txt
```

## Usage instructions
The main.py file serves as the entry point for the application, orchestrating its core functionality. It initializes and runs the program, connecting key components like habit creation and tracking. This file includes detailed function docstrings to make the code easy to understand and navigate, ensuring clarity for all users and developers. By running main.py, users can interact with the app directly. Its structure ensures the program is intuitive, well-documented, and easy to maintain or extend.
Start...
```shell
python main.py
```

and follow instructions on screen.

Additional code is structured into modular components with logically separated files to enhance readability and maintainability:

dp.py => Creation and maintenance the SQL table structure in SQLite3 to efficiently store and manage app data.          
counter.py => Storage of functional code modules for managing habit & tracking operations (creation, deletion, reset functionality).    
analyse.py => Storage of functional code modules for managing the analysis of tracking information.


## Test instructions
The test_project.py file is designed to ensure the reliability and correctness of the application by running automated tests. It verifies key features, such as creating habits, tracking progress, and analyzing streaks, to ensure they work as expected. By using this file, developers can identify bugs early, validate changes, and maintain the app’s functionality over time. To run the tests, simply execute the file using a testing framework like pytest or unittest. Regular testing helps keep the project stable and robust.
```shell
python -m pytest
```