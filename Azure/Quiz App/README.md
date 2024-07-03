
# Capital City Quiz Application

The Capital City Quiz Application is a `Node.js` web application that quizzes users on the capital cities of various states of India, stored in `Azure SQL Database`. Users can specify the number of questions they want to attempt, and the quiz will end either when the specified number of questions has been answered or when a wrong answer is given.

Azure Web App Link: [Click Here](https://quizjs.azurewebsites.net/)

## Features

- Allows users to specify the number of quiz questions.
- Randomly selects questions from a database of capital cities.
- Tracks and displays the user's score.

## Prerequisites

- Node.js and npm installed. [Click Here](https://nodejs.org/)
- Azure SQL Server database setup with a table named `capitals` containing columns `state` and `capital`. [Click Here](https://portal.azure.com/)

## Installation

1. **Clone the repository:**

```bash
  git clone https://github.com/arittASinha2003/QuizJS.git
  cd QuizJS
```

2. **Install dependencies:**

```bash
  npm install
```

3. **Set up environment variables:**

Create a `.env` file in the root directory and add your database connection string and port number:

```plaintext
  DB_CONNECTION_STRING = your_connection_string_here
  PORT = 8080
```

4. **Run the application:**

```bash
  npm start
```

5. **Open your browser and navigate to:**

```plaintext
  https://localhost:8080/
```

## Folder Structure

```plaintext
Quiz/
├── node_modules/
├── public/
│   ├── images/
│   │   ├── background.jpg
│   ├── styles/
│       ├── main.css
├── views
│   ├── start.ejs
│   ├── index.ejs
│   ├── end.ejs
├── .env
├── index.js
├── package.json
├── package-lock.json
```
