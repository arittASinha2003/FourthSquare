import express from "express";
import bodyParser from "body-parser";
import mssql from "mssql";
const { ConnectionPool } = mssql;
import dotenv from "dotenv";
dotenv.config();

const connectionString = process.env.DB_CONNECTION_STRING;
const pool = new ConnectionPool(connectionString);
const app = express();
const port = process.env.PORT || 3000;

let quiz = [];
let totalQuestions = 0; // Total questions user wants to attempt
let currentQuestionIndex = 0; // Current question index
let currentQuestion = {}; // Current question

pool.connect().then(() => {
  pool.request().query("SELECT * FROM capitals", (err, res) => {
    if (err) {
      console.error("Error executing query", err.stack);
    } else {
      quiz = res.recordset;
    }
    pool.close();
  });
});

let totalCorrect = 0;

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));
app.set("view engine", "ejs");

// Initial route to ask the number of questions
app.get("/", (req, res) => {
  res.render("start.ejs");
});

app.post("/start", (req, res) => {
  totalQuestions = parseInt(req.body.totalQuestions, 10);
  totalCorrect = 0;
  currentQuestionIndex = 0;
  nextQuestion();
  res.render("index.ejs", { question: currentQuestion, totalScore: totalCorrect, wasCorrect: null });
});

// POST a new answer
app.post("/submit", (req, res) => {
  let answer = req.body.answer.trim();
  let isCorrect = false;
  if (currentQuestion.capital.toLowerCase() === answer.toLowerCase()) {
    totalCorrect++;
    isCorrect = true;
  }

  currentQuestionIndex++;
  if (isCorrect && currentQuestionIndex < totalQuestions) {
    nextQuestion();
    res.render("index.ejs", {
      question: currentQuestion,
      wasCorrect: isCorrect,
      totalScore: totalCorrect,
    });
  } else {
    res.render("end.ejs", {
      totalScore: totalCorrect,
      maxScore: totalQuestions,
      completed: isCorrect && currentQuestionIndex === totalQuestions,
    });
  }
});

function nextQuestion() {
  const randomState = quiz[Math.floor(Math.random() * quiz.length)];
  currentQuestion = randomState;
}

app.listen(port, () => {
  console.log(`Server is running at https://localhost:${port}`);
});
