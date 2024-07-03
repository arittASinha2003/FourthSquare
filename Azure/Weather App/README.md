
# Real-Time Weather Application

This is a real-time weather application built using `Node.js`, `Express`, and the `OpenWeatherMap API`. It allows users to get current weather information by entering a city name.

Azure Web App Link: [Click Here](https://weatherjs.azurewebsites.net/)

## Features

- Fetches real-time weather data for any city.
- Displays temperature, humidity, cloudiness, visibility, wind speed, wind direction, pressure, sunrise, and sunset times.
- Changes background image based on the time of day (day or night).
- Provides a visually appealing and responsive user interface.

## Prerequisites

- Node.js and npm installed. [Click Here](https://nodejs.org/)
- OpenWeatherMap API Key. [Click Here](https://openweathermap.org/api)

## Installation

1. **Clone the repository:**

```bash
  git clone https://github.com/arittASinha2003/WeatherJS.git
  cd WeatherJS
```

2. **Install dependencies:**

```bash
  npm install
```

3. **Set up environment variables:**

Create a `.env` file in the root directory and add your OpenWeatherMap API Key and port number:

```plaintext
  API_KEY = your_openweathermap_api_key
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
Weather/
├── node_modules/
├── public/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── Day.jpg
│   ├── Night.jpg
├── .env
├── index.js
├── package.json
├── package-lock.json
```
