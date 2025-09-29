import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleExplain = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      // ✅ Use relative URL so it works in production and dev
      const response = await fetch("/explain", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: message }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error fetching explanation:", error);
      setResult({ error: "Failed to fetch explanation." });
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        backgroundColor: "#f9f9f9",
        minHeight: "100vh",
        padding: "40px",
      }}
    >
      <div
        style={{
          maxWidth: "700px",
          margin: "0 auto",
          backgroundColor: "#fff",
          padding: "30px",
          borderRadius: "12px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
        }}
      >
        <h1 style={{ textAlign: "center", marginBottom: "20px" }}>
          📩 SMS Spam Detection
        </h1>

        <form onSubmit={handleExplain} style={{ marginBottom: "20px" }}>
          <textarea
            rows="4"
            placeholder="Enter SMS message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            style={{
              width: "100%",
              padding: "12px",
              fontSize: "16px",
              borderRadius: "8px",
              border: "1px solid #ccc",
              resize: "none",
            }}
          />
          <button
            type="submit"
            style={{
              marginTop: "15px",
              width: "100%",
              padding: "12px",
              fontSize: "16px",
              borderRadius: "8px",
              border: "none",
              cursor: "pointer",
              backgroundColor: "#007bff",
              color: "white",
              fontWeight: "bold",
            }}
          >
            {loading ? "Analyzing..." : "Check Message"}
          </button>
        </form>

        {result && !result.error && (
          <div>
            <h2 style={{ textAlign: "center" }}>
              Prediction:{" "}
              <span
                style={{
                  color: result.label === "spam" ? "red" : "green",
                  fontWeight: "bold",
                }}
              >
                {result.label.toUpperCase()}
              </span>
            </h2>

            <h3 style={{ marginTop: "20px" }}>Top Contributing Words:</h3>

            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={result.top_contributing_words.map(([word, score]) => ({
                  word,
                  score,
                }))}
                margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="word" />
                <YAxis />
                <Tooltip />
                <Bar
                  dataKey="score"
                  fill={result.label === "spam" ? "#dc3545" : "#28a745"}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {result && result.error && (
          <p style={{ color: "red", textAlign: "center" }}>{result.error}</p>
        )}
      </div>
    </div>
  );
}

export default App;
