import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    // Create image preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const result = await response.json();
      setPrediction({
        soilType: result.soil_type,
        description: result.description,
      });
      setError(null);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    }
  };

  return (
    <div className="App">
      <h1>Soil Classification</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit">Predict</button>
      </form>

      {imagePreview && (
        <div className="image-preview">
          <img
            src={imagePreview}
            alt="Preview"
            style={{
              maxWidth: "100%",
              maxHeight: "300px",
              marginTop: "1rem",
              borderRadius: "8px",
            }}
          />
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {prediction && (
        <div className="result">
          <h2>{prediction.soilType}</h2>
          <p>
            <strong>Description:</strong> {prediction.description}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
