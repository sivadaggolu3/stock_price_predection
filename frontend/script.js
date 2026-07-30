const API_BASE = "http://localhost:5000";

const symbolInput = document.getElementById("symbolInput");
const predictBtn = document.getElementById("predictBtn");
const resultSection = document.getElementById("resultSection");
const chartSection = document.getElementById("chartSection");
const errorSection = document.getElementById("errorSection");

const resultSymbol = document.getElementById("resultSymbol");
const resultCurrent = document.getElementById("resultCurrent");
const resultPredicted = document.getElementById("resultPredicted");
const resultDateInfo = document.getElementById("resultDateInfo");

let priceChart = null;

function showError(message) {
    errorSection.textContent = message;
    errorSection.classList.remove("d-none");
    resultSection.classList.add("d-none");
    chartSection.classList.add("d-none");
}

function hideError() {
    errorSection.classList.add("d-none");
}

function setLoading(loading) {
    predictBtn.disabled = loading;
    const btnText = predictBtn.querySelector(".btn-text");
    const spinner = predictBtn.querySelector(".spinner-border");
    if (loading) {
        btnText.textContent = "Predicting...";
        spinner.classList.remove("d-none");
    } else {
        btnText.textContent = "Predict Price";
        spinner.classList.add("d-none");
    }
}

function formatPrice(value) {
    return "$" + Number(value).toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function renderChart(historicalData, predictedPrice, predictionDate) {
    const ctx = document.getElementById("priceChart").getContext("2d");

    if (priceChart) {
        priceChart.destroy();
    }

    const labels = historicalData.map(d => d.date);
    const closes = historicalData.map(d => d.close);
    const extendedLabels = [...labels, predictionDate];
    const actualData = [...closes, null];
    const predictedData = closes.map(() => null).concat([predictedPrice]);

    priceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: extendedLabels,
            datasets: [
                {
                    label: "Actual Price",
                    data: actualData,
                    borderColor: "rgba(29, 155, 240, 0.9)",
                    backgroundColor: "rgba(29, 155, 240, 0.1)",
                    fill: true,
                    tension: 0.2,
                    spanGaps: true
                },
                {
                    label: "Predicted Price",
                    data: predictedData,
                    borderColor: "rgba(0, 186, 124, 0.9)",
                    backgroundColor: "rgba(0, 186, 124, 0.2)",
                    pointRadius: 8,
                    pointHoverRadius: 10
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#8b98a5"
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: "#8b98a5",
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: "rgba(56, 68, 77, 0.5)"
                    }
                },
                y: {
                    ticks: {
                        color: "#8b98a5"
                    },
                    grid: {
                        color: "rgba(56, 68, 77, 0.5)"
                    }
                }
            }
        }
    });
}

async function handlePredict() {
    const symbol = symbolInput.value.trim();
    if (!symbol) {
        showError("Please enter a stock symbol.");
        return;
    }

    hideError();
    setLoading(true);
    resultSection.classList.add("d-none");
    chartSection.classList.add("d-none");

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ symbol })
        });

        const data = await res.json();

        if (!res.ok) {
            showError(data.message || data.error || "Prediction failed.");
            return;
        }

        resultSymbol.textContent = data.symbol;
        resultCurrent.textContent = formatPrice(data.current_price);
        resultPredicted.textContent = formatPrice(data.predicted_price);
        resultDateInfo.textContent = `Prediction for ${data.prediction_date} (based on data up to ${data.last_date})`;

        resultSection.classList.remove("d-none");
        chartSection.classList.remove("d-none");

        renderChart(
            data.historical_data,
            data.predicted_price,
            data.prediction_date
        );
    } catch (err) {
        showError("Cannot reach the API. Make sure the backend is running on http://localhost:5000");
        console.error(err);
    } finally {
        setLoading(false);
    }
}

predictBtn.addEventListener("click", handlePredict);

symbolInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handlePredict();
});
