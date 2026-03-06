const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const statusIndicator = document.getElementById('api-status');
const statusDot = document.querySelector('.status-dot');
const btnAdvance = document.getElementById('btn-advance');
const btnFailure = document.getElementById('btn-failure');
const groundTruthBanner = document.getElementById('ground-truth-banner');
const tableBody = document.querySelector('#assets-table tbody');
const assetSelect = document.getElementById('asset-select');
const riskScoreBox = document.getElementById('risk-score-box');
const detailScore = document.getElementById('detail-score');
const recommendationBox = document.getElementById('recommendation-box');
const valCritical = document.getElementById('val-critical');
const valAvgRisk = document.getElementById('val-avg-risk');
const valTime = document.getElementById('val-time');

// Chart Instances
let shapChart = null;
let trendChart = null;

// History Arrays
let timeHistory = [];
let riskHistory = [];
async function init() {
    initTrendChart();
    await checkStatus();
}

function initTrendChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeHistory,
            datasets: [{
                label: 'Avg System Risk',
                data: riskHistory,
                borderColor: '#d29922',
                tension: 0.4,
                fill: true,
                backgroundColor: 'rgba(210, 153, 34, 0.1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#8b949e', maxTicksLimit: 5 }, grid: { display: false } },
                y: { min: 0, max: 100, ticks: { color: '#8b949e' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();

        if (data.status === 'ready') {
            statusIndicator.textContent = 'Pipeline Ready & Connected';
            statusDot.classList.add('ready');
            btnAdvance.disabled = false;
            btnFailure.disabled = false;
            assetSelect.disabled = false;

            // Initial Data Load
            await updateDashboard();
        } else {
            statusIndicator.textContent = 'Initializing External Models...';
            setTimeout(checkStatus, 2000);
        }
    } catch (e) {
        statusIndicator.textContent = 'Backend Disconnected';
        statusDot.classList.remove('ready');
        setTimeout(checkStatus, 3000);
    }
}

// Event Listeners
btnAdvance.addEventListener('click', async () => {
    btnAdvance.disabled = true;
    try {
        await fetch(`${API_BASE}/simulation/advance`);
        await updateDashboard();
    } catch (e) { console.error(e); }
    btnAdvance.disabled = false;
});



btnFailure.addEventListener('click', async () => {
    btnFailure.disabled = true;
    try {
        await fetch(`${API_BASE}/simulation/failure`);
        await updateDashboard();
    } catch (e) { console.error(e); }
    btnFailure.disabled = false;
});

assetSelect.addEventListener('change', async () => {
    await updateDashboard();
});

function getRiskClass(score) {
    if (score < 30) return 'low';
    if (score < 60) return 'med';
    return 'high';
}

function getRiskColorHex(score) {
    if (score < 30) return '#2ea043';
    if (score < 60) return '#d29922';
    return '#f85149';
}

async function updateDashboard() {
    try {
        const res = await fetch(`${API_BASE}/assets`);
        if (!res.ok) return;
        const data = await res.json();

        // 1. Update Metrics
        valTime.textContent = data.timestamp.split(' ')[1]; // Just show time

        const avgRisk = data.assets.reduce((sum, a) => sum + a.risk_score, 0) / data.assets.length;
        valAvgRisk.textContent = avgRisk.toFixed(1);

        const criticalCount = data.assets.filter(a => a.risk_score > 80).length;
        valCritical.textContent = criticalCount;

        // --- 1c. Update Trend Chart ---
        if (trendChart) {
            const timeStr = data.timestamp.split(' ')[1];

            // Push to arrays
            timeHistory.push(timeStr);
            riskHistory.push(avgRisk);

            // Keep window size of 20
            if (timeHistory.length > 20) {
                timeHistory.shift();
                riskHistory.shift();
            }

            trendChart.update();
        }

        // 2. Banner
        if (data.ground_truth_failure) {
            groundTruthBanner.classList.remove('hidden');
            groundTruthBanner.innerHTML = `🚨 ACTUAL SYSTEM GROUND TRUTH: Anomalous degradation occurred at ${data.timestamp}!`;
        } else {
            groundTruthBanner.classList.add('hidden');
        }

        // 3. Table
        tableBody.innerHTML = '';
        const limit = data.assets.length > 5 ? 5 : data.assets.length;
        data.assets.slice(0, limit).forEach(asset => {
            const riskClass = getRiskClass(asset.risk_score);
            const warningIcon = asset.early_failure_warning ? '⚠️ Yes' : '✅ No';
            // Use same generic risk colors for action text if desired
            let actionColor = '#8b949e';
            if (asset.risk_score > 80) actionColor = '#f85149';
            else if (asset.risk_score > 50) actionColor = '#d29922';

            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${asset.id}</td>
                <td class="score ${riskClass}">${asset.risk_score.toFixed(1)}</td>
                <td>${asset.trend}</td>
                <td>${asset.lstm_mse.toFixed(3)}</td>
                <td>${warningIcon}</td>
                <td style="color: ${actionColor}; font-weight: 500;">${asset.maintenance_recommendation}</td>
            `;
            tableBody.appendChild(row);
        });

        // 4. Update Detail Panel for the currently selected asset
        const activeAsset = data.assets.find(a => a.id === assetSelect.value);
        if (activeAsset) {
            // Update newly added Maintenance and Usage Stats
            document.getElementById('stat-maint').textContent = activeAsset.maintenance_history_days;
            document.getElementById('stat-usage').textContent = activeAsset.usage_frequency_score.toFixed(1);

            // New RUL and Fail prob
            document.getElementById('stat-rul').textContent = activeAsset.remaining_useful_life;
            document.getElementById('stat-fail-prob').textContent = activeAsset.risk_score.toFixed(1);

            detailScore.textContent = activeAsset.risk_score.toFixed(1);

            riskScoreBox.className = 'risk-box';
            if (activeAsset.risk_score > 80) {
                riskScoreBox.classList.add('score-red');
            } else if (activeAsset.risk_score > 50) {
                riskScoreBox.classList.add('score-orange');
            } else {
                riskScoreBox.className = 'risk-box';
            }

            // Use consistent icon based on risk
            let recIcon = '✅';
            if (activeAsset.risk_score > 80) recIcon = '🚨';
            else if (activeAsset.risk_score > 50) recIcon = '⚠️';

            // Smart Alert Engine Array parsing
            let recHTML = `<strong>${recIcon} Recommended Action:</strong><br><ul style="list-style: none; padding-left: 0; margin-top: 0.5rem; color: #e6edf3; font-size: 0.9rem;">`;
            if (Array.isArray(activeAsset.maintenance_recommendation)) {
                activeAsset.maintenance_recommendation.forEach(rec => {
                    recHTML += `<li style="margin-bottom: 0.25rem;">✓ ${rec}</li>`;
                });
            } else {
                recHTML += `<li>✓ ${activeAsset.maintenance_recommendation}</li>`;
            }
            recHTML += `</ul>`;

            recommendationBox.innerHTML = recHTML;

            await updateShapChart(activeAsset.id);
        }

        // --- 1e. Update Live Sensor Feed Console ---
        const consoleEl = document.getElementById('sensor-console');
        if (consoleEl && data.assets.length > 0) {
            let streamHTML = "";
            // Show top 3 assets streaming
            data.assets.slice(0, 3).forEach(a => {
                const s = a.sensor_data;
                if (s) {
                    streamHTML += `[LIVE] ${a.id.padEnd(18)} | Temp: ${s.temperature.toFixed(1)}°C | Vib: ${s.vibration.toFixed(2)}g | Load: ${s.load.toFixed(1)}%<br>`;
                }
            });
            consoleEl.innerHTML = streamHTML;
        }

    } catch (e) {
        console.error("Dashboard Update Failed", e);
    }
}

async function updateShapChart(assetId) {
    try {
        const res = await fetch(`${API_BASE}/asset/${assetId}/shap`);
        const data = await res.json();

        // Update RCA Text
        const rcaEl = document.getElementById('rca-text');
        if (rcaEl && data.root_cause_analysis) {
            rcaEl.textContent = data.root_cause_analysis;
        }

        const ctx = document.getElementById('shapChart').getContext('2d');

        if (shapChart) {
            shapChart.destroy();
        }

        // Determine colors (red for increased risk, green for reduced risk)
        const bgColors = data.values.map(v => v > 0 ? 'rgba(248, 81, 73, 0.8)' : 'rgba(46, 160, 67, 0.8)');

        // Use percentages for labels if available, otherwise raw values
        const displayData = data.percentages ? data.percentages : data.values;
        // Format labels: Feature (+35%)
        const formattedLabels = data.features.map((f, i) => {
            const sign = data.values[i] > 0 ? '+' : '';
            const perc = data.percentages ? data.percentages[i].toFixed(1) : '?';
            return `${f} (${sign}${perc}%)`;
        });

        shapChart = new Chart(ctx, {
            type: 'bar', // A simple bar chart showing positive/negative contribution magnitude
            data: {
                labels: formattedLabels,
                datasets: [{
                    data: displayData,
                    backgroundColor: bgColors,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y', // Horizontal bar chart
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return `Contribution: ${context.raw}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#7d8590' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#e6edf3' }
                    }
                }
            }
        });

    } catch (e) {
        console.error("Failed to load SHAP data", e);
    }
}

// Start sequence
init();


