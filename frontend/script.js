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

// Chart Instance
let shapChart = null;

// Initialization
async function init() {
    await checkStatus();
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

            detailScore.textContent = activeAsset.risk_score.toFixed(1);

            riskScoreBox.className = 'risk-box';
            if (activeAsset.risk_score > 80) {
                riskScoreBox.classList.add('score-red');
            } else if (activeAsset.risk_score > 50) {
                riskScoreBox.classList.add('score-orange');
            } else {
                riskScoreBox.className = 'risk-box';
            }

            let recIcon = '✅';
            if (activeAsset.risk_score > 80) recIcon = '🚨';
            else if (activeAsset.risk_score > 50) recIcon = '⚠️';

            recommendationBox.innerHTML = `${recIcon} <strong>Action:</strong> ${activeAsset.maintenance_recommendation}`;

            await updateShapChart(activeAsset.id);
        }

    } catch (e) {
        console.error("Dashboard Update Failed", e);
    }
}

async function updateShapChart(assetId) {
    try {
        const res = await fetch(`${API_BASE}/asset/${assetId}/shap`);
        const data = await res.json();

        const ctx = document.getElementById('shapChart').getContext('2d');

        if (shapChart) {
            shapChart.destroy();
        }

        // Determine colors (red for increased risk, green for reduced risk)
        const bgColors = data.values.map(v => v > 0 ? 'rgba(248, 81, 73, 0.8)' : 'rgba(46, 160, 67, 0.8)');

        shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.features,
                datasets: [{
                    data: data.values,
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
                    legend: { display: false }
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
