/**
 * HPC Network Traffic Analysis — Interactive Benchmark Frontend
 */

// ── Global State ──────────────────────────────────────────────
let charts = {};
let isRunning = false;

// ── Initialization ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    init();
});

function init() {
    checkStatus();
    loadHistory();
    initCharts();
    document.getElementById('mpiRanks').addEventListener('change', updateHybridTotal);
    document.getElementById('ompThreads').addEventListener('change', updateHybridTotal);
    setInterval(checkStatus, 30000);
}

// ── Status Checking ───────────────────────────────────────────
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const s = await response.json();

        const indicator = document.getElementById('statusIndicator');
        const text      = document.getElementById('statusText');
        const detail    = document.getElementById('statusDetail');
        const chips     = document.getElementById('statusChips');

        let issues = [];
        if (!s.gcc)           issues.push('GCC missing');
        if (!s.mpicc)         issues.push('MPI compiler missing');
        if (!s.mpirun)        issues.push('mpirun missing');
        if (!s.dataset_exists) issues.push('Dataset missing');

        const cpuBinaries = ['serial','openmp','pthreads','mpi','hybrid'];
        const allCpu = cpuBinaries.every(k => s.binaries[k]);
        if (!allCpu) issues.push('CPU binaries not built');

        // Build status chips
        const chipDefs = [
            { label: 'GCC',     ok: s.gcc },
            { label: 'MPI',     ok: s.mpicc && s.mpirun },
            { label: 'Binaries', ok: allCpu },
            { label: s.binaries.cuda ? 'CUDA' : 'CUDA★SIM',
              ok: true, sim: !s.binaries.cuda },
        ];

        chips.innerHTML = chipDefs.map(c => {
            const cls = c.sim ? 'chip chip-sim' : (c.ok ? 'chip chip-ok' : 'chip chip-bad');
            return `<span class="${cls}">${c.label}</span>`;
        }).join('');

        if (issues.length === 0) {
            indicator.className = 'status-indicator status-ok';
            text.textContent = 'System Ready';
            detail.textContent = s.cuda_simulated ? 'CUDA will use simulation (no GPU)' : 'All tools available';
        } else {
            indicator.className = 'status-indicator status-warn';
            text.textContent = 'Issues Detected';
            detail.textContent = issues.join(' · ');
        }
    } catch (e) {
        document.getElementById('statusIndicator').className = 'status-indicator status-error';
        document.getElementById('statusText').textContent = 'Backend Offline';
        document.getElementById('statusDetail').textContent = 'Cannot connect to Flask server';
    }
}

// ── UI Controls ───────────────────────────────────────────────
function updateWorkerControls() {
    const impl        = document.getElementById('implementation').value;
    const workerGroup = document.getElementById('workerGroup');
    const hybridGroup = document.getElementById('hybridGroup');
    const cudaGroup   = document.getElementById('cudaGroup');
    const cudaNotice  = document.getElementById('cudaNotice');

    workerGroup.classList.add('hidden');
    hybridGroup.classList.add('hidden');
    cudaGroup.classList.add('hidden');
    cudaNotice.classList.add('hidden');

    if (impl === 'hybrid') {
        hybridGroup.classList.remove('hidden');
        updateHybridTotal();
    } else if (impl === 'cuda') {
        cudaGroup.classList.remove('hidden');
        cudaNotice.classList.remove('hidden');
    } else if (impl !== 'serial') {
        workerGroup.classList.remove('hidden');
    }
}

function updateHybridTotal() {
    const np = parseInt(document.getElementById('mpiRanks').value);
    const nt = parseInt(document.getElementById('ompThreads').value);
    document.getElementById('hybridTotal').textContent = `Total: ${np * nt} workers`;
}

function setControlsEnabled(enabled) {
    ['btnBuild','btnRun','btnRunAll','btnExport','btnClear'].forEach(id => {
        document.getElementById(id).disabled = !enabled;
    });
    document.getElementById('btnStop').disabled = enabled;
    ['implementation','workers','mpiRanks','ompThreads','blockSize'].forEach(id => {
        document.getElementById(id).disabled = !enabled;
    });

    const progressWrap = document.getElementById('progressWrap');
    if (!enabled) {
        progressWrap.classList.remove('hidden');
        animateProgress();
    } else {
        progressWrap.classList.add('hidden');
    }
}

let progressInterval = null;
function animateProgress() {
    const bar = document.getElementById('progressBar');
    let pct = 0;
    clearInterval(progressInterval);
    bar.style.width = '0%';
    progressInterval = setInterval(() => {
        pct = Math.min(pct + Math.random() * 3, 92);
        bar.style.width = pct + '%';
    }, 200);
}

// ── Console Output ────────────────────────────────────────────
function appendConsole(text, type = 'output') {
    const div  = document.getElementById('console');
    const line = document.createElement('div');
    line.className = `console-line console-${type}`;
    line.textContent = text;
    div.appendChild(line);
    div.scrollTop = div.scrollHeight;
}

function clearConsole() {
    document.getElementById('console').innerHTML =
        '<div class="console-line console-info">Console cleared.</div>';
}

// ── SSE Streaming ─────────────────────────────────────────────
async function streamRequest(url, body) {
    isRunning = true;
    setControlsEnabled(false);
    clearConsole();

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try { handleStreamData(JSON.parse(line.slice(6))); }
                    catch (e) { console.error('Parse error:', e); }
                }
            }
        }
        if (buffer.startsWith('data: ')) {
            try { handleStreamData(JSON.parse(buffer.slice(6))); } catch (e) {}
        }
    } catch (e) {
        appendConsole(`Error: ${e.message}`, 'error');
    } finally {
        clearInterval(progressInterval);
        document.getElementById('progressBar').style.width = '100%';
        setTimeout(() => {
            isRunning = false;
            setControlsEnabled(true);
        }, 300);
    }
}

function handleStreamData(data) {
    switch (data.type) {
        case 'output': appendConsole(data.text, 'output'); break;
        case 'error':  appendConsole(data.text, 'error');  break;
        case 'header': appendConsole(data.text, 'header'); break;
        case 'result':
            if (data.data) displayResults(data.data);
            break;
        case 'done':
            appendConsole(data.text || 'Operation completed.', 'info');
            loadHistory();
            updateCharts();
            break;
    }
}

// ── Actions ───────────────────────────────────────────────────
function buildAll() {
    appendConsole('Building all implementations...', 'header');
    streamRequest('/api/build', { target: 'all' });
}

function runBenchmark() {
    const impl    = document.getElementById('implementation').value;
    const dataset = document.getElementById('dataset').value;

    let body = { implementation: impl, dataset };

    if (impl === 'hybrid') {
        body.mpi_ranks   = parseInt(document.getElementById('mpiRanks').value);
        body.omp_threads = parseInt(document.getElementById('ompThreads').value);
    } else if (impl === 'cuda') {
        body.block_size = parseInt(document.getElementById('blockSize').value);
    } else if (impl !== 'serial') {
        body.workers = parseInt(document.getElementById('workers').value);
    }

    appendConsole(`Running ${impl.toUpperCase()} benchmark...`, 'header');
    streamRequest('/api/run', body);
}

function runAllBenchmarks() {
    const dataset = document.getElementById('dataset').value;
    appendConsole('Running full benchmark suite (all implementations including CUDA simulation)...', 'header');
    streamRequest('/api/run-all', {
        dataset,
        implementations: ['serial', 'openmp', 'pthreads', 'mpi', 'hybrid', 'cuda'],
        workers: [1, 2, 4, 8, 16],
        block_sizes: [256]
    });
}

async function stopBenchmark() {
    clearInterval(progressInterval);
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const result   = await response.json();
        appendConsole(result.message, result.success ? 'info' : 'error');
    } catch (e) {
        appendConsole('Failed to stop process', 'error');
    }
}

async function clearResults() {
    if (!confirm('Clear all results history?')) return;
    try {
        const response = await fetch('/api/results/clear', { method: 'POST' });
        const result   = await response.json();
        if (result.success) {
            loadHistory();
            updateCharts();
            appendConsole('Results history cleared.', 'info');
        }
    } catch (e) {
        appendConsole('Failed to clear results', 'error');
    }
}

async function exportResults() {
    try {
        const response = await fetch('/api/results');
        const results  = await response.json();
        if (!results.length) {
            appendConsole('No results to export.', 'info');
            return;
        }
        const headers = ['timestamp','implementation','workers','mpi_ranks','omp_threads',
                         'block_size','time_s','speedup','efficiency','throughput',
                         'accuracy','precision','recall','f1','rmse','simulated'];
        const rows = results.map(r => {
            const m = r.metrics || {};
            return [
                r.timestamp, r.implementation,
                r.workers || '', r.mpi_ranks || '', r.omp_threads || '',
                r.block_size || '',
                m.single_time || '', m.speedup || '', m.efficiency || '',
                m.throughput || '', m.accuracy || '', m.precision || '',
                m.recall || '', m.f1 || '', m.rmse || '',
                r.simulated ? 'yes' : 'no'
            ].join(',');
        });
        const csv  = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const a    = document.createElement('a');
        a.href     = URL.createObjectURL(blob);
        a.download = `hpc_results_${new Date().toISOString().slice(0,10)}.csv`;
        a.click();
        appendConsole(`Exported ${results.length} results to CSV.`, 'info');
    } catch (e) {
        appendConsole('Export failed: ' + e.message, 'error');
    }
}

// ── Results Display ───────────────────────────────────────────
function displayResults(result) {
    document.getElementById('resultsCard').classList.remove('hidden');

    const m        = result.metrics || {};
    const simBadge = document.getElementById('simBadge');
    if (result.simulated) simBadge.classList.remove('hidden');
    else                  simBadge.classList.add('hidden');

    updateMetric('metricTime',       m.single_time,  v => v ? v.toFixed(4) : '—');
    updateMetric('metricThroughput', m.throughput,   v => v ? formatNumber(v) : '—');
    updateMetric('metricSpeedup',    m.speedup,      v => v ? v.toFixed(2) + '×' : '—');
    updateMetric('metricEfficiency', m.efficiency,   v => v ? v.toFixed(1) + '%' : '—');
    updateMetric('metricAccuracy',   m.accuracy,     v => v ? v.toFixed(2) + '%' : '—');
    updateMetric('metricF1',         m.f1,           v => v ? v.toFixed(2) + '%' : '—');

    document.getElementById('cmTN').textContent = m.tn !== undefined ? formatNumber(m.tn) : '—';
    document.getElementById('cmFP').textContent = m.fp !== undefined ? formatNumber(m.fp) : '—';
    document.getElementById('cmFN').textContent = m.fn !== undefined ? formatNumber(m.fn) : '—';
    document.getElementById('cmTP').textContent = m.tp !== undefined ? formatNumber(m.tp) : '—';

    updateAccuracyBar('barPrecision', 'valPrecision', m.precision);
    updateAccuracyBar('barRecall',    'valRecall',    m.recall);
    updateAccuracyBar('barF1',        'valF1',        m.f1);
    document.getElementById('valRMSE').textContent =
        m.rmse !== undefined ? m.rmse.toFixed(6) : '—';

    const statusEl = document.getElementById('valStatus');
    if (m.status) {
        statusEl.textContent = m.status;
        statusEl.className = `acc-status status-${m.status.toLowerCase()}`;
    } else {
        statusEl.textContent = '—';
        statusEl.className = 'acc-status';
    }
}

function updateMetric(id, value, formatter) {
    document.getElementById(id).querySelector('.metric-value').textContent = formatter(value);
}

function updateAccuracyBar(barId, valueId, value) {
    document.getElementById(barId).style.width = (value !== undefined && value !== null)
        ? Math.min(value, 100) + '%' : '0%';
    document.getElementById(valueId).textContent = (value !== undefined && value !== null)
        ? value.toFixed(2) + '%' : '—';
}

// ── History Table ─────────────────────────────────────────────
async function loadHistory() {
    try {
        const response = await fetch('/api/results');
        const results  = await response.json();
        const tbody    = document.getElementById('historyBody');

        if (!results.length) {
            tbody.innerHTML = '<tr><td colspan="10" class="table-empty">No results yet. Run a benchmark to see results here.</td></tr>';
            return;
        }

        results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        tbody.innerHTML = results.map(r => {
            const m         = r.metrics || {};
            const implClass = `row-${r.implementation}`;
            const timeStr   = new Date(r.timestamp).toLocaleTimeString();

            let workers = r.workers;
            if (r.implementation === 'hybrid' && r.mpi_ranks)
                workers = `${r.mpi_ranks}×${r.omp_threads}`;
            else if (r.implementation === 'cuda')
                workers = `bs=${r.block_size || 256}`;

            const note = r.simulated ? '<span class="sim-badge" style="font-size:9px">SIM</span>' : '';

            return `<tr>
                <td>${timeStr}</td>
                <td class="${implClass}">${r.implementation.toUpperCase()}</td>
                <td>${workers || '—'}</td>
                <td>${m.single_time ? m.single_time.toFixed(4) : '—'}</td>
                <td class="${m.speedup > 1 ? 'val-good' : ''}">${m.speedup ? m.speedup.toFixed(2) + '×' : '—'}</td>
                <td>${m.efficiency ? m.efficiency.toFixed(1) + '%' : '—'}</td>
                <td>${m.throughput ? formatNumber(m.throughput) : '—'}</td>
                <td class="${m.accuracy > 90 ? 'val-good' : m.accuracy > 75 ? 'val-warn' : 'val-poor'}">${m.accuracy ? m.accuracy.toFixed(2) + '%' : '—'}</td>
                <td>${m.f1 ? m.f1.toFixed(2) + '%' : '—'}</td>
                <td>${note}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

// ── Charts ────────────────────────────────────────────────────
function initCharts() {
    const gridColor  = '#21262d';
    const textColor  = '#484f58';
    const labelColor = '#8b949e';

    const OMP  = '#58a6ff';
    const PTH  = '#bc8cff';
    const MPI  = '#db6d28';
    const HYB  = '#3fb950';
    const CUDA = '#e3b341';

    const baseOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: labelColor,
                    font: { family: 'JetBrains Mono, monospace', size: 11 },
                    boxWidth: 10,
                    boxHeight: 10,
                    padding: 14,
                }
            },
            tooltip: {
                backgroundColor: '#161b22',
                borderColor: '#30363d',
                borderWidth: 1,
                titleColor: '#e6edf3',
                bodyColor: '#8b949e',
                padding: 10,
                cornerRadius: 6,
            }
        },
        scales: {
            x: {
                grid: { color: gridColor, drawBorder: false },
                ticks: { color: textColor, font: { family: 'JetBrains Mono, monospace', size: 11 } }
            },
            y: {
                grid: { color: gridColor, drawBorder: false },
                ticks: { color: textColor, font: { family: 'JetBrains Mono, monospace', size: 11 } }
            }
        }
    };

    charts.speedup = new Chart(document.getElementById('chartSpeedup'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Ideal',    data: [], borderColor: 'rgba(255,255,255,.12)', borderDash: [5,4], pointRadius: 0, tension: 0, borderWidth: 1.5 },
                { label: 'OpenMP',   data: [], borderColor: OMP,  backgroundColor: OMP  + '18', pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'Pthreads', data: [], borderColor: PTH,  backgroundColor: PTH  + '18', pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'MPI',      data: [], borderColor: MPI,  backgroundColor: MPI  + '18', pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'Hybrid',   data: [], borderColor: HYB,  backgroundColor: HYB  + '18', pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'CUDA',     data: [], borderColor: CUDA, backgroundColor: CUDA + '18', pointRadius: 7, pointHoverRadius: 9, pointStyle: 'star', tension: 0, borderWidth: 2 },
            ]
        },
        options: { ...baseOpts, scales: { ...baseOpts.scales,
            y: { ...baseOpts.scales.y, title: { display: true, text: 'Speedup (×)', color: labelColor, font: { size: 11 } } }
        }}
    });

    charts.efficiency = new Chart(document.getElementById('chartEfficiency'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'OpenMP',   data: [], borderColor: OMP,  pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'Pthreads', data: [], borderColor: PTH,  pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'MPI',      data: [], borderColor: MPI,  pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
                { label: 'Hybrid',   data: [], borderColor: HYB,  pointRadius: 4, pointHoverRadius: 6, tension: 0.25, borderWidth: 2 },
            ]
        },
        options: { ...baseOpts, scales: { ...baseOpts.scales,
            y: { ...baseOpts.scales.y, min: 0, max: 130, title: { display: true, text: 'Efficiency (%)', color: labelColor, font: { size: 11 } } }
        }}
    });

    charts.time = new Chart(document.getElementById('chartTime'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                { label: 'OpenMP',   data: [], backgroundColor: OMP  + 'aa', borderRadius: 3 },
                { label: 'Pthreads', data: [], backgroundColor: PTH  + 'aa', borderRadius: 3 },
                { label: 'MPI',      data: [], backgroundColor: MPI  + 'aa', borderRadius: 3 },
                { label: 'Hybrid',   data: [], backgroundColor: HYB  + 'aa', borderRadius: 3 },
                { label: 'CUDA',     data: [], backgroundColor: CUDA + 'aa', borderRadius: 3 },
            ]
        },
        options: { ...baseOpts, scales: { ...baseOpts.scales,
            y: { ...baseOpts.scales.y, title: { display: true, text: 'Time (seconds)', color: labelColor, font: { size: 11 } } }
        }}
    });

    charts.throughput = new Chart(document.getElementById('chartThroughput'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                { label: 'OpenMP',   data: [], backgroundColor: OMP  + 'aa', borderRadius: 3 },
                { label: 'Pthreads', data: [], backgroundColor: PTH  + 'aa', borderRadius: 3 },
                { label: 'MPI',      data: [], backgroundColor: MPI  + 'aa', borderRadius: 3 },
                { label: 'Hybrid',   data: [], backgroundColor: HYB  + 'aa', borderRadius: 3 },
                { label: 'CUDA',     data: [], backgroundColor: CUDA + 'aa', borderRadius: 3 },
            ]
        },
        options: { ...baseOpts, scales: { ...baseOpts.scales,
            y: { ...baseOpts.scales.y, title: { display: true, text: 'Throughput (rec/s)', color: labelColor, font: { size: 11 } } }
        }}
    });

    updateCharts();
}

async function updateCharts() {
    try {
        const response = await fetch('/api/charts-data');
        const data = await response.json();

        const allWorkers = new Set();
        ['openmp','pthreads','mpi'].forEach(impl => {
            (data[impl] || []).forEach(d => allWorkers.add(d.workers));
        });
        const labels = Array.from(allWorkers).sort((a, b) => a - b);

        // Speedup
        const cudaSpeedup = data.cuda && data.cuda.length
            ? [{ workers: 'GPU', speedup: data.cuda[0].speedup }] : [];

        charts.speedup.data.labels = labels;
        charts.speedup.data.datasets[0].data = labels;
        charts.speedup.data.datasets[1].data = getChartData(data.openmp,   labels, 'speedup');
        charts.speedup.data.datasets[2].data = getChartData(data.pthreads, labels, 'speedup');
        charts.speedup.data.datasets[3].data = getChartData(data.mpi,      labels, 'speedup');
        charts.speedup.data.datasets[4].data = getChartData(data.hybrid,   labels, 'speedup');
        // CUDA single point — show at the last label position
        if (data.cuda && data.cuda.length) {
            const cudaLabels = [...labels];
            const cudaPts = labels.map(() => null);
            cudaPts[cudaPts.length - 1] = data.cuda[0].speedup;
            charts.speedup.data.datasets[5].data = cudaPts;
        } else {
            charts.speedup.data.datasets[5].data = [];
        }
        charts.speedup.update();

        // Efficiency
        charts.efficiency.data.labels = labels;
        charts.efficiency.data.datasets[0].data = getChartData(data.openmp,   labels, 'efficiency');
        charts.efficiency.data.datasets[1].data = getChartData(data.pthreads, labels, 'efficiency');
        charts.efficiency.data.datasets[2].data = getChartData(data.mpi,      labels, 'efficiency');
        charts.efficiency.data.datasets[3].data = getChartData(data.hybrid,   labels, 'efficiency');
        charts.efficiency.update();

        // Time + Throughput (CUDA shown as single bar at "GPU" label)
        const barLabels = labels.length ? labels : [];
        const cudaLabels = [...barLabels, 'GPU'];

        charts.time.data.labels = cudaLabels;
        charts.time.data.datasets[0].data = [...getChartData(data.openmp,   barLabels, 'time'), null];
        charts.time.data.datasets[1].data = [...getChartData(data.pthreads, barLabels, 'time'), null];
        charts.time.data.datasets[2].data = [...getChartData(data.mpi,      barLabels, 'time'), null];
        charts.time.data.datasets[3].data = [...getChartData(data.hybrid,   barLabels, 'time'), null];
        charts.time.data.datasets[4].data = [
            ...barLabels.map(() => null),
            data.cuda && data.cuda.length ? data.cuda[0].time : null
        ];
        charts.time.update();

        charts.throughput.data.labels = cudaLabels;
        charts.throughput.data.datasets[0].data = [...getChartData(data.openmp,   barLabels, 'throughput'), null];
        charts.throughput.data.datasets[1].data = [...getChartData(data.pthreads, barLabels, 'throughput'), null];
        charts.throughput.data.datasets[2].data = [...getChartData(data.mpi,      barLabels, 'throughput'), null];
        charts.throughput.data.datasets[3].data = [...getChartData(data.hybrid,   barLabels, 'throughput'), null];
        charts.throughput.data.datasets[4].data = [
            ...barLabels.map(() => null),
            data.cuda && data.cuda.length ? data.cuda[0].throughput : null
        ];
        charts.throughput.update();

    } catch (e) {
        console.error('Failed to update charts:', e);
    }
}

function getChartData(implData, labels, field) {
    if (!implData) return labels.map(() => null);
    const map = {};
    implData.forEach(d => { map[d.workers] = d[field]; });
    return labels.map(w => map[w] !== undefined ? map[w] : null);
}

// ── Utilities ─────────────────────────────────────────────────
function formatNumber(num) {
    if (num === undefined || num === null) return '—';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toString();
}
