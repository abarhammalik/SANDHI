// ARTHROSCAN Multimodal Diagnostic Assistant Client Logic

const GAUGE_CIRCUMFERENCE = 565.48; // 2 * Math.PI * 90
let debounceTimer = null;
let currentBMIUnit = 'metric';
let lastPredictionResult = null;

// ==========================================================================
// 1. THEME SWITCHING (LIGHT / DARK)
// ==========================================================================

function initTheme() {
  const savedTheme = localStorage.getItem('arthroscan_theme');
  if (savedTheme) {
    document.documentElement.setAttribute('data-theme', savedTheme);
  } else {
    // Detect OS preference
    const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
    document.documentElement.setAttribute('data-theme', prefersLight ? 'light' : 'dark');
  }
  updateThemeButtonText();
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('arthroscan_theme', next);
  updateThemeButtonText();
}

function updateThemeButtonText() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const label = document.getElementById('theme-text');
  if (label) {
    label.innerText = current === 'dark' ? 'Dark' : 'Light';
  }
}

// ==========================================================================
// 2. BMI CALCULATOR LOGIC
// ==========================================================================

function toggleBMICalculator() {
  const drawer = document.getElementById('bmi-calculator-drawer');
  drawer.classList.toggle('open');
  if (drawer.classList.contains('open')) {
    calculateBMI();
  }
}

function setBMIUnit(unit) {
  currentBMIUnit = unit;
  document.getElementById('unit-metric').classList.toggle('active', unit === 'metric');
  document.getElementById('unit-imperial').classList.toggle('active', unit === 'imperial');

  document.getElementById('height-metric-group').classList.toggle('hide', unit !== 'metric');
  document.getElementById('weight-metric-group').classList.toggle('hide', unit !== 'metric');
  document.getElementById('height-imperial-group').classList.toggle('hide', unit !== 'imperial');
  document.getElementById('weight-imperial-group').classList.toggle('hide', unit !== 'imperial');

  calculateBMI();
}

function calculateBMI() {
  let bmi = 25.0;

  if (currentBMIUnit === 'metric') {
    const h_cm = parseFloat(document.getElementById('calc-height').value) || 170;
    const w_kg = parseFloat(document.getElementById('calc-weight').value) || 70;
    const h_m = h_cm / 100.0;
    bmi = w_kg / (h_m * h_m);
  } else {
    const h_in = parseFloat(document.getElementById('calc-height-in').value) || 67;
    const w_lbs = parseFloat(document.getElementById('calc-weight-lbs').value) || 155;
    bmi = (w_lbs / (h_in * h_in)) * 703.0;
  }

  bmi = Math.max(12.0, Math.min(60.0, bmi));
  const roundedBMI = Math.round(bmi * 10) / 10;

  document.getElementById('calc-bmi-output').innerText = `${roundedBMI.toFixed(1)} kg/m²`;

  // Classification & Category Badge
  let cat = 'Normal Weight';
  let badgeClass = 'var(--status-green)';
  let bgClass = 'var(--status-green-bg)';
  let borderClass = 'var(--status-green-border)';

  if (roundedBMI < 18.5) {
    cat = 'Underweight';
    badgeClass = '#38bdf8';
    bgClass = 'rgba(56, 189, 248, 0.15)';
    borderClass = 'rgba(56, 189, 248, 0.35)';
  } else if (roundedBMI < 25.0) {
    cat = 'Normal Weight';
    badgeClass = 'var(--status-green)';
    bgClass = 'var(--status-green-bg)';
    borderClass = 'var(--status-green-border)';
  } else if (roundedBMI < 30.0) {
    cat = 'Overweight';
    badgeClass = 'var(--status-yellow)';
    bgClass = 'var(--status-yellow-bg)';
    borderClass = 'var(--status-yellow-border)';
  } else {
    cat = 'Obese Class';
    badgeClass = 'var(--status-red)';
    bgClass = 'var(--status-red-bg)';
    borderClass = 'var(--status-red-border)';
  }

  const catEl = document.getElementById('calc-bmi-category');
  catEl.innerText = cat;
  catEl.style.color = badgeClass;
  catEl.style.backgroundColor = bgClass;
  catEl.style.borderColor = borderClass;

  // Visual marker position: 15 to 40 range mapped to 0% - 100%
  const markerPos = Math.max(0, Math.min(100, ((roundedBMI - 15) / 25) * 100));
  document.getElementById('bmi-marker').style.left = `${markerPos}%`;

  return roundedBMI;
}

function applyCalculatedBMI() {
  const calculated = calculateBMI();
  const mainBMIInput = document.getElementById('bmi');
  mainBMIInput.value = calculated.toFixed(1);
  updateBMIPill(calculated);
  toggleBMICalculator();
  handleInputChange();
}

function updateBMIPill(bmiVal) {
  const pill = document.getElementById('bmi-status-pill');
  if (!pill) return;
  if (bmiVal < 18.5) {
    pill.innerText = 'Underweight';
    pill.style.color = '#38bdf8';
  } else if (bmiVal < 25.0) {
    pill.innerText = 'Normal';
    pill.style.color = 'var(--status-green)';
  } else if (bmiVal < 30.0) {
    pill.innerText = 'Overweight';
    pill.style.color = 'var(--status-yellow)';
  } else {
    pill.innerText = 'Obese';
    pill.style.color = 'var(--status-red)';
  }
}

// ==========================================================================
// 3. STEPPERS & INTERACTIVE CONTROLS
// ==========================================================================

function stepVal(id, delta) {
  const el = document.getElementById(id);
  if (!el) return;
  let val = parseFloat(el.value) || 0;
  val = Math.max(parseFloat(el.min) || 0, Math.min(parseFloat(el.max) || 999, val + delta));
  el.value = el.step && el.step.includes('.') ? val.toFixed(1) : Math.round(val);
  if (id === 'bmi') updateBMIPill(val);
  handleInputChange();
}

function setGender(val) {
  document.getElementById('gender').value = val;
  handleInputChange();
}

function updateQualityText(mod) {
  const slider = document.getElementById(`${mod}_quality`);
  const label = document.getElementById(`${mod}-q-val`);
  if (slider && label) {
    const pct = Math.round(slider.value * 100);
    const qualityDesc = pct >= 80 ? '(High Quality)' : pct >= 50 ? '(Moderate SNR)' : '(Degraded / Noisy)';
    label.innerText = `${pct}% ${qualityDesc}`;
  }
}

function toggleModality(mod) {
  const chk = document.getElementById(`toggle-has-${mod}`);
  const card = document.getElementById(`${mod}-card`);
  const statusText = document.getElementById(`${mod}-status-text`);

  if (chk.checked) {
    card.classList.remove('disabled');
    statusText.innerText = 'Sensor Connected';
  } else {
    card.classList.add('disabled');
    statusText.innerText = 'Sensor Missing';
  }
  handleInputChange();
}

function handleInputChange() {
  const liveToggle = document.getElementById('live-predict-toggle');
  if (liveToggle && liveToggle.checked) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      submitPrediction();
    }, 150);
  }
}

// ==========================================================================
// 4. DATA COLLECTION & PREDICTION API
// ==========================================================================

function getFormData() {
  const has_imu = document.getElementById('toggle-has-imu').checked;
  const has_semg = document.getElementById('toggle-has-semg').checked;

  const data = {
    age: parseFloat(document.getElementById('age').value) || 60,
    gender: parseFloat(document.getElementById('gender').value) || 1,
    bmi: parseFloat(document.getElementById('bmi').value) || 25,
    has_imu: has_imu,
    has_semg: has_semg,
    imu_quality: parseFloat(document.getElementById('imu_quality').value) || 1.0,
    semg_quality: parseFloat(document.getElementById('semg_quality').value) || 1.0,

    // IMU features
    imu_cadence: parseFloat(document.getElementById('imu_cadence').value),
    imu_stride_time: parseFloat(document.getElementById('imu_stride_time').value),
    imu_gait_speed: parseFloat(document.getElementById('imu_gait_speed').value),
    imu_stride_asymmetry: parseFloat(document.getElementById('imu_stride_asymmetry').value),
    imu_stride_time_variability: parseFloat(document.getElementById('imu_stride_time_variability').value),
    imu_stance_phase_asymmetry: parseFloat(document.getElementById('imu_stance_phase_asymmetry').value),
    imu_acceleration_rms: parseFloat(document.getElementById('imu_acceleration_rms').value),
    imu_angular_velocity_mean: parseFloat(document.getElementById('imu_angular_velocity_mean').value),

    // sEMG features
    semg_vm_rms: parseFloat(document.getElementById('semg_vm_rms').value),
    semg_vl_rms: parseFloat(document.getElementById('semg_vl_rms').value),
    semg_vm_vl_rms_ratio: parseFloat(document.getElementById('semg_vm_vl_rms_ratio').value),
    semg_vm_median_frequency: parseFloat(document.getElementById('semg_vm_median_frequency').value),
    semg_vl_median_frequency: parseFloat(document.getElementById('semg_vl_median_frequency').value),
    semg_hq_ratio: parseFloat(document.getElementById('semg_hq_ratio').value),
    semg_rf_rms: parseFloat(document.getElementById('semg_rf_rms').value),
    semg_bf_rms: parseFloat(document.getElementById('semg_bf_rms').value)
  };

  return data;
}

function setFormData(data) {
  if (data.age !== undefined && data.age !== null) document.getElementById('age').value = Math.round(data.age);
  if (data.gender !== undefined && data.gender !== null) {
    document.getElementById('gender').value = data.gender;
    const radios = document.getElementsByName('gender-rad');
    radios.forEach(r => { r.checked = (parseInt(r.value) === parseInt(data.gender)); });
  }
  if (data.bmi !== undefined && data.bmi !== null) {
    const b = parseFloat(data.bmi);
    document.getElementById('bmi').value = b.toFixed(1);
    updateBMIPill(b);
  }

  // Modality presence
  const imuChk = document.getElementById('toggle-has-imu');
  const semgChk = document.getElementById('toggle-has-semg');
  imuChk.checked = (data.has_imu === 1 || data.has_imu === true);
  semgChk.checked = (data.has_semg === 1 || data.has_semg === true);
  toggleModality('imu');
  toggleModality('semg');

  if (data.imu_quality !== undefined) {
    document.getElementById('imu_quality').value = data.imu_quality;
    updateQualityText('imu');
  }
  if (data.semg_quality !== undefined) {
    document.getElementById('semg_quality').value = data.semg_quality;
    updateQualityText('semg');
  }

  // Populate numeric fields
  const fields = [
    'imu_cadence', 'imu_stride_time', 'imu_gait_speed', 'imu_stride_asymmetry',
    'imu_stride_time_variability', 'imu_stance_phase_asymmetry', 'imu_acceleration_rms',
    'imu_angular_velocity_mean', 'semg_vm_rms', 'semg_vl_rms', 'semg_vm_vl_rms_ratio',
    'semg_vm_median_frequency', 'semg_vl_median_frequency', 'semg_hq_ratio',
    'semg_rf_rms', 'semg_bf_rms'
  ];

  fields.forEach(f => {
    const el = document.getElementById(f);
    if (el && data[f] !== undefined && data[f] !== null) {
      el.value = parseFloat(data[f]).toFixed(3).replace(/\.?0+$/, '');
    }
  });
}

async function loadPreset(type) {
  try {
    const res = await fetch(`/api/sample?type=${type}`);
    const data = await res.json();
    setFormData(data);
    await submitPrediction();
  } catch (err) {
    console.error('Error loading preset:', err);
  }
}

function resetForm() {
  setFormData({
    age: 62,
    gender: 1,
    bmi: 26.5,
    has_imu: 1,
    has_semg: 1,
    imu_quality: 1.0,
    semg_quality: 1.0,
    imu_cadence: 106.0,
    imu_stride_time: 1.14,
    imu_gait_speed: 1.28,
    imu_stride_asymmetry: 0.035,
    imu_stride_time_variability: 0.028,
    imu_stance_phase_asymmetry: 0.040,
    imu_acceleration_rms: 10.2,
    imu_angular_velocity_mean: 0.54,
    semg_vm_rms: 25.0,
    semg_vl_rms: 24.0,
    semg_vm_vl_rms_ratio: 1.04,
    semg_vm_median_frequency: 70.0,
    semg_vl_median_frequency: 72.0,
    semg_hq_ratio: 0.44,
    semg_rf_rms: 18.0,
    semg_bf_rms: 14.5
  });
  submitPrediction();
}

async function submitPrediction() {
  const payload = getFormData();
  const btn = document.getElementById('btn-predict');
  if (btn) btn.style.opacity = '0.75';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await res.json();
    lastPredictionResult = result;
    renderPrediction(result);
  } catch (err) {
    console.error('Prediction request failed:', err);
  } finally {
    if (btn) btn.style.opacity = '1.0';
  }
}

function renderPrediction(res) {
  if (res.status !== 'success') return;

  const pct = res.risk_percentage;
  const prob = res.probability;

  // 1. Update Radial Gauge
  const gaugeBar = document.getElementById('gauge-bar');
  const riskPctEl = document.getElementById('risk-pct');
  const offset = GAUGE_CIRCUMFERENCE - (pct / 100) * GAUGE_CIRCUMFERENCE;
  gaugeBar.style.strokeDashoffset = offset;

  let gaugeColor = 'var(--status-green)';
  let panelAura = 'rgba(16, 185, 129, 0.22)';
  if (res.risk_class === 'moderate') {
    gaugeColor = 'var(--status-yellow)';
    panelAura = 'rgba(245, 158, 11, 0.22)';
  } else if (res.risk_class === 'koa') {
    gaugeColor = 'var(--status-red)';
    panelAura = 'rgba(244, 63, 94, 0.25)';
  }
  gaugeBar.style.stroke = gaugeColor;
  riskPctEl.innerText = `${pct.toFixed(1)}%`;
  riskPctEl.style.color = gaugeColor;

  const resultsPanel = document.getElementById('results-panel');
  if (resultsPanel) {
    resultsPanel.style.setProperty('--panel-aura', panelAura);
  }

  // 2. Risk Status Badge
  const badge = document.getElementById('risk-badge');
  const title = document.getElementById('risk-title');
  badge.className = `risk-status-badge ${res.risk_class}`;
  title.innerText = res.risk_level;

  // 3. Diagnostic Metrics Grid
  document.getElementById('metric-class').innerText = res.prediction === 1 ? 'Positive KOA' : 'Negative (Healthy)';
  const conf = Math.abs(prob - 0.5) * 2;
  document.getElementById('metric-conf').innerText = `${Math.round(conf * 100)}% Certainty`;

  const mods = [];
  if (res.modalities_used.has_imu) mods.push('IMU');
  if (res.modalities_used.has_semg) mods.push('sEMG');
  document.getElementById('metric-mods').innerText = mods.length > 0 ? mods.join(' + ') : 'None (Imputed)';
  document.getElementById('metric-latency').innerText = `${res.latency_ms} ms`;

  // 4. Timestamp
  const now = new Date();
  document.getElementById('prediction-time').innerText = `Evaluated at ${now.toLocaleTimeString()}`;

  // 5. Biomarkers List
  const list = document.getElementById('biomarkers-list');
  const countBadge = document.getElementById('biomarker-count');
  if (res.biomarkers && res.biomarkers.length > 0) {
    countBadge.innerText = `${res.biomarkers.length} factors`;
    list.innerHTML = res.biomarkers.map(b => `
      <div class="biomarker-card ${b.impact}">
        <div>
          <div class="b-title">${b.name}</div>
          <div class="b-desc">${b.desc}</div>
        </div>
        <div class="b-val">${b.val}</div>
      </div>
    `).join('');
  } else {
    countBadge.innerText = '0 factors';
    list.innerHTML = `<div class="empty-state">All evaluated kinematic and muscular markers are within healthy physiological limits.</div>`;
  }

  // 6. Clinical Decision Support Text
  const recText = document.getElementById('rec-text');
  if (res.risk_class === 'koa') {
    recText.innerText = 'High probability of unilateral or bilateral Knee Osteoarthritis. Recommend radiographic imaging (K-L grading) and targeted quadriceps strengthening protocol targeting Vastus Medialis co-activation.';
  } else if (res.risk_class === 'moderate') {
    recText.innerText = 'Patient exhibits borderline antalgic asymmetry or mild muscular fatigue. Periodic gait monitoring and physical therapy assessment recommended to prevent acceleration of cartilage wear.';
  } else {
    recText.innerText = 'Normal gait mechanics and balanced quadriceps co-activation detected. No immediate structural osteoarthritis intervention indicated.';
  }
}

// ==========================================================================
// 5. COPY CLINICAL SUMMARY TO CLIPBOARD
// ==========================================================================

function copyDiagnosticReport() {
  if (!lastPredictionResult) return;

  const data = getFormData();
  const summary = `=== ARTHROSCAN CLINICAL ASSESSMENT SUMMARY ===
Evaluation Time: ${new Date().toLocaleString()}
Diagnosis: ${lastPredictionResult.prediction === 1 ? 'Positive Knee Osteoarthritis' : 'Healthy Control / Negative'}
Estimated KOA Risk: ${lastPredictionResult.risk_percentage}% (${lastPredictionResult.risk_level})
Model Confidence: ${Math.round(Math.abs(lastPredictionResult.probability - 0.5) * 200)}%
Sensors Used: ${data.has_imu ? 'IMU (Quality ' + Math.round(data.imu_quality*100) + '%)' : 'No IMU'} | ${data.has_semg ? 'sEMG (Quality ' + Math.round(data.semg_quality*100) + '%)' : 'No sEMG'}

Patient Profile:
- Age: ${data.age} yrs | Sex: ${data.gender === 1 ? 'Male' : 'Female'} | BMI: ${data.bmi} kg/m²
- Cadence: ${data.imu_cadence || 'N/A'} spm | Gait Speed: ${data.imu_gait_speed || 'N/A'} m/s | Asymmetry: ${data.imu_stride_asymmetry ? (data.imu_stride_asymmetry*100).toFixed(1)+'%' : 'N/A'}
- VM/VL Ratio: ${data.semg_vm_vl_rms_ratio || 'N/A'} | VM Median Freq: ${data.semg_vm_median_frequency || 'N/A'} Hz

Recommendation:
${document.getElementById('rec-text').innerText}
==============================================`;

  navigator.clipboard.writeText(summary).then(() => {
    const btnText = document.getElementById('copy-btn-text');
    btnText.innerText = 'Copied!';
    setTimeout(() => { btnText.innerText = 'Copy Summary'; }, 2000);
  }).catch(err => {
    console.error('Clipboard copy failed:', err);
  });
}

// Initialization on DOM Load
window.addEventListener('DOMContentLoaded', () => {
  initTheme();
  const initialBMI = parseFloat(document.getElementById('bmi').value) || 28.4;
  updateBMIPill(initialBMI);
  submitPrediction();
});
