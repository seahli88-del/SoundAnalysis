const state = {
  listening: false,
  audioContext: null,
  analyser: null,
  stream: null,
  animationId: null,
  lastProfile: null,
  lastLogTime: 0,
  events: [],
};

const elements = {
  recordButton: document.getElementById("recordButton"),
  resetButton: document.getElementById("resetButton"),
  exportButton: document.getElementById("exportButton"),
  errorBox: document.getElementById("errorBox"),
  canvas: document.getElementById("spectrumCanvas"),
  frequency: document.getElementById("frequencyReadout"),
  level: document.getElementById("levelValue"),
  meter: document.getElementById("meterFill"),
  banner: document.getElementById("alertBanner"),
  threshold: document.getElementById("thresholdSlider"),
  thresholdValue: document.getElementById("thresholdValue"),
  classification: document.getElementById("classification"),
  confidence: document.getElementById("confidenceReadout"),
  scores: document.getElementById("scoreList"),
  eventCount: document.getElementById("eventCount"),
  alertCount: document.getElementById("alertCount"),
  dominant: document.getElementById("dominantLabel"),
  log: document.getElementById("eventLog"),
  logStatus: document.getElementById("logStatus"),
};

const canvasContext = elements.canvas.getContext("2d");

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.hidden = false;
}

function hideError() {
  elements.errorBox.hidden = true;
}

function dbfs(rms) {
  return rms > 0 ? Math.max(-80, 20 * Math.log10(rms)) : -80;
}

function classify(features) {
  const { low, mid, high, centroid, level } = features;
  const total = low + mid + high || 1;
  const lowRatio = low / total;
  const midRatio = mid / total;
  const highRatio = high / total;

  if (level < -58) return { label: "Quiet / room tone", icon: "◌", confidence: 0.94, scores: [["Quiet / room tone", 0.94], ["Silence", 0.82], ["Ambient noise", 0.18]] };
  if (lowRatio > 0.52 && centroid < 500) return { label: "Bass-heavy sound", icon: "◒", confidence: Math.min(0.96, 0.62 + lowRatio / 2), scores: [["Bass-heavy sound", 0.88], ["Mechanical hum", 0.42], ["Music-like", 0.31]] };
  if (highRatio > 0.48 && centroid > 2400) return { label: "Bright / noisy sound", icon: "✳", confidence: Math.min(0.94, 0.55 + highRatio / 2), scores: [["Bright / noisy sound", 0.84], ["Clap / impact", 0.48], ["Static-like", 0.35]] };
  if (midRatio > 0.40 && centroid >= 500 && centroid < 2400) return { label: "Speech-like sound", icon: "◖", confidence: Math.min(0.91, 0.58 + midRatio / 2), scores: [["Speech-like sound", 0.81], ["Vocal / talk", 0.56], ["Music-like", 0.38]] };
  return { label: "Music-like sound", icon: "♪", confidence: 0.63, scores: [["Music-like sound", 0.63], ["Ambient noise", 0.44], ["Mixed sound", 0.39]] };
}

function analyze(buffer, sampleRate) {
  let sum = 0;
  let weightedFrequency = 0;
  let magnitudeTotal = 0;
  const quarter = Math.max(1, Math.floor(buffer.length / 4));
  let low = 0; let mid = 0; let high = 0;
  for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
  for (let i = 1; i < buffer.length; i += 1) {
    const frequency = i * sampleRate / (buffer.length * 2);
    const magnitude = buffer[i];
    weightedFrequency += frequency * magnitude;
    magnitudeTotal += magnitude;
    if (i < quarter) low += magnitude;
    else if (i < quarter * 2.5) mid += magnitude;
    else high += magnitude;
  }
  const rms = Math.sqrt(sum / buffer.length);
  const level = dbfs(rms);
  const centroid = magnitudeTotal ? weightedFrequency / magnitudeTotal : 0;
  return { rms, level, centroid, low, mid, high, dominantFrequency: centroid };
}

function drawSpectrum(buffer) {
  const width = elements.canvas.width;
  const height = elements.canvas.height;
  canvasContext.clearRect(0, 0, width, height);
  const gradient = canvasContext.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#f08a5d");
  gradient.addColorStop(0.52, "#f5cd65");
  gradient.addColorStop(1, "#65b8a6");
  canvasContext.strokeStyle = gradient;
  canvasContext.lineWidth = 3;
  canvasContext.beginPath();
  const visible = Math.min(buffer.length, 180);
  for (let i = 0; i < visible; i += 1) {
    const x = i / (visible - 1) * width;
    const y = height - (buffer[i] / 255) * (height * 0.88) - 8;
    if (i === 0) canvasContext.moveTo(x, y); else canvasContext.lineTo(x, y);
  }
  canvasContext.stroke();
  canvasContext.strokeStyle = "rgba(242, 232, 216, .1)";
  canvasContext.lineWidth = 1;
  for (let y = 40; y < height; y += 54) { canvasContext.beginPath(); canvasContext.moveTo(0, y); canvasContext.lineTo(width, y); canvasContext.stroke(); }
}

function render(features, result) {
  const threshold = Number(elements.threshold.value);
  const alert = features.level >= threshold;
  elements.frequency.textContent = `${Math.round(features.dominantFrequency)} Hz`;
  elements.level.textContent = `${features.level.toFixed(1)} dBFS`;
  elements.meter.style.width = `${Math.max(0, Math.min(100, (features.level + 80) / 80 * 100))}%`;
  elements.banner.textContent = alert ? `Loud sound detected — ${features.level.toFixed(1)} dBFS` : `Level normal — ${features.level.toFixed(1)} dBFS`;
  elements.banner.className = `alert-banner ${alert ? "alert" : "normal"}`;
  elements.classification.innerHTML = `<span class="classification-icon">${result.icon}</span><div><strong>${result.label}</strong><p>Feature profile from the current microphone window.</p></div>`;
  elements.confidence.textContent = `${Math.round(result.confidence * 100)}% confidence`;
  elements.scores.innerHTML = result.scores.map(([label, score]) => `<div class="score-row"><span>${label}</span><div class="score-track"><i style="width:${score * 100}%"></i></div><b>${Math.round(score * 100)}%</b></div>`).join("");
  return alert;
}

function maybeLog(result, features, alert) {
  const now = Date.now();
  if (now - state.lastLogTime < 1800 || (result.label === state.lastProfile && !alert)) return;
  state.lastLogTime = now;
  state.lastProfile = result.label;
  state.events.unshift({ time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }), label: result.label, confidence: result.confidence, level: features.level, alert });
  renderLog();
}

function renderLog() {
  elements.eventCount.textContent = state.events.length;
  elements.alertCount.textContent = state.events.filter((event) => event.alert).length;
  elements.dominant.textContent = state.events[0]?.label?.replace(" sound", "") || "--";
  elements.logStatus.textContent = state.events.length ? `${state.events.length} recorded` : "No events yet";
  elements.log.innerHTML = state.events.length ? state.events.map((event) => `<tr><td class="mono">${event.time}</td><td>${event.label}</td><td>${Math.round(event.confidence * 100)}%</td><td class="mono">${event.level.toFixed(1)} dBFS</td><td>${event.alert ? "⚠ Alert" : "Normal"}</td></tr>`).join("") : `<tr class="empty-row"><td colspan="5">Start listening to populate the event log.</td></tr>`;
}

function frame() {
  if (!state.listening) return;
  const spectrum = new Uint8Array(state.analyser.frequencyBinCount);
  const waveform = new Float32Array(state.analyser.fftSize);
  state.analyser.getByteFrequencyData(spectrum);
  state.analyser.getFloatTimeDomainData(waveform);
  const features = analyze(spectrum, state.audioContext.sampleRate);
  const result = classify(features);
  const alert = render(features, result);
  drawSpectrum(spectrum);
  maybeLog(result, features, alert);
  state.animationId = requestAnimationFrame(frame);
}

async function startListening() {
  hideError();
  if (!navigator.mediaDevices?.getUserMedia) { showError("Microphone access is unavailable. Open this Space over HTTPS and use a modern browser."); return; }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = state.audioContext.createMediaStreamSource(state.stream);
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 2048;
    state.analyser.smoothingTimeConstant = 0.72;
    source.connect(state.analyser);
    state.listening = true;
    elements.recordButton.textContent = "Stop listening";
    elements.recordButton.classList.add("active");
    frame();
  } catch (error) {
    showError(`Microphone access failed: ${error.name === "NotAllowedError" ? "permission was denied" : error.message}. Check your browser permission and try again.`);
  }
}

function stopListening() {
  state.listening = false;
  cancelAnimationFrame(state.animationId);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.audioContext?.close();
  state.stream = null;
  state.audioContext = null;
  elements.recordButton.textContent = "Start listening";
  elements.recordButton.classList.remove("active");
}

function exportLog() {
  if (!state.events.length) { showError("There are no events to export yet."); return; }
  const header = "time,profile,confidence,level_dbfs,alert\n";
  const body = state.events.map((event) => [event.time, event.label, event.confidence.toFixed(3), event.level.toFixed(1), event.alert].map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([header + body], { type: "text/csv" }));
  link.download = "sound-events.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

elements.recordButton.addEventListener("click", () => (state.listening ? stopListening() : startListening()));
elements.resetButton.addEventListener("click", () => { stopListening(); state.events = []; state.lastProfile = null; state.lastLogTime = 0; renderLog(); elements.scores.innerHTML = ""; elements.classification.innerHTML = '<span class="classification-icon">◌</span><div><strong>Microphone idle</strong><p>Start listening to analyze the incoming signal.</p></div>'; elements.level.textContent = "-80.0 dBFS"; elements.frequency.textContent = "-- Hz"; elements.banner.textContent = "Waiting for microphone permission"; elements.banner.className = "alert-banner normal"; });
elements.exportButton.addEventListener("click", exportLog);
elements.threshold.addEventListener("input", () => { elements.thresholdValue.textContent = `${elements.threshold.value} dBFS`; });
renderLog();
