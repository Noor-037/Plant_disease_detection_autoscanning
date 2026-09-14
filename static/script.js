// ============================================================
// ELEMENT REFS
// ============================================================
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const cameraFrame = document.getElementById('cameraFrame');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const scanOverlay = document.getElementById('scanOverlay');
const captureFlash = document.getElementById('captureFlash');
const scanStatusText = document.getElementById('scanStatusText');
const progressChecklist = document.getElementById('progressChecklist');
const cameraActionsIdle = document.getElementById('cameraActionsIdle');
const cameraActionsScanning = document.getElementById('cameraActionsScanning');

const stateBadge = document.getElementById('stateBadge');
const resultIdle = document.getElementById('resultIdle');
const resultBody = document.getElementById('resultBody');
const capturedThumb = document.getElementById('capturedThumb');
const conditionBanner = document.getElementById('conditionBanner');
const conditionIcon = document.getElementById('conditionIcon');
const conditionText = document.getElementById('conditionText');

const uploadFallbackInput = document.getElementById('uploadFallbackInput');
const uploadFallbackLabel = document.getElementById('uploadFallbackLabel');

let cameraStream = null;
let scanIntervalId = null;
let consecutivePlantFrames = 0;
const REQUIRED_CONSECUTIVE_FRAMES = 2;
let isCapturing = false; // guards against re-triggering while a capture is in flight

// ============================================================
// NAV HELPER
// ============================================================
function goToScanAndStart() {
  document.getElementById('scan').scrollIntoView({ behavior: 'smooth' });
  setTimeout(startAutoScan, 450);
}

// ============================================================
// LIGHTWEIGHT CLIENT-SIDE PLANT PRESENCE CHECK
// ============================================================
// Runs on every sampled frame purely to decide UI status ("Searching..."
// vs "Plant detected!") and when to trigger auto-capture. The SERVER
// re-checks plant presence (and image quality) authoritatively on the
// captured frame before returning any result — this client-side check
// is only ever used to drive the live scanning UI, never to directly
// produce a result.
function quickPlantCheck(ctx, w, h) {
  const data = ctx.getImageData(0, 0, w, h).data;
  let green = 0, unhealthy = 0, total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
    total++;
    if (g > r * 1.05 && g > b * 1.05 && g > 0.2) {
      green++;
    } else if (
      (r > 0.45 && g > 0.45 && b < r * 0.75) ||                         // yellow-ish
      (r >= g && r > 0.05 && r < 0.78 && b < r * 0.85) ||                // brown-ish
      (r > 0.75 && g > 0.75 && b > 0.65 && Math.abs(r - g) < 0.08)       // white-ish
    ) {
      unhealthy++;
    }
  }

  const vegetationPct = ((green + unhealthy) / total) * 100;
  const greenPct = (green / total) * 100;
  return (greenPct >= 5 && vegetationPct >= 25) ||
         (greenPct < 5 && vegetationPct >= 50);
}

// ============================================================
// CAMERA ERROR HANDLING
// ============================================================
function describeCameraError(err) {
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    return {
      icon: '📷', title: 'Camera Permission Required',
      message: 'Please allow camera access to use automatic plant scanning.',
      showUploadFallback: false
    };
  }
  if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    return {
      icon: '📷', title: 'Camera Unavailable',
      message: 'No camera was found on this device.',
      showUploadFallback: true
    };
  }
  if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
    return {
      icon: '📷', title: 'Camera Unavailable',
      message: 'The camera is already in use by another application.',
      showUploadFallback: true
    };
  }
  return {
    icon: '📷', title: 'Camera Unavailable',
    message: 'Unable to access the camera (' + (err.message || err.name) + ').',
    showUploadFallback: true
  };
}

function showCameraErrorModal({ icon, title, message, showUploadFallback }) {
  document.getElementById('cameraErrorIcon').innerText = icon;
  document.getElementById('cameraErrorTitle').innerText = title;
  document.getElementById('cameraErrorMessage').innerText = message;
  uploadFallbackLabel.style.display = showUploadFallback ? 'inline-block' : 'none';
  document.getElementById('cameraErrorModal').style.display = 'flex';
}

function closeCameraErrorModal() {
  document.getElementById('cameraErrorModal').style.display = 'none';
}

// ============================================================
// START / STOP AUTO SCAN
// ============================================================
function startAutoScan() {
  if (!window.isSecureContext) {
    showCameraErrorModal({
      icon: '📷', title: 'Camera Unavailable',
      message: `Live camera access needs https:// or localhost — not a plain ` +
               `http:// address like ${location.host}. Open this site as ` +
               `http://localhost:5000 on the computer running it, or set up HTTPS.`,
      showUploadFallback: true
    });
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraErrorModal({
      icon: '📷', title: 'Camera Unavailable',
      message: 'This browser does not support camera access.',
      showUploadFallback: true
    });
    return;
  }

  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    .then(stream => {
      cameraStream = stream;
      video.srcObject = stream;
      cameraPlaceholder.style.display = 'none';
      scanOverlay.style.display = 'flex';
      cameraActionsIdle.style.display = 'none';
      cameraActionsScanning.style.display = 'flex';
      progressChecklist.style.display = 'none';
      resetChecklist();
      setScanStatus('🔍 Searching for plant...');
      consecutivePlantFrames = 0;
      isCapturing = false;
      scanIntervalId = setInterval(sampleFrame, 500);
    })
    .catch(err => showCameraErrorModal(describeCameraError(err)));
}

function stopAutoScan() {
  if (scanIntervalId) { clearInterval(scanIntervalId); scanIntervalId = null; }
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  scanOverlay.style.display = 'none';
  cameraPlaceholder.style.display = 'flex';
  cameraActionsIdle.style.display = 'flex';
  cameraActionsScanning.style.display = 'none';
  progressChecklist.style.display = 'none';
  setScanStatus('Tap "Start Auto Scan" to begin.');
}

// ============================================================
// FRAME SAMPLING LOOP
// ============================================================
const sampleCanvas = document.createElement('canvas');
sampleCanvas.width = 80; sampleCanvas.height = 60;
const sampleCtx = sampleCanvas.getContext('2d');

function sampleFrame() {
  if (isCapturing || !video.videoWidth) return;

  sampleCtx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
  const plantLikely = quickPlantCheck(sampleCtx, sampleCanvas.width, sampleCanvas.height);

  if (plantLikely) {
    consecutivePlantFrames++;
    if (consecutivePlantFrames >= REQUIRED_CONSECUTIVE_FRAMES) {
      finalizeCapture();
    } else {
      setScanStatus('🌿 Plant detected!');
    }
  } else {
    consecutivePlantFrames = 0;
    setScanStatus('🔍 Searching for plant...');
  }
}

// Manual fallback — bypasses the consecutive-frame requirement
function captureNow() {
  if (isCapturing || !video.videoWidth) return;
  finalizeCapture();
}

function setScanStatus(text) {
  scanStatusText.innerText = text;
}

// ============================================================
// CAPTURE + ANALYZE
// ============================================================
function resetChecklist() {
  ['pcPlant', 'pcCapture', 'pcCondition', 'pcDisease', 'pcResult'].forEach(id => {
    const el = document.getElementById(id);
    el.className = '';
  });
}

function markChecklist(id, state) {
  document.getElementById(id).className = state; // 'pc-active' | 'pc-done'
}

function finalizeCapture() {
  isCapturing = true;
  clearInterval(scanIntervalId);
  scanIntervalId = null;

  // Freeze the frame at full resolution
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  // Visual feedback: flash + status sequence + checklist
  captureFlash.classList.add('flash-active');
  setTimeout(() => captureFlash.classList.remove('flash-active'), 350);

  setScanStatus('🌿 Plant Found');
  progressChecklist.style.display = 'flex';
  resetChecklist();
  markChecklist('pcPlant', 'pc-done');

  setTimeout(() => {
    setScanStatus('📸 Capturing image...');
    markChecklist('pcCapture', 'pc-done');
  }, 250);

  setTimeout(() => {
    setScanStatus('🤖 AI is analyzing your plant...');
    markChecklist('pcCondition', 'pc-active');
    sendCapturedFrame();
  }, 700);
}

function sendCapturedFrame() {
  const plantType = document.getElementById('plant_type').value;

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('plant_type', plantType);
    formData.append('image', blob, 'scan.jpg');

    fetch('/analyze', { method: 'POST', body: formData })
      .then(res => res.json())
      .then(() => pollStatus())
      .catch(() => {
        stopAutoScan();
        isCapturing = false;
      });
  }, 'image/jpeg', 0.9);
}

function pollStatus() {
  fetch('/status')
    .then(res => res.json())
    .then(data => {
      if (data.state === 'detecting') {
        setTimeout(pollStatus, 400);
        return;
      }

      markChecklist('pcCondition', 'pc-done');
      markChecklist('pcDisease', 'pc-done');
      markChecklist('pcResult', 'pc-done');
      setScanStatus('✓ Analysis Complete');

      stateBadge.innerText = data.state === 'done' ? 'Done' : data.state;
      stateBadge.className = 'badge ' + data.state;

      if (data.state === 'no_plant') {
        document.getElementById('noPlantModal').style.display = 'flex';
        return;
      }
      if (data.state === 'unclear') {
        document.getElementById('unclearMessage').innerText = data.message;
        document.getElementById('unclearModal').style.display = 'flex';
        return;
      }
      if (data.state === 'uncertain') {
        document.getElementById('uncertainModal').style.display = 'flex';
        return;
      }
      if (data.state === 'error') {
        stopAutoScan();
        isCapturing = false;
        return;
      }

      renderResult(data);
      loadHistory();
      stopAutoScan();
      isCapturing = false;
    });
}

// After closing a "no plant" / "unclear" / "uncertain" popup, resume scanning
function closeModalAndRescan(modalId) {
  document.getElementById(modalId).style.display = 'none';
  isCapturing = false;
  consecutivePlantFrames = 0;
  resetChecklist();
  progressChecklist.style.display = 'none';

  if (cameraStream) {
    // Camera is still open — just resume the sampling loop
    setScanStatus('🔍 Searching for plant...');
    cameraActionsIdle.style.display = 'none';
    cameraActionsScanning.style.display = 'flex';
    scanIntervalId = setInterval(sampleFrame, 500);
  } else {
    startAutoScan();
  }
}

// ============================================================
// RESULT RENDERING
// ============================================================
function renderResult(data) {
  resultIdle.style.display = 'none';
  resultBody.style.display = 'block';

  capturedThumb.src = canvas.toDataURL('image/jpeg', 0.85);
  capturedThumb.classList.add('shown');

  document.getElementById('plantOut').innerText = data.plant || '—';

  const diseaseRow = document.getElementById('diseaseRow');
  const confidenceBlock = document.getElementById('confidenceBlock');
  const severityRow = document.getElementById('severityRow');
  const treatmentBlock = document.getElementById('treatmentBlock');
  const stepsList = document.getElementById('steps');
  const safetyNote = document.getElementById('safetyNote');
  const pesticideLine = document.getElementById('pesticideLine');

  conditionBanner.className = 'condition-banner';

  if (data.condition === 'Healthy') {
    conditionBanner.classList.add('state-healthy');
    conditionIcon.innerText = '🌿';
    conditionText.innerText = 'Healthy Leaf';
    pesticideLine.innerText = '✓ No disease detected — ' + (data.pesticide || 'No pesticide is required.');
    diseaseRow.style.display = 'none';
    confidenceBlock.style.display = 'none';
    severityRow.style.display = 'none';
    treatmentBlock.style.display = 'none';
    safetyNote.style.display = 'none';

  } else if (data.condition === 'Dry/Dead Leaf') {
    conditionBanner.classList.add('state-drydead');
    conditionIcon.innerText = '🍂';
    conditionText.innerText = 'Dry / Dead Leaf';
    pesticideLine.innerText = '🚫 ' + (data.pesticide || 'No pesticide is required.');
    diseaseRow.style.display = 'none';
    confidenceBlock.style.display = 'none';
    severityRow.style.display = 'none';
    treatmentBlock.style.display = 'none';
    safetyNote.style.display = 'none';

  } else {
    // Disease Detected
    conditionBanner.classList.add('state-disease');
    conditionIcon.innerText = '⚠️';
    conditionText.innerText = 'Disease Detected';
    pesticideLine.innerText = '';

    diseaseRow.style.display = 'flex';
    document.getElementById('diseaseOut').innerText = data.disease || '—';

    confidenceBlock.style.display = 'block';
    document.getElementById('confidenceValue').innerText = (data.confidence || 0) + '%';
    document.getElementById('confidenceFill').style.width = (data.confidence || 0) + '%';

    severityRow.style.display = 'flex';
    const sevBadge = document.getElementById('severityBadge');
    sevBadge.innerText = data.severity || '—';
    sevBadge.className = 'badge sev-' + (data.severity || 'None');

    treatmentBlock.style.display = 'block';
    document.getElementById('pesticideOut').innerText = data.pesticide || '—';
    document.getElementById('dosageOut').innerText = data.dosage || '—';
    document.getElementById('frequencyOut').innerText = data.frequency || '—';

    safetyNote.style.display = 'block';
    safetyNote.innerText = data.safety_note || '';
  }

  stepsList.innerHTML = '';
  (data.steps || []).forEach(s => {
    const li = document.createElement('li');
    li.innerText = s;
    stepsList.appendChild(li);
  });
  stepsList.style.display = (data.steps && data.steps.length) ? 'block' : 'none';
}

function scanAnother() {
  resultIdle.style.display = 'block';
  resultBody.style.display = 'none';
  stateBadge.innerText = 'Idle';
  stateBadge.className = 'badge idle';
  capturedThumb.classList.remove('shown');
  startAutoScan();
  document.getElementById('scan').scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// UPLOAD FALLBACK (only offered when the camera genuinely can't open)
// ============================================================
uploadFallbackInput.addEventListener('change', () => {
  const file = uploadFallbackInput.files[0];
  if (!file) return;
  closeCameraErrorModal();

  isCapturing = true;
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);

    progressChecklist.style.display = 'flex';
    resetChecklist();
    markChecklist('pcPlant', 'pc-done');
    markChecklist('pcCapture', 'pc-done');
    markChecklist('pcCondition', 'pc-active');
    setScanStatus('🤖 AI is analyzing your plant...');
    sendCapturedFrame();
  };
  img.src = URL.createObjectURL(file);
});

// ============================================================
// HISTORY
// ============================================================
function loadHistory() {
  fetch('/history')
    .then(res => res.json())
    .then(items => {
      const list = document.getElementById('historyList');
      if (!items.length) {
        list.innerHTML = '<span id="historyEmpty">No scans yet. Scans with no plant detected or unclear images aren\'t recorded.</span>';
        return;
      }
      list.innerHTML = items.map(item => {
        const icon = item.condition === 'Healthy' ? '🌿' :
                     item.condition === 'Dry/Dead Leaf' ? '🍂' : '⚠️';
        const label = item.disease || item.condition;
        return `
          <div class="history-item">
            <div class="history-left">
              <b>${icon} ${item.plant}</b>
              <span>${label}${item.severity && item.severity !== 'None' ? ' — ' + item.severity : ''}</span>
            </div>
            <div class="history-right">
              ${item.confidence ? item.confidence + '%' : ''}<br>${item.time}
            </div>
          </div>
        `;
      }).join('');
    });
}

loadHistory();