const MAX_AGE = 216;
let serverBase = "http://127.0.0.1:3000";

const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput"),
  uploadZone = $("uploadZone");
const previewWrap = $("previewWrap"),
  previewImg = $("previewImg");
const removeBtn = $("removeBtn"),
  analyzeBtn = $("analyzeBtn");
const spinner = $("spinner"),
  btnIcon = $("btnIcon");
const btnText = $("btnText"),
  errorMsg = $("errorMsg");
const endpointInput = $("endpointInput"),
  testBtn = $("testBtn");
const statusDot = $("statusDot"),
  statusLabel = $("statusLabel");
const resultPlaceholder = $("resultPlaceholder"),
  resultContent = $("resultContent");
const resultMonths = $("resultMonths"),
  resultYears = $("resultYears");
const rowMonths = $("rowMonths"),
  rowYears = $("rowYears");
const rowStage = $("rowStage"),
  rowSex = $("rowSex");
const percentileFill = $("percentileFill"),
  percentilePct = $("percentilePct");
const noteSection = $("noteSection"),
  noteBox = $("noteBox");
const stages = document.querySelectorAll(".age-stage");
const steps = [1, 2, 3, 4].map((i) => $("step" + i));

function setStep(n) {
  steps.forEach((s, i) => {
    s.classList.remove("done", "active");
    if (i < n) s.classList.add("done");
    else if (i === n) s.classList.add("active");
  });
}
setStep(0);

/* ── Server health ──────────────────────────────────────── */
async function checkHealth(base) {
  try {
    const r = await fetch(base + "/health", {
      signal: AbortSignal.timeout(4000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

async function updateStatus() {
  serverBase = endpointInput.value.trim().replace(/\/$/, "");
  statusDot.className = "status-dot";
  statusLabel.textContent = "Checking…";
  const ok = await checkHealth(serverBase);
  statusDot.classList.add(ok ? "online" : "offline");
  statusLabel.textContent = ok ? "Server online" : "Server offline";
}

testBtn.addEventListener("click", updateStatus);
endpointInput.addEventListener("change", updateStatus);
updateStatus();

/* ── File handling ──────────────────────────────────────── */
let currentFile = null;

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  currentFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewWrap.style.display = "block";
  analyzeBtn.disabled = false;
  errorMsg.style.display = "none";
  setStep(1);
}

fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () =>
  uploadZone.classList.remove("drag-over"),
);
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  handleFile(e.dataTransfer.files[0]);
});

removeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  currentFile = null;
  previewImg.src = "";
  previewWrap.style.display = "none";
  fileInput.value = "";
  analyzeBtn.disabled = true;
  resultPlaceholder.style.display = "";
  resultContent.style.display = "none";
  noteSection.style.display = "none";
  errorMsg.style.display = "none";
  stages.forEach((s) => s.classList.remove("active"));
  setStep(0);
});

/* ── Helpers ────────────────────────────────────────────── */
function highlightStage(m) {
  stages.forEach((s) =>
    s.classList.toggle("active", m >= +s.dataset.min && m <= +s.dataset.max),
  );
}

/* ── Analyse → POST multipart to Flask /predict ─────────── */
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  const sexVal = document.querySelector('input[name="sex"]:checked').value;
  serverBase = endpointInput.value.trim().replace(/\/$/, "");

  analyzeBtn.disabled = true;
  spinner.style.display = "block";
  btnIcon.style.display = "none";
  btnText.textContent = "Analysing…";
  errorMsg.style.display = "none";
  setStep(2);

  try {
    /* ── Build multipart form — what app.py expects ──────── */
    const form = new FormData();
    form.append("image", currentFile); // file field named "image"
    form.append("sex", sexVal); // text field: "male" | "female"

    setStep(3);

    const response = await fetch(serverBase + "/predict", {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Server returned ${response.status}`);
    }

    /*
        app.py returns JSON:
        { predicted_months, years, months_remainder, age_label,
          stage, sex, percentile, confidence_note }
      */
    const d = await response.json();
    const months = Math.round(d.predicted_months);
    const pct = d.percentile ?? Math.round((months / MAX_AGE) * 100);

    resultPlaceholder.style.display = "none";
    resultContent.style.display = "block";
    resultContent.classList.add("fade-in");

    resultMonths.innerHTML = `${months} <span>mo</span>`;
    resultYears.textContent = d.age_label;
    rowMonths.textContent = `${months} months`;
    rowYears.textContent = d.age_label;
    rowStage.textContent = d.stage;
    rowSex.textContent = sexVal.charAt(0).toUpperCase() + sexVal.slice(1);

    percentilePct.textContent = `${pct}%`;
    setTimeout(() => {
      percentileFill.style.width = pct + "%";
    }, 50);

    if (d.confidence_note) {
      noteBox.textContent = d.confidence_note;
      noteSection.style.display = "block";
    }

    highlightStage(months);
    setStep(4);
  } catch (err) {
    let msg = err.message || "Unknown error.";
    if (/fetch|Failed|Network|Load/i.test(msg)) {
      msg = `Cannot reach the server at "${serverBase}". Make sure app.py is running and click Test.`;
    }
    errorMsg.textContent = "⚠ " + msg;
    errorMsg.style.display = "block";
    setStep(1);
  } finally {
    analyzeBtn.disabled = false;
    spinner.style.display = "none";
    btnIcon.style.display = "";
    btnText.textContent = "Analyse X-Ray";
  }
});
