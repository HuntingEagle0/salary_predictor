/**
 * Salary Predictor — Frontend Logic
 */

(function () {
  "use strict";

  const API_BASE = "";

  /* ── Human-readable labels ──────────────────────────────────── */
  const EXP_LABELS = {
    EN: "Entry-level",
    MI: "Mid-level",
    SE: "Senior",
    EX: "Executive",
  };
  const EMP_LABELS = {
    FT: "Full-time",
    PT: "Part-time",
    CT: "Contract",
    FL: "Freelance",
  };
  const SIZE_LABELS = {
    S: "Small (< 50)",
    M: "Medium (50 – 250)",
    L: "Large (> 250)",
  };
  const REMOTE_LABELS = {
    0: "On-site (0 %)",
    50: "Hybrid (50 %)",
    100: "Fully Remote (100 %)",
  };

  /* ── DOM references ─────────────────────────────────────────── */
  const form = document.getElementById("prediction-form");
  const predictBtn = document.getElementById("predict-btn");
  const btnText = predictBtn.querySelector(".btn__text");
  const btnLoader = predictBtn.querySelector(".btn__loader");
  const resultsSection = document.getElementById("results-section");
  const toastEl = document.getElementById("error-toast");
  const toastMsg = document.getElementById("error-message");

  /* ── Helpers ─────────────────────────────────────────────────── */
  function showToast(msg, ms = 4000) {
    toastMsg.textContent = msg;
    toastEl.hidden = false;
    requestAnimationFrame(() => toastEl.classList.add("show"));
    setTimeout(() => {
      toastEl.classList.remove("show");
      setTimeout(() => (toastEl.hidden = true), 350);
    }, ms);
  }

  function populateSelect(id, values, labelMap) {
    const sel = document.getElementById(id);
    if (!sel) return;
    values.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = labelMap ? labelMap[v] || v : v;
      sel.appendChild(opt);
    });
  }

  function formatCurrency(n) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(n);
  }

  function setLoading(loading) {
    predictBtn.disabled = loading;
    btnText.hidden = loading;
    btnLoader.hidden = !loading;
  }

  /* ── Fetch dropdown options ─────────────────────────────────── */
  async function loadOptions() {
    try {
      const res = await fetch(`${API_BASE}/api/options`);
      if (!res.ok) throw new Error("Failed to load options");
      const data = await res.json();

      populateSelect("work_year", data.work_year || []);
      populateSelect("experience_level", data.experience_level || [], EXP_LABELS);
      populateSelect("employment_type", data.employment_type || [], EMP_LABELS);
      populateSelect("job_title", data.job_title || []);
      populateSelect("salary_currency", data.salary_currency || []);
      populateSelect("employee_residence", data.employee_residence || []);
      populateSelect("remote_ratio", data.remote_ratio || [], REMOTE_LABELS);
      populateSelect("company_location", data.company_location || []);
      populateSelect("company_size", data.company_size || [], SIZE_LABELS);
    } catch (err) {
      showToast("Unable to load form options. Is the server running?");
      console.error(err);
    }
  }

  /* ── Submit prediction ──────────────────────────────────────── */
  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    resultsSection.hidden = true;

    const fd = new FormData(form);
    const payload = {};
    for (const [key, value] of fd.entries()) {
      payload[key] = value;
    }

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      renderResults(data);
    } catch (err) {
      showToast(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  /* ── Render results ─────────────────────────────────────────── */
  function renderResults(data) {
    // Predicted salary
    document.getElementById("predicted-salary").textContent = formatCurrency(
      data.predicted_salary_usd
    );
    const r2 = data.model_scores?.regression_r2 ?? "—";
    document.getElementById("regression-meta").textContent = `Linear Regression · R² ${r2}`;

    // KNN
    const knnVal = document.getElementById("knn-result");
    knnVal.innerHTML = badgeHtml(data.knn_classification);
    const knnAcc = data.model_scores?.knn_accuracy ?? "—";
    document.getElementById("knn-meta").textContent = `Accuracy: ${knnAcc}`;

    // SVM
    const svmVal = document.getElementById("svm-result");
    svmVal.innerHTML = badgeHtml(data.svm_classification);
    const svmAcc = data.model_scores?.svm_accuracy ?? "—";
    document.getElementById("svm-meta").textContent = `Accuracy: ${svmAcc}`;

    // Median
    document.getElementById("median-value").textContent = formatCurrency(
      data.median_salary_usd
    );

    // Show section
    resultsSection.hidden = false;

    // Re-trigger animations by cloning nodes
    resultsSection.querySelectorAll(".result-card").forEach((card) => {
      card.style.animation = "none";
      void card.offsetHeight;           // force reflow
      card.style.animation = "";
    });

    // Scroll into view
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function badgeHtml(label) {
    const isAbove = label === "Above Median";
    const cls = isAbove ? "badge--above" : "badge--below";
    return `<span class="badge ${cls}">${label}</span>`;
  }

  /* ── Init ────────────────────────────────────────────────────── */
  loadOptions();
  form.addEventListener("submit", handleSubmit);
})();
