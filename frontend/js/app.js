function predict() {

  const age = ageValue("age");
  const gender = value("gender");
  const admission = value("admission");
  const stay = ageValue("stay");
  const meds = ageValue("meds");
  const prev = ageValue("prev");

  if ([age, gender, admission, stay, meds, prev].includes(null)) {
    alert("Please fill all fields correctly");
    return;
  }

  const data = { age, gender, admission, stay, meds, prev };

  showLoader(true);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  })
  .then(res => res.json())
  .then(result => {
    showLoader(false);
    showResult(result.result);
  })
  .catch(() => {
    showLoader(false);
    alert("Backend not running");
  });
}

function showResult(value) {
  const box = document.getElementById("resultBox");
  const bar = document.getElementById("bar");
  const text = document.getElementById("riskText");
  const percent = document.getElementById("riskValue");

  box.classList.remove("hidden");

  let risk = value === 1 ? 75 : 25;
  bar.style.width = risk + "%";
  percent.innerText = risk + "% Readmission Risk";

  if (risk > 60) {
    bar.style.background = "#d00000";
    text.innerText = "⚠️ High Risk of Readmission";
  } else {
    bar.style.background = "#2a9d8f";
    text.innerText = "✅ Low Risk of Readmission";
  }
}

function showLoader(show) {
  document.getElementById("loader").classList.toggle("hidden", !show);
}

function value(id) {
  const v = document.getElementById(id).value;
  return v === "" ? null : Number(v);
}

function ageValue(id) {
  const v = document.getElementById(id).value;
  return v === "" ? null : parseInt(v);
}