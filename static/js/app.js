// ── Navigation ──────────────────────────────────────────────────────────
const titles = {pose:"Live Pose Estimation",bmi:"BMI Calculator",recs:"Exercise Recommendations",nutrition:"Nutrition Plan",rl:"RL Progress Tracker",profile:"User Profile"};
function showPanel(id, btn) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
  document.getElementById("panel-"+id).classList.add("active");
  btn.classList.add("active");
  document.getElementById("page-title").textContent = titles[id];
  if (id==="recs")      loadRecs();
  if (id==="nutrition") loadNutrition();
  if (id==="rl")        loadRL();
}
 
// ── Mode toggle ─────────────────────────────────────────────────────────
function switchMode(mode) {
  document.getElementById("camera-view").style.display = mode==="camera" ? "block" : "none";
  document.getElementById("upload-view").style.display = mode==="upload" ? "block" : "none";
  document.getElementById("tab-camera").classList.toggle("active", mode==="camera");
  document.getElementById("tab-upload").classList.toggle("active", mode==="upload");
  if (mode==="camera" && cameraRunning) stopCamera();
}
 
// ── Live camera ─────────────────────────────────────────────────────────
let currentExercise = "squat";
let cameraRunning   = false;
let frameInterval   = null;
 
async function startCamera() {
  try {
    // Check if mediaDevices API is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera API not available. Ensure you're using HTTPS or localhost.");
    }
    
    document.getElementById("cam-status").textContent = "Requesting camera permission...";
    const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:"user"}, audio:false});
    const video  = document.getElementById("webcam");
    video.srcObject = stream;
    await video.play();
    cameraRunning = true;
    document.getElementById("cam-badge").textContent  = "LIVE";
    document.getElementById("cam-badge").style.color  = "#6dfabc";
    document.getElementById("start-cam-btn").textContent = "&#9209; Stop Camera";
    document.getElementById("start-cam-btn").onclick  = stopCamera;
    document.getElementById("cam-status").textContent = "Camera active — analysing every 2s";
    frameInterval = setInterval(sendFrame, 2000);
  } catch(err) {
    document.getElementById("cam-status").textContent = "Camera error: " + err.message;
    document.getElementById("cam-status").style.color = "#fa6d8e";
  }
}
 
function stopCamera() {
  const video = document.getElementById("webcam");
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  cameraRunning = false;
  clearInterval(frameInterval);
  document.getElementById("cam-badge").textContent  = "CAMERA OFF";
  document.getElementById("cam-badge").style.color  = "";
  document.getElementById("start-cam-btn").innerHTML = "&#9654; Start Camera";
  document.getElementById("start-cam-btn").onclick  = startCamera;
  document.getElementById("cam-status").textContent = "Camera stopped";
}
 
function sendFrame() {
  if (!cameraRunning) return;
  const video  = document.getElementById("webcam");
  const canvas = document.getElementById("pose-canvas");
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  submitFrameData(canvas.toDataURL("image/jpeg", 0.7));
}
 
// ── Image upload ────────────────────────────────────────────────────────
let uploadedDataURL = null;

function resetUpload() {
  uploadedDataURL = null;
  document.getElementById("img-input").value = "";
  document.getElementById("upload-preview").style.display = "none";
  document.getElementById("upload-drop").style.display = "block";
  document.getElementById("upload-status").textContent = "Select a new image.";
}
 
function handleDrop(e) {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadImageFile(file);
}
 
function handleImageUpload(e) {
  const file = e.target.files[0];
  if (file) loadImageFile(file);
}
 
function loadImageFile(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    uploadedDataURL = e.target.result;
    document.getElementById("preview-img").src = uploadedDataURL;
    document.getElementById("upload-preview").style.display = "block";
    document.getElementById("upload-drop").style.display = "none";
    document.getElementById("upload-status").textContent = "Image loaded. Click Analyse Pose.";
  };
  reader.readAsDataURL(file);
}
 
function analyseUploadedImage() {
  if (!uploadedDataURL) return;
  const btn = document.getElementById("analyse-btn");
  btn.disabled = true;
  btn.textContent = "Analysing...";
  document.getElementById("upload-status").textContent = "Processing pose...";
  document.getElementById("cue-list").innerHTML = '<div class="analysing">Analysing pose...</div>';
 
  // Create a temporary image and wait for it to load
  const tempImg = new Image();
  tempImg.onload = function() {
    const canvas = document.getElementById("pose-canvas");
    canvas.width  = tempImg.naturalWidth  || 640;
    canvas.height = tempImg.naturalHeight || 480;
    canvas.getContext("2d").drawImage(tempImg, 0, 0, canvas.width, canvas.height);
    
    submitFrameData(uploadedDataURL, () => {
      btn.disabled = false;
      btn.textContent = "&#9654; Analyse Pose";
      document.getElementById("upload-status").textContent = "Analysis complete.";
    });
  };
  tempImg.onerror = function() {
    document.getElementById("upload-status").textContent = "Error loading image. Please try uploading again.";
    document.getElementById("upload-status").style.color = "#fa6d8e";
    btn.disabled = false;
    btn.textContent = "&#9654; Analyse Pose";
  };
  tempImg.src = uploadedDataURL;
}
 
// ── Shared: send frame to Flask ─────────────────────────────────────────
function submitFrameData(dataURL, callback) {
  fetch("/api/process_frame", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({frame: dataURL, exercise: currentExercise})
  })
  .then(r => r.json())
  .then(d => { updateFeedbackUI(d); if (callback) callback(); })
  .catch(err => {
    document.getElementById("cue-list").innerHTML = '<div class="cue">Could not reach server.</div>';
    if (callback) callback();
  });
}
 
function updateFeedbackUI(d) {
  const score = d.score || 0;
  document.getElementById("score-val").textContent = score + "%";
  const offset = (2*Math.PI*42) - (score/100)*(2*Math.PI*42);
  const ring = document.getElementById("ring-fg");
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = score>=70 ? "#6dfabc" : score>=40 ? "#7c6dfa" : "#fa6d8e";
  const cues = d.cues || [];
  document.getElementById("cue-list").innerHTML =
    cues.map(c => `<div class="cue ${score>=70?'good':''}">${c}</div>`).join("") ||
    '<div class="cue">No feedback yet.</div>';

  const diffs = d.pose_comparison || [];
  const diffContainer = document.getElementById("pose-diff-list");
  if (diffContainer) {
    diffContainer.innerHTML = diffs.length
      ? diffs.map(item => {
          const msg =
            item.status === "good"
              ? `${item.joint}: good (${item.actual}deg vs ideal ${item.ideal}deg)`
              : `${item.joint}: ${Math.abs(item.difference)}deg ${item.status === "too_open" ? "more open" : "more bent"} than ideal`;
          return `<div class="cue">${msg}</div>`;
        }).join("")
      : '<div class="cue">Pose difference report will appear after analysis.</div>';
  }

  const planStatus = document.getElementById("plan-status");
  if (planStatus) planStatus.textContent = "Photo analysis ready. Click Generate Plan.";
}
 
function switchExercise(ex, btn) {
  document.querySelectorAll(".ex-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentExercise = ex;
  document.getElementById("ex-label").textContent = ex.replace(/_/g," ").toUpperCase();
}

// ── BMI Calculator ─────────────────────────────────────────────────────
function calculateBMI() {
  const height = parseFloat(document.getElementById("bmi-height").value);
  const weight = parseFloat(document.getElementById("bmi-weight").value);
  
  if (!height || !weight || height <= 0 || weight <= 0) {
    alert("Please enter valid height and weight values");
    return;
  }
  
  // Calculate BMI: weight(kg) / (height(m))^2
  const heightInMeters = height / 100;
  const bmi = weight / (heightInMeters * heightInMeters);
  
  let category, categoryColor, description;
  
  if (bmi < 18.5) {
    category = "Underweight";
    categoryColor = "var(--muted)";
    description = "You may want to gain weight. Consult with a healthcare provider for personalized advice.";
  } else if (bmi < 25) {
    category = "Normal Weight";
    categoryColor = "var(--accent3)";
    description = "Your BMI is in the healthy range. Keep up with regular exercise and balanced nutrition!";
  } else if (bmi < 30) {
    category = "Overweight";
    categoryColor = "#f9c74f";
    description = "You may benefit from increased physical activity and dietary adjustments. Consider consulting a fitness coach.";
  } else {
    category = "Obese";
    categoryColor = "var(--accent2)";
    description = "It's recommended to consult with a healthcare provider and consider a structured fitness program.";
  }
  
  // Update result display
  document.getElementById("bmi-value").textContent = bmi.toFixed(1);
  document.getElementById("bmi-value").style.color = categoryColor;
  document.getElementById("bmi-category").textContent = category;
  document.getElementById("bmi-category").style.color = categoryColor;
  document.getElementById("bmi-description").textContent = description;
  document.getElementById("bmi-result").style.display = "block";
  
  // Also populate the profile and pose panel BMI fields
  document.getElementById("p-bmi").value = bmi.toFixed(1);
  document.getElementById("bmi-input").value = bmi.toFixed(1);
}

function generatePersonalizedPlan() {
  const bmiText = document.getElementById("bmi-input").value.trim();
  const status = document.getElementById("plan-status");
  const output = document.getElementById("plan-output");
  if (status) status.textContent = "Generating personalized recommendations...";
  if (output) output.innerHTML = "";

  fetch("/api/personalized_plan", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ bmi: bmiText === "" ? null : +bmiText })
  })
  .then(async (r) => {
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Failed to generate plan.");
    return data;
  })
  .then((plan) => {
    if (status) status.textContent = "Personalized plan generated.";
    if (!output) return;
    output.innerHTML =
      `<div class="plan-tags">
         <span class="plan-tag">BMI ${plan.bmi}</span>
         <span class="plan-tag">${plan.bmi_category.toUpperCase()}</span>
         <span class="plan-tag">FORM ${plan.pose_score}%</span>
       </div>
       <div class="card-title" style="margin:10px 0 6px 0;">Diet focus</div>
       <div class="plan-list">
         ${plan.diet_recommendations.map(item => `<div class="plan-item">${item}</div>`).join("")}
       </div>
       <div class="card-title" style="margin:14px 0 6px 0;">Exercise focus</div>
       <div class="plan-list">
         ${plan.exercise_recommendations.map(item => `<div class="plan-item">${item}</div>`).join("")}
       </div>`;
  })
  .catch((err) => {
    if (status) status.textContent = "Could not generate plan.";
    if (output) output.innerHTML = `<div class="cue">${err.message}</div>`;
  });
}

function estimateBodyFat() {
  const payload = {
    age: +(document.getElementById("p-age")?.value || 0),
    gender: document.getElementById("p-gender")?.value || "male",
    weight: +(document.getElementById("p-weight")?.value || 0),
    height: +(document.getElementById("p-height")?.value || 0),
    bmi: document.getElementById("bmi-input")?.value ? +document.getElementById("bmi-input").value : null
  };

  fetch("/api/body_fat_estimate", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  })
  .then(async (r) => {
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Unable to estimate body fat.");
    return data;
  })
  .then((data) => {
    const value = document.getElementById("bodyfat-value");
    const meta = document.getElementById("bodyfat-meta");
    const note = document.getElementById("bodyfat-note");
    if (value) value.textContent = `${data.body_fat_percent}%`;
    if (meta) meta.textContent = `Category: ${data.category} | BMI used: ${data.bmi_used} | Pose score: ${data.pose_score_used}%`;
    if (note) {
      note.style.display = "block";
      note.textContent = data.disclaimer || "Experimental estimate.";
    }
  })
  .catch((err) => {
    const meta = document.getElementById("bodyfat-meta");
    if (meta) meta.textContent = err.message;
  });
}

// ── Recommendations ─────────────────────────────────────────────────────
let recsLoaded = false;
function loadRecs() {
  if (recsLoaded) return;
  fetch("/api/recommendations").then(r=>r.json()).then(data=>{
    recsLoaded = true;
    document.getElementById("rec-container").innerHTML =
      '<div class="rec-list">' +
      data.map((r,i)=>`
        <div class="rec-item">
          <span class="rec-rank">${i+1}</span>
          <span class="rec-name">${r.name}</span>
          <div class="rec-bar-wrap"><div class="rec-bar" style="width:${Math.round(r.score*100)}%"></div></div>
          <span class="rec-score">${(r.score*100).toFixed(0)}%</span>
        </div>`).join("") + '</div>';
  });
}

// ── Nutrition ────────────────────────────────────────────────────────────
let nutritionLoaded = false;
function loadNutrition(force=false, pref=null) {
  if (nutritionLoaded && !force) return;
  const query = pref ? `?diet_preference=${encodeURIComponent(pref)}` : "";
  fetch(`/api/nutrition${query}`).then(r=>r.json()).then(d=>{
    nutritionLoaded = true;
    const m = d.macros;
    const dietPref = d.diet_preference || "non_veg";
    const dietLabel = dietPref === "veg" ? "Vegetarian" : "Non-vegetarian";
    document.getElementById("nutrition-container").innerHTML =
      `<div class="card">
        <div class="card-title">Diet preference</div>
        <div class="profile-form" style="grid-template-columns:1fr auto;align-items:end;">
          <div class="form-group">
            <label>Choose your diet plan</label>
            <select id="nutrition-diet-pref" onchange="setNutritionDietPreference()">
              <option value="veg" ${dietPref==="veg" ? "selected" : ""}>Vegetarian</option>
              <option value="non_veg" ${dietPref==="non_veg" ? "selected" : ""}>Non-vegetarian</option>
            </select>
          </div>
          <div class="cam-status" style="margin:0;">Current: ${dietLabel}</div>
        </div>
      </div>
      <div class="card" style="margin-top:1rem"><div class="card-title">Daily macro targets</div>
        <div class="macro-grid">
          <div class="macro-card"><div class="macro-val cal">${Math.round(m.calories)}</div><div class="macro-label">Calories</div></div>
          <div class="macro-card"><div class="macro-val prot">${Math.round(m.protein_g)}g</div><div class="macro-label">Protein</div></div>
          <div class="macro-card"><div class="macro-val carb">${Math.round(m.carbs_g)}g</div><div class="macro-label">Carbs</div></div>
          <div class="macro-card"><div class="macro-val fat">${Math.round(m.fat_g)}g</div><div class="macro-label">Fat</div></div>
        </div></div>
       <div class="card" style="margin-top:1rem"><div class="card-title">Meal plan</div>
        <div class="meal-list">` +
      d.meals.map(meal=>`
        <div class="meal-item">
          <div class="meal-header"><span class="meal-name">${meal.name}</span><span class="meal-kcal">${meal.calories} kcal</span></div>
          <div class="meal-foods">${meal.foods.join(", ")}</div>
          <div class="meal-macros">
            <span class="meal-macro" style="color:#7c6dfa">${meal.protein_g}g <span>protein</span></span>
            <span class="meal-macro" style="color:#f9c74f">${meal.carbs_g}g <span>carbs</span></span>
            <span class="meal-macro" style="color:#6dfabc">${meal.fat_g}g <span>fat</span></span>
          </div>
        </div>`).join("") +
      `</div>${d.notes && d.notes.length ? `<div style="margin-top:1rem;padding:12px 14px;background:rgba(250,109,142,0.08);border-left:3px solid var(--accent2);border-radius:0 8px 8px 0;font-size:0.825rem;color:var(--muted);">${d.notes.join("<br>")}</div>` : ""}</div>`;
  });
}

function setNutritionDietPreference() {
  const pref = document.getElementById("nutrition-diet-pref").value;
  loadNutrition(true, pref);
}

// ── RL Progress ──────────────────────────────────────────────────────────
let rlLoaded = false;
function loadRL() {
  if (rlLoaded) return;
  fetch("/api/rl_progress").then(r=>r.json()).then(data=>{
    rlLoaded = true;
    const last = data[data.length-1];
    document.getElementById("stat-sessions").textContent  = data.length;
    document.getElementById("stat-form").textContent      = (last.form*100).toFixed(0)+"%";
    document.getElementById("stat-fitness").textContent   = (last.fitness*100).toFixed(0)+"%";
    new Chart(document.getElementById("rl-chart").getContext("2d"), {
      type:"line",
      data:{
        labels:data.map(d=>d.episode),
        datasets:[
          {label:"Form %",    data:data.map(d=>+(d.form*100).toFixed(1)),   borderColor:"#6dfabc",backgroundColor:"transparent",tension:0.4,pointRadius:0,borderWidth:2},
          {label:"Fitness %", data:data.map(d=>+(d.fitness*100).toFixed(1)),borderColor:"#fa6d8e",backgroundColor:"transparent",tension:0.4,pointRadius:0,borderWidth:2},
          {label:"Reward",    data:data.map(d=>d.reward),                   borderColor:"#7c6dfa",backgroundColor:"transparent",tension:0.4,pointRadius:0,borderWidth:1.5,borderDash:[4,3]},
        ]
      },
      options:{responsive:true,maintainAspectRatio:false,
        plugins:{legend:{labels:{color:"#6b6a7e",font:{family:"DM Mono",size:11}}}},
        scales:{
          x:{grid:{color:"rgba(255,255,255,0.04)"},ticks:{color:"#6b6a7e",font:{family:"DM Mono",size:10}}},
          y:{grid:{color:"rgba(255,255,255,0.04)"},ticks:{color:"#6b6a7e",font:{family:"DM Mono",size:10}}}
        }
      }
    });
  });
}

// ── Profile ──────────────────────────────────────────────────────────────
function saveProfile() {
  fetch("/api/update_profile",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      name:document.getElementById("p-name").value,
      age:+document.getElementById("p-age").value,
      weight:+document.getElementById("p-weight").value,
      height:+document.getElementById("p-height").value,
      bmi:document.getElementById("p-bmi").value ? +document.getElementById("p-bmi").value : null,
      gender:document.getElementById("p-gender").value,
      activity:document.getElementById("p-activity").value,
      goal:document.getElementById("p-goal").value,
      fitness_level:document.getElementById("p-fitness").value,
      diet_preference:document.getElementById("p-diet") ? document.getElementById("p-diet").value : undefined,
    })
  }).then(async (r)=>{
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Profile update failed");
    return data;
  }).then(()=>{
    document.getElementById("bmi-input").value = document.getElementById("p-bmi").value;
    const ok=document.getElementById("save-ok");
    ok.style.display="block";
    nutritionLoaded=false;recsLoaded=false;rlLoaded=false;
    setTimeout(()=>ok.style.display="none",2500);
    if (typeof disableProfileEdit === "function") disableProfileEdit();
  }).catch((err)=>{
    const ok=document.getElementById("save-ok");
    ok.style.display="block";
    ok.style.color="#fa6d8e";
    ok.textContent = err.message;
    setTimeout(()=>{
      ok.style.display="none";
      ok.style.color="";
      ok.textContent="✓ Profile updated";
    },2500);
  });
}

function setProfileInputsDisabled(disabled) {
  document.querySelectorAll("#panel-profile input, #panel-profile select").forEach(el => {
    el.disabled = disabled;
  });
}

function enableProfileEdit() {
  setProfileInputsDisabled(false);
  const editBtn = document.getElementById("edit-btn");
  const saveBtn = document.getElementById("save-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  if (editBtn) editBtn.style.display = "none";
  if (saveBtn) saveBtn.style.display = "inline-block";
  if (cancelBtn) cancelBtn.style.display = "inline-block";
}

function disableProfileEdit() {
  setProfileInputsDisabled(true);
  const editBtn = document.getElementById("edit-btn");
  const saveBtn = document.getElementById("save-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  if (editBtn) editBtn.style.display = "inline-block";
  if (saveBtn) saveBtn.style.display = "none";
  if (cancelBtn) cancelBtn.style.display = "none";
}

disableProfileEdit();

