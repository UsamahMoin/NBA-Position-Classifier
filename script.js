const positionNames = {
  C: "Center",
  PF: "Power forward",
  PG: "Point guard",
  SF: "Small forward",
  SG: "Shooting guard",
};

const positionCopy = {
  C: "Interior scoring, rebounding, and rim protection tend to define the center profile.",
  PF: "Power forwards combine frontcourt rebounding with more perimeter scoring and mobility.",
  PG: "Point guards stand out through ball handling, assists, and on-ball creation.",
  SF: "Small forwards occupy the overlap between perimeter skill and frontcourt size.",
  SG: "Shooting guards pair perimeter volume with scoring and secondary playmaking.",
};

const featureNames = {
  MP: "Minutes",
  "FG%": "Field goal %",
  "3P": "3-pointers made",
  "3PA": "3-point attempts",
  "2P%": "2-point %",
  ORB: "Offensive rebounds",
  DRB: "Defensive rebounds",
  TRB: "Total rebounds",
  AST: "Assists",
  BLK: "Blocks",
  TOV: "Turnovers",
  PTS: "Points",
};

const state = {
  data: null,
  player: null,
  values: {},
  profile: "C",
};

const elements = {
  picker: document.querySelector("#player-picker"),
  pickerMenu: document.querySelector("#picker-menu"),
  playerSearch: document.querySelector("#player-search"),
  playerOptions: document.querySelector("#player-options"),
  selectedPlayer: document.querySelector("#selected-player"),
  selectedTeam: document.querySelector("#selected-team"),
  sliderGrid: document.querySelector("#slider-grid"),
  resetPlayer: document.querySelector("#reset-player"),
  predictedPosition: document.querySelector("#predicted-position"),
  predictedName: document.querySelector("#predicted-name"),
  decisionSummary: document.querySelector("#decision-summary"),
  actualPosition: document.querySelector("#actual-position"),
  voteChart: document.querySelector("#vote-chart"),
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function decimalsFor(feature) {
  return feature.includes("%") ? 3 : 1;
}

function formatFeature(feature, value) {
  if (feature.includes("%")) return `${(value * 100).toFixed(1)}%`;
  return Number(value).toFixed(1);
}

function inputStep(feature) {
  return feature.includes("%") ? 0.001 : 0.1;
}

function setRangeFill(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const fill = max === min ? 0 : ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty("--range-fill", `${fill}%`);
}

function classify(values) {
  const { selectedFeatures, scalerMean, scalerScale, model } = state.data;
  const scaled = selectedFeatures.map((feature, index) => (
    (values[feature] - scalerMean[index]) / scalerScale[index]
  ));
  const votes = Object.fromEntries(model.classes.map((label) => [label, 0]));
  const margins = [];

  model.pairLabels.forEach(([first, second], pairIndex) => {
    const margin = model.coefficients[pairIndex].reduce(
      (sum, coefficient, featureIndex) => sum + coefficient * scaled[featureIndex],
      model.intercepts[pairIndex],
    );
    const winner = margin < 0 ? second : first;
    votes[winner] += 1;
    margins.push({ first, second, margin, winner });
  });

  const predicted = model.classes.reduce((best, label) => (
    votes[label] > votes[best] ? label : best
  ), model.classes[0]);

  return { predicted, votes, margins };
}

function renderDecision() {
  const result = classify(state.values);
  const actual = state.player.position;
  const hasChanges = state.data.selectedFeatures.some(
    (feature) => Math.abs(state.values[feature] - state.player.values[feature]) > 0.0001,
  );
  const matches = result.predicted === actual;

  elements.predictedPosition.textContent = result.predicted;
  elements.predictedName.textContent = positionNames[result.predicted];
  elements.actualPosition.textContent = `${actual} / ${positionNames[actual]}`;

  if (hasChanges) {
    elements.decisionSummary.textContent = matches
      ? "The edited profile still agrees with the roster label."
      : `The edited profile now crosses from ${positionNames[actual].toLowerCase()} to ${positionNames[result.predicted].toLowerCase()}.`;
  } else {
    elements.decisionSummary.textContent = matches
      ? "The model agrees with this player's roster label."
      : `The model reads this profile as ${positionNames[result.predicted].toLowerCase()}, showing where roles overlap.`;
  }

  elements.voteChart.innerHTML = state.data.labels.map((label) => `
    <div class="vote-row ${label === result.predicted ? "winner" : ""}">
      <span>${label}</span>
      <div class="vote-track"><div class="vote-fill" style="width:${result.votes[label] * 25}%"></div></div>
      <span>${result.votes[label]}</span>
    </div>
  `).join("");
}

function createSliders() {
  elements.sliderGrid.innerHTML = state.data.selectedFeatures.map((feature) => {
    const stats = state.data.featureStats[feature];
    const value = state.values[feature];
    const id = `feature-${feature.replace(/[^a-z0-9]/gi, "-")}`;
    return `
      <div class="stat-control">
        <header>
          <label for="${id}">${featureNames[feature] || feature}</label>
          <output id="${id}-output" for="${id}">${formatFeature(feature, value)}</output>
        </header>
        <input id="${id}" type="range" data-feature="${feature}" min="${stats.min}" max="${stats.max}" step="${inputStep(feature)}" value="${value}">
        <div class="range-limits"><span>${formatFeature(feature, stats.min)}</span><span>${formatFeature(feature, stats.max)}</span></div>
      </div>
    `;
  }).join("");

  elements.sliderGrid.querySelectorAll('input[type="range"]').forEach((input) => {
    setRangeFill(input);
    input.addEventListener("input", () => {
      const feature = input.dataset.feature;
      state.values[feature] = Number(input.value);
      document.querySelector(`#${CSS.escape(input.id)}-output`).textContent = formatFeature(feature, Number(input.value));
      setRangeFill(input);
      renderDecision();
    });
  });
}

function selectPlayer(player) {
  state.player = player;
  state.values = { ...player.values };
  elements.selectedPlayer.textContent = player.player;
  elements.selectedTeam.textContent = `${player.team} / ${player.position} / ${positionNames[player.position]}`;
  closePicker();
  createSliders();
  renderDecision();
  renderPlayerOptions("");
}

function renderPlayerOptions(query) {
  const normalized = query.trim().toLowerCase();
  const matches = state.data.players.filter((player) => (
    !normalized
    || player.player.toLowerCase().includes(normalized)
    || player.team.toLowerCase().includes(normalized)
    || player.position.toLowerCase() === normalized
  )).slice(0, 80);

  if (!matches.length) {
    elements.playerOptions.innerHTML = '<p class="empty-option">No matching players</p>';
    return;
  }

  elements.playerOptions.innerHTML = matches.map((player) => `
    <button class="player-option" type="button" role="option" aria-selected="${state.player?.player === player.player}" data-player="${player.player.replaceAll('"', "&quot;")}">
      <span>${player.player}</span>
      <span>${player.team} / ${player.position}</span>
    </button>
  `).join("");

  elements.playerOptions.querySelectorAll(".player-option").forEach((option) => {
    option.addEventListener("click", () => {
      const player = state.data.players.find((item) => item.player === option.dataset.player);
      if (player) selectPlayer(player);
    });
  });
}

function openPicker() {
  elements.pickerMenu.hidden = false;
  elements.picker.setAttribute("aria-expanded", "true");
  elements.playerSearch.value = "";
  renderPlayerOptions("");
  requestAnimationFrame(() => elements.playerSearch.focus());
}

function closePicker() {
  elements.pickerMenu.hidden = true;
  elements.picker.setAttribute("aria-expanded", "false");
}

function renderMetrics() {
  const { metrics } = state.data;
  document.querySelector("#player-count").textContent = state.data.playerCount;
  document.querySelector("#feature-count").textContent = state.data.selectedFeatures.length;
  document.querySelector("#test-accuracy").textContent = formatPercent(metrics.testAccuracy);
  document.querySelector("#metric-test").textContent = formatPercent(metrics.testAccuracy);
  document.querySelector("#metric-balanced").textContent = formatPercent(metrics.balancedAccuracy);
  document.querySelector("#metric-cv").textContent = formatPercent(metrics.crossValidationBalancedAccuracy);
  document.querySelector("#metric-training").textContent = formatPercent(metrics.trainingAccuracy);
}

function renderConfusionMatrix() {
  const matrix = state.data.confusionMatrix;
  const maxCell = Math.max(...matrix.flat());
  const parts = ['<div class="matrix-label"></div>'];
  state.data.labels.forEach((label) => {
    parts.push(`<div class="matrix-label">${label}</div>`);
  });
  matrix.forEach((row, rowIndex) => {
    parts.push(`<div class="matrix-label">${state.data.labels[rowIndex]}</div>`);
    row.forEach((value, columnIndex) => {
      const alpha = 0.06 + (value / maxCell) * 0.34;
      parts.push(`<div class="matrix-cell ${rowIndex === columnIndex ? "diagonal" : ""}" style="--cell-alpha:${alpha.toFixed(3)}" title="${value} actual ${state.data.labels[rowIndex]} players predicted as ${state.data.labels[columnIndex]}">${value}</div>`);
    });
  });
  document.querySelector("#confusion-matrix").innerHTML = parts.join("");
}

function renderClassReport() {
  document.querySelector("#class-report").innerHTML = state.data.labels.map((label) => {
    const metrics = state.data.classificationReport[label];
    const rows = [
      ["Precision", metrics.precision],
      ["Recall", metrics.recall],
      ["F1", metrics["f1-score"]],
    ].map(([name, value]) => `
      <div class="report-bar">
        <span>${name}</span>
        <div class="report-track"><div class="report-fill" style="width:${value * 100}%"></div></div>
        <span class="report-value">${value.toFixed(2)}</span>
      </div>
    `).join("");
    return `<div class="report-row"><span class="report-position">${label}</span><div class="report-bars">${rows}</div></div>`;
  }).join("");
}

function renderBalance() {
  const maxCount = Math.max(...Object.values(state.data.balancedClassCounts));
  document.querySelector("#balance-chart").innerHTML = state.data.labels.map((label) => {
    const before = state.data.trainClassCounts[label];
    const after = state.data.balancedClassCounts[label];
    return `
      <div class="balance-row">
        <span>${label}</span>
        <div class="balance-bars">
          <div class="balance-line"><div class="balance-track"><div class="balance-fill before" style="width:${before / maxCount * 100}%"></div></div><span>${before}</span></div>
          <div class="balance-line"><div class="balance-track"><div class="balance-fill after" style="width:${after / maxCount * 100}%"></div></div><span>${after}</span></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderImportance() {
  const maxImportance = state.data.featureImportance[0].importance;
  document.querySelector("#importance-chart").innerHTML = state.data.featureImportance.map(({ feature, importance }) => `
    <div class="importance-row">
      <span>${featureNames[feature] || feature}</span>
      <div class="importance-track"><div class="importance-fill" style="width:${importance / maxImportance * 100}%"></div></div>
      <span class="importance-value">${importance.toFixed(3)}</span>
    </div>
  `).join("");
}

function renderProfile() {
  const profileFeatures = ["PTS", "TRB", "AST", "3PA", "ORB", "BLK"];
  const profile = state.data.positionProfiles[state.profile];
  document.querySelector("#profile-code").textContent = state.profile;
  document.querySelector("#profile-name").textContent = positionNames[state.profile];
  document.querySelector("#profile-copy").textContent = positionCopy[state.profile];

  document.querySelectorAll(".position-tab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.position === state.profile));
  });

  document.querySelector("#profile-chart").innerHTML = profileFeatures.map((feature) => {
    const stats = state.data.featureStats[feature];
    const value = profile[feature];
    const normalized = (value - stats.min) / (stats.max - stats.min) * 100;
    return `
      <div class="profile-row">
        <span class="profile-label">${featureNames[feature]}</span>
        <div class="profile-track"><div class="profile-fill" style="width:${normalized}%"></div></div>
        <span class="profile-value">${formatFeature(feature, value)}</span>
      </div>
    `;
  }).join("");
}

function renderPositionTabs() {
  document.querySelector("#position-tabs").innerHTML = state.data.labels.map((label) => `
    <button class="position-tab" type="button" role="tab" data-position="${label}" aria-selected="${label === state.profile}">${label}</button>
  `).join("");
  document.querySelectorAll(".position-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.profile = tab.dataset.position;
      renderProfile();
    });
  });
  renderProfile();
}

function bindInteractions() {
  elements.picker.addEventListener("click", () => {
    if (elements.pickerMenu.hidden) openPicker();
    else closePicker();
  });
  elements.playerSearch.addEventListener("input", () => renderPlayerOptions(elements.playerSearch.value));
  elements.resetPlayer.addEventListener("click", () => selectPlayer(state.player));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".picker-wrap")) closePicker();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePicker();
  });
}

async function initialize() {
  try {
    const response = await fetch("model-artifact.json");
    if (!response.ok) throw new Error(`Model artifact request failed: ${response.status}`);
    state.data = await response.json();
    renderMetrics();
    renderConfusionMatrix();
    renderClassReport();
    renderBalance();
    renderImportance();
    renderPositionTabs();
    bindInteractions();

    const initialPlayer = state.data.players.find((player) => player.player === "Stephen Curry")
      || state.data.players[0];
    selectPlayer(initialPlayer);
  } catch (error) {
    console.error(error);
    elements.selectedPlayer.textContent = "Model unavailable";
    elements.selectedTeam.textContent = "Run this page through a local web server.";
  }
}

initialize();
