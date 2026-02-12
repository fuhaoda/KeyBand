const DEGREE_SEMITONES = {
  1: 0,
  2: 2,
  3: 4,
  4: 5,
  5: 7,
  6: 9,
  7: 11,
};

const KEY_OFFSETS = {
  C: 0,
  "C#/Db": 1,
  D: 2,
  "D#/Eb": 3,
  E: 4,
  F: 5,
  "F#/Gb": 6,
  G: 7,
  "G#/Ab": 8,
  A: 9,
  "A#/Bb": 10,
  B: 11,
};

const GENDER_BASE_DO = {
  male: 130.8,
  female: 261.6,
};

const RELEASE_SEC_BY_INSTRUMENT = {
  piano: 0.7,
  guitar: 1.0,
};

const UI = {
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  audioButton: document.getElementById("audioButton"),
  instrumentSelect: document.getElementById("instrumentSelect"),
  genderSelect: document.getElementById("genderSelect"),
  keySelect: document.getElementById("keySelect"),
  sustainPill: document.getElementById("sustainPill"),
  octaveLabel: document.getElementById("octaveLabel"),
  octUpButton: document.getElementById("octUpButton"),
  octDownButton: document.getElementById("octDownButton"),
  keys: Array.from(document.querySelectorAll(".key[data-degree]")),
  sustainButton: document.querySelector(".key.sustain"),
};

const STATE = {
  manifest: null,
  samples: [],
  samplesSorted: [],
  buffers: new Map(),
  audioContext: null,
  masterGain: null,
  compressor: null,
  keyVoices: new Map(),
  pointerVoices: new Map(),
  sustainKeys: new Set(),
  sustainTouch: false,
  sustainedVoices: new Set(),
  touchOctaveShift: 0,
};

function updateStatus(text, ok) {
  UI.statusText.textContent = text;
  UI.statusDot.style.background = ok ? "#7fffd4" : "#ff8aa0";
  UI.statusDot.style.boxShadow = ok ? "0 0 12px rgba(127,255,212,0.6)" : "0 0 12px rgba(255,138,160,0.6)";
}

function getInstrumentId() {
  return UI.instrumentSelect?.value || "piano";
}

function getReleaseTime(instrumentId) {
  return RELEASE_SEC_BY_INSTRUMENT[instrumentId] ?? 0.7;
}

function stopAllVoices() {
  for (const voice of STATE.keyVoices.values()) {
    releaseVoice(voice, 0.02);
  }
  for (const voice of STATE.pointerVoices.values()) {
    releaseVoice(voice, 0.02);
  }
  for (const voice of STATE.sustainedVoices.values()) {
    releaseVoice(voice, 0.02);
  }
  STATE.keyVoices.clear();
  STATE.pointerVoices.clear();
  STATE.sustainedVoices.clear();
}

function clearBuffers() {
  STATE.buffers.clear();
}

async function loadManifest(instrument) {
  const response = await fetch(`/assets/audio/${instrument}/manifest.json`);
  if (!response.ok) {
    throw new Error("Audio files are missing");
  }
  const data = await response.json();
  STATE.manifest = data;
  STATE.samples = Array.isArray(data.samples) ? data.samples : [];
  STATE.samplesSorted = [...STATE.samples].sort((a, b) => a.hz - b.hz);
  updateStatus("Ready", true);
}

function ensureAudioContext() {
  if (STATE.audioContext) {
    return STATE.audioContext;
  }
  const context = new (window.AudioContext || window.webkitAudioContext)();
  const masterGain = context.createGain();
  masterGain.gain.value = 0.9;
  const compressor = context.createDynamicsCompressor();
  compressor.threshold.value = -20;
  compressor.knee.value = 12;
  compressor.ratio.value = 2.2;
  compressor.attack.value = 0.003;
  compressor.release.value = 0.2;
  masterGain.connect(compressor).connect(context.destination);
  STATE.audioContext = context;
  STATE.masterGain = masterGain;
  STATE.compressor = compressor;
  return context;
}

async function unlockAudio() {
  const context = ensureAudioContext();
  if (context.state !== "running") {
    await context.resume();
  }
  updateStatus("Audio ready", true);
}

async function ensureBuffer(sampleId) {
  if (STATE.buffers.has(sampleId)) {
    return STATE.buffers.get(sampleId);
  }
  const sample = STATE.samples.find((item) => item.id === sampleId);
  if (!sample) {
    throw new Error(`Sample not found: ${sampleId}`);
  }
  const response = await fetch(sample.file);
  const data = await response.arrayBuffer();
  const context = ensureAudioContext();
  const buffer = await context.decodeAudioData(data.slice(0));
  STATE.buffers.set(sampleId, buffer);
  return buffer;
}

function mapTargetHzToSample(targetHz) {
  const samples = STATE.samplesSorted;
  if (!samples.length) {
    return null;
  }
  let lo = 0;
  let hi = samples.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (samples[mid].hz < targetHz) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const candidate = samples[lo];
  const prev = samples[Math.max(0, lo - 1)];
  const next = samples[Math.min(samples.length - 1, lo + 1)];
  const best = [prev, candidate, next].reduce((closest, sample) => {
    const cents = Math.abs(1200 * Math.log2(targetHz / sample.hz));
    if (!closest) {
      return { sample, cents };
    }
    return cents < closest.cents ? { sample, cents } : closest;
  }, null);
  if (!best) {
    return null;
  }
  return {
    sampleId: best.sample.id,
    playbackRate: targetHz / best.sample.hz,
  };
}

function getToneContext() {
  const gender = UI.genderSelect.value;
  const key = UI.keySelect.value;
  const base = GENDER_BASE_DO[gender];
  const keyOffset = KEY_OFFSETS[key] ?? 0;
  const doFrequency = base * 2 ** (keyOffset / 12);
  return { gender, key, doFrequency };
}

function resolveOctaveShiftFromEvent(event) {
  const up = Boolean(event.shiftKey);
  const down = Boolean(event.ctrlKey);
  if (up && down) {
    return 0;
  }
  if (up) {
    return 1;
  }
  if (down) {
    return -1;
  }
  return 0;
}

function resolveKeyboardDegree(event) {
  if (/^[1-7]$/.test(event.key)) {
    return Number(event.key);
  }
  const code = event.code || "";
  const match = /^(Digit|Numpad)([1-7])$/.exec(code);
  if (match) {
    return Number(match[2]);
  }
  const alternativeMap = {
    KeyM: 1,
    Comma: 2,
    Period: 3,
    KeyJ: 4,
    KeyK: 5,
    KeyL: 6,
    KeyU: 7,
  };
  return alternativeMap[code] ?? null;
}

function isSustainKey(event) {
  const code = event.code || "";
  return (
    code === "Digit8" ||
    code === "Digit9" ||
    code === "Numpad8" ||
    code === "Numpad9" ||
    code === "KeyI" ||
    code === "KeyO"
  );
}

function setSustainActive(active) {
  const currently = UI.sustainPill.dataset.active === "true";
  if (active === currently) {
    return;
  }
  UI.sustainPill.dataset.active = active ? "true" : "false";
  UI.sustainPill.textContent = active ? "Sustain On" : "Sustain Off";
  UI.sustainPill.style.background = active ? "rgba(127,255,212,0.18)" : "rgba(255,255,255,0.08)";
  if (!active) {
    for (const voice of Array.from(STATE.sustainedVoices)) {
      const releaseTime = getReleaseTime(voice.instrument || "piano");
      releaseVoice(voice, releaseTime);
    }
    STATE.sustainedVoices.clear();
  }
}

function updateSustainState() {
  const active = STATE.sustainKeys.size > 0 || STATE.sustainTouch;
  setSustainActive(active);
}

async function playDegree(degree, octaveShift, keyId) {
  const context = ensureAudioContext();
  const toneContext = getToneContext();
  const semitone = DEGREE_SEMITONES[degree];
  if (semitone === undefined) {
    return;
  }
  const targetHz =
    toneContext.doFrequency * 2 ** (semitone / 12) * 2 ** octaveShift;
  const mapping = mapTargetHzToSample(targetHz);
  if (!mapping) {
    return;
  }
  const instrumentId = getInstrumentId();
  const buffer = await ensureBuffer(mapping.sampleId);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = mapping.playbackRate;
  const gainNode = context.createGain();
  gainNode.gain.value = 1.0;
  source.connect(gainNode).connect(STATE.masterGain);
  const voice = { source, gainNode, released: false, instrument: instrumentId };
  if (keyId) {
    voice.keyId = keyId;
    STATE.keyVoices.set(keyId, voice);
  }
  source.start(context.currentTime + 0.001);
  source.onended = () => {
    STATE.sustainedVoices.delete(voice);
    if (voice.keyId && STATE.keyVoices.get(voice.keyId) === voice) {
      STATE.keyVoices.delete(voice.keyId);
    }
  };
  return voice;
}

function releaseVoice(voice, releaseTime = RELEASE_SEC) {
  if (!voice || voice.released) {
    return;
  }
  voice.released = true;
  const context = ensureAudioContext();
  const now = context.currentTime;
  voice.gainNode.gain.cancelScheduledValues(now);
  voice.gainNode.gain.setValueAtTime(voice.gainNode.gain.value, now);
  voice.gainNode.gain.linearRampToValueAtTime(0.0001, now + releaseTime);
  voice.source.stop(now + releaseTime + 0.05);
}

function releaseKeyVoice(keyId) {
  const voice = STATE.keyVoices.get(keyId);
  if (!voice) {
    return;
  }
  STATE.keyVoices.delete(keyId);
  const sustainActive = UI.sustainPill.dataset.active === "true";
  if (sustainActive) {
    STATE.sustainedVoices.add(voice);
    return;
  }
  releaseVoice(voice, getReleaseTime(voice.instrument || "piano"));
}

function updateTouchOctaveLabel() {
  UI.octaveLabel.textContent = String(STATE.touchOctaveShift);
}

function bindKeyboardEvents() {
  window.addEventListener("keydown", async (event) => {
    if (event.repeat) {
      return;
    }
    const targetTag = event.target?.tagName;
    if (targetTag === "INPUT" || targetTag === "TEXTAREA" || targetTag === "SELECT") {
      return;
    }
    await unlockAudio();
    if (isSustainKey(event)) {
      event.preventDefault();
      STATE.sustainKeys.add(event.code);
      updateSustainState();
      return;
    }
    const degree = resolveKeyboardDegree(event);
    if (!degree) {
      return;
    }
    event.preventDefault();
    const octaveShift = resolveOctaveShiftFromEvent(event);
    const keyId = event.code || event.key;
    await playDegree(degree, octaveShift, keyId);
    const button = UI.keys.find((el) => Number(el.dataset.degree) === degree);
    if (button) {
      button.classList.add("is-active");
    }
  });

  window.addEventListener("keyup", (event) => {
    const targetTag = event.target?.tagName;
    if (targetTag === "INPUT" || targetTag === "TEXTAREA" || targetTag === "SELECT") {
      return;
    }
    if (isSustainKey(event)) {
      event.preventDefault();
      STATE.sustainKeys.delete(event.code);
      updateSustainState();
      return;
    }
    const keyId = event.code || event.key;
    releaseKeyVoice(keyId);
    const degree = resolveKeyboardDegree(event);
    if (degree) {
      const button = UI.keys.find((el) => Number(el.dataset.degree) === degree);
      if (button) {
        button.classList.remove("is-active");
      }
    }
  });
}

function bindTouchKeys() {
  UI.keys.forEach((button) => {
    button.addEventListener("pointerdown", async (event) => {
      const degree = Number(button.dataset.degree);
      if (!degree) {
        return;
      }
      await unlockAudio();
      button.classList.add("is-active");
      button.setPointerCapture(event.pointerId);
      const octaveShift = STATE.touchOctaveShift;
      const voice = await playDegree(degree, octaveShift, null);
      if (voice) {
        STATE.pointerVoices.set(event.pointerId, voice);
      }
    });
    const releaseHandler = (event) => {
      const voice = STATE.pointerVoices.get(event.pointerId);
      if (voice) {
        const sustainActive = UI.sustainPill.dataset.active === "true";
        if (sustainActive) {
          STATE.sustainedVoices.add(voice);
        } else {
          releaseVoice(voice, getReleaseTime(voice.instrument || "piano"));
        }
        STATE.pointerVoices.delete(event.pointerId);
      }
      button.classList.remove("is-active");
    };
    button.addEventListener("pointerup", releaseHandler);
    button.addEventListener("pointercancel", releaseHandler);
  });

  UI.sustainButton.addEventListener("pointerdown", (event) => {
    UI.sustainButton.classList.add("is-active");
    UI.sustainButton.setPointerCapture(event.pointerId);
    STATE.sustainTouch = true;
    updateSustainState();
  });
  UI.sustainButton.addEventListener("pointerup", () => {
    UI.sustainButton.classList.remove("is-active");
    STATE.sustainTouch = false;
    updateSustainState();
  });
  UI.sustainButton.addEventListener("pointercancel", () => {
    UI.sustainButton.classList.remove("is-active");
    STATE.sustainTouch = false;
    updateSustainState();
  });
}

function bindControlButtons() {
  UI.audioButton.addEventListener("click", async () => {
    await unlockAudio();
  });

  UI.instrumentSelect.addEventListener("change", async () => {
    stopAllVoices();
    clearBuffers();
    updateStatus("Loading…", false);
    try {
      await loadManifest(UI.instrumentSelect.value);
    } catch (error) {
      updateStatus(error.message || "Failed to load audio", false);
    }
  });

  UI.octUpButton.addEventListener("click", () => {
    STATE.touchOctaveShift = Math.min(2, STATE.touchOctaveShift + 1);
    updateTouchOctaveLabel();
  });

  UI.octDownButton.addEventListener("click", () => {
    STATE.touchOctaveShift = Math.max(-2, STATE.touchOctaveShift - 1);
    updateTouchOctaveLabel();
  });
}

async function init() {
  try {
    updateStatus("Loading…", false);
    await loadManifest(UI.instrumentSelect.value);
  } catch (error) {
    updateStatus(error.message || "Failed to load audio", false);
  }
  bindKeyboardEvents();
  bindTouchKeys();
  bindControlButtons();
  updateTouchOctaveLabel();
  updateSustainState();
}

init();
