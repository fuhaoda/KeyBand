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
  flute: 1.0,
};

const DEFAULT_RELEASE_SEC = 0.7;
const INSTRUMENT_GAIN_MIN = 0.5;
const INSTRUMENT_GAIN_MAX = 2.5;
const AUDIO_UNLOCK_TIMEOUT_MS = 1200;
const AUDIO_UNLOCK_RECREATE_FAILURE_THRESHOLD = 2;
const MOBILE_OVERLAY_EXIT_HOLD_MS = 500;
const WARMUP_NOTE_TARGETS = [
  { degree: 1, octaveShift: 0 },
  { degree: 2, octaveShift: 0 },
  { degree: 3, octaveShift: 0 },
  { degree: 4, octaveShift: 0 },
  { degree: 5, octaveShift: 0 },
  { degree: 6, octaveShift: 0 },
  { degree: 7, octaveShift: 0 },
  { degree: 1, octaveShift: 1 },
  { degree: 2, octaveShift: 1 },
  { degree: 4, octaveShift: -1 },
  { degree: 5, octaveShift: -1 },
  { degree: 6, octaveShift: -1 },
];

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
  mobileKeyboardButton: document.getElementById("mobileKeyboardButton"),
  mobileOverlay: document.getElementById("mobileOverlay"),
  mobileKeys: Array.from(document.querySelectorAll(".mobile-key.degree")),
  mobilePed: document.querySelector(".mobile-key.ped"),
  mobileOctUp: document.querySelector('.mobile-key.oct[data-mobile-oct="up"]'),
  mobileOctDown: document.querySelector('.mobile-key.oct[data-mobile-oct="down"]'),
  keypadOctUp: document.querySelector('.key.oct[data-oct="up"]'),
  keypadOctDown: document.querySelector('.key.oct[data-oct="down"]'),
};

const STATE = {
  manifest: null,
  samples: [],
  samplesSorted: [],
  buffers: new Map(),
  bufferPromises: new Map(),
  bufferEpoch: 0,
  audioContext: null,
  masterGain: null,
  compressor: null,
  instrumentGain: 1.0,
  referencePeak: null,
  keyVoices: new Map(),
  pointerVoices: new Map(),
  pendingKeys: new Map(),
  pendingPointers: new Map(),
  sustainKeys: new Set(),
  sustainTouch: false,
  sustainedVoices: new Set(),
  touchOctaveShift: 0,
  touchOctaveModifier: 0,
  keypadOctHoldUp: false,
  keypadOctHoldDown: false,
  mobileOverlayActive: false,
  mobileOctHoldUp: false,
  mobileOctHoldDown: false,
  mobileExitHoldTimerId: null,
  audioUnlocked: false,
  audioUnlockPendingGesture: false,
  audioUnlockFailures: 0,
  audioUnlockPromise: null,
  warmupTaskId: 0,
  warmupTimerId: null,
  lastWarmupSignature: "",
  warmupPendingSelection: false,
};

function updateStatus(text, ok) {
  UI.statusText.textContent = text;
  UI.statusDot.style.background = ok ? "#7fffd4" : "#ff8aa0";
  UI.statusDot.style.boxShadow = ok ? "0 0 12px rgba(127,255,212,0.6)" : "0 0 12px rgba(255,138,160,0.6)";
}

function pushDebug(message) {
  const time = new Date().toLocaleTimeString();
  console.log(`[${time}] ${message}`);
}

function setDebugStatus(message, ok = true) {
  if (!message) {
    return;
  }
  console.log(`[Debug] ${message} (${ok ? "ok" : "fail"})`);
}

function setDebugError(message) {
  if (message) {
    console.warn(`[Debug] ${message}`);
  }
}

function updateDebugInfo() {
  return;
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
  STATE.bufferEpoch += 1;
  STATE.buffers.clear();
  STATE.bufferPromises.clear();
  STATE.lastWarmupSignature = "";
  STATE.warmupPendingSelection = false;
  if (STATE.warmupTimerId) {
    window.clearTimeout(STATE.warmupTimerId);
    STATE.warmupTimerId = null;
  }
  STATE.warmupTaskId += 1;
}

function clearMobileExitHoldTimer() {
  if (!STATE.mobileExitHoldTimerId) {
    return;
  }
  window.clearTimeout(STATE.mobileExitHoldTimerId);
  STATE.mobileExitHoldTimerId = null;
}

async function loadManifest(instrument) {
  pushDebug(`Load manifest: ${instrument}`);
  const response = await fetch(`assets/audio/${instrument}/manifest.json`);
  if (!response.ok) {
    setDebugStatus("Manifest load failed", false);
    setDebugError("Manifest fetch failed");
    throw new Error("Audio files are missing");
  }
  const data = await response.json();
  if (instrument === "piano" && data?.quality?.maxPeakLinear) {
    STATE.referencePeak = data.quality.maxPeakLinear;
  } else if (!STATE.referencePeak && data?.quality?.maxPeakLinear) {
    STATE.referencePeak = data.quality.maxPeakLinear;
  }
  STATE.instrumentGain = resolveInstrumentGain(data);
  updateDebugInfo();
  STATE.manifest = data;
  STATE.samples = Array.isArray(data.samples) ? data.samples : [];
  STATE.samplesSorted = [...STATE.samples].sort((a, b) => a.hz - b.hz);
  queueWarmupForCurrentSelection("manifest");
  updateStatus("Ready", true);
  setDebugStatus("Ready", true);
}

function resolveInstrumentGain(manifest) {
  const peak = manifest?.quality?.maxPeakLinear;
  const reference = STATE.referencePeak;
  if (!peak || !reference) {
    return 1.0;
  }
  const gain = reference / peak;
  return Math.min(INSTRUMENT_GAIN_MAX, Math.max(INSTRUMENT_GAIN_MIN, gain));
}

function configureAudioSessionIfSupported() {
  try {
    const audioSession = navigator.audioSession;
    if (!audioSession) {
      return;
    }
    if (audioSession.type !== "playback") {
      audioSession.type = "playback";
    }
    pushDebug(`audioSession.type=${audioSession.type}`);
  } catch (error) {
    pushDebug(`audioSession failed: ${error?.message || "unknown"}`);
  }
}

async function resumeAudioContextWithTimeout(context, label = "primary") {
  if (context.state === "running") {
    return;
  }
  const resumeResult = context.resume();
  if (resumeResult && typeof resumeResult.then === "function") {
    await Promise.race([
      resumeResult,
      new Promise((_, reject) => {
        window.setTimeout(() => {
          reject(new Error(`AudioContext resume timed out (${label})`));
        }, AUDIO_UNLOCK_TIMEOUT_MS);
      }),
    ]);
    return;
  }
  await new Promise((resolve) => window.setTimeout(resolve, 40));
  if (context.state !== "running") {
    throw new Error(`AudioContext resume did not enter running state (${label})`);
  }
}

function createAudioUnlockPendingError() {
  const error = new Error("Audio unlock pending user gesture");
  error.name = "AudioUnlockPendingError";
  return error;
}

function isAudioUnlockPendingError(error) {
  return error?.name === "AudioUnlockPendingError";
}

async function recreateAudioContext() {
  const previous = STATE.audioContext;
  STATE.audioContext = null;
  STATE.audioUnlocked = false;
  if (previous && previous.state !== "closed") {
    try {
      await previous.close();
      pushDebug("Previous AudioContext closed");
    } catch (error) {
      pushDebug(`AudioContext close failed: ${error?.message || "unknown"}`);
    }
  }
  const fresh = getOrCreateAudioContext();
  pushDebug("Created fresh AudioContext");
  return fresh;
}

function getOrCreateAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("WebAudio API is not available in this browser");
  }
  if (STATE.audioContext && STATE.audioContext.state === "closed") {
    STATE.audioContext = null;
  }
  if (!STATE.audioContext) {
    STATE.audioContext = new AudioContextClass({ latencyHint: "interactive" });
    const masterGain = STATE.audioContext.createGain();
    masterGain.gain.value = 0.9;
    const compressor = STATE.audioContext.createDynamicsCompressor();
    compressor.threshold.value = -20;
    compressor.knee.value = 12;
    compressor.ratio.value = 2.2;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.2;
    masterGain.connect(compressor).connect(STATE.audioContext.destination);
    STATE.masterGain = masterGain;
    STATE.compressor = compressor;
  }
  updateDebugInfo();
  return STATE.audioContext;
}

function primeAudioContextTick(context) {
  const gainNode = context.createGain();
  gainNode.gain.value = 0.00001;
  const oscillator = context.createOscillator();
  oscillator.type = "sine";
  oscillator.frequency.value = 880;
  oscillator.connect(gainNode);
  gainNode.connect(context.destination);
  const now = context.currentTime;
  oscillator.start(now);
  oscillator.stop(now + 0.01);
}

async function unlockAudioPipeline() {
  let context = getOrCreateAudioContext();
  configureAudioSessionIfSupported();
  if (context.state === "running") {
    STATE.audioUnlocked = true;
    STATE.audioUnlockPendingGesture = false;
    STATE.audioUnlockFailures = 0;
    pushDebug("AudioContext already running");
    return true;
  }

  if (!STATE.audioUnlockPromise) {
    STATE.audioUnlockPromise = (async () => {
      try {
        await resumeAudioContextWithTimeout(context, "primary");
      } catch (primaryError) {
        STATE.audioUnlockFailures += 1;
        STATE.audioUnlockPendingGesture = true;
        pushDebug(`Primary resume failed: ${primaryError?.message || "unknown error"}`);
        if (STATE.audioUnlockFailures < AUDIO_UNLOCK_RECREATE_FAILURE_THRESHOLD) {
          return false;
        }
        try {
          context = await recreateAudioContext();
          configureAudioSessionIfSupported();
          await resumeAudioContextWithTimeout(context, "recreated");
        } catch (recreatedError) {
          STATE.audioUnlockFailures += 1;
          STATE.audioUnlockPendingGesture = true;
          pushDebug(`Recreated resume failed: ${recreatedError?.message || "unknown error"}`);
          return false;
        }
      }
      configureAudioSessionIfSupported();
      primeAudioContextTick(context);
      STATE.audioUnlocked = true;
      STATE.audioUnlockPendingGesture = false;
      STATE.audioUnlockFailures = 0;
      pushDebug(`AudioContext resumed: ${context.state}`);
      return true;
    })().finally(() => {
      STATE.audioUnlockPromise = null;
    });
  }

  return STATE.audioUnlockPromise;
}

async function ensureAudioContextRunning() {
  let context = getOrCreateAudioContext();
  if (context.state === "running") {
    return context;
  }
  let unlocked = await unlockAudioPipeline();
  context = STATE.audioContext || context;
  if (!unlocked || context.state !== "running") {
    pushDebug(`Context state after unlock: ${context.state}`);
    unlocked = await unlockAudioPipeline();
    context = STATE.audioContext || context;
  }
  if (context.state !== "running") {
    if (!unlocked || STATE.audioUnlockPendingGesture) {
      throw createAudioUnlockPendingError();
    }
    throw new Error("Audio context is not running");
  }
  return context;
}

function warmAudioPipelineOnce() {
  if (STATE.audioUnlocked) {
    queueWarmupForCurrentSelection("warm-listener-ready");
    removeAudioUnlockListeners();
    return;
  }
  void unlockAudioPipeline()
    .then(() => {
      if (STATE.audioUnlocked) {
        queueWarmupForCurrentSelection("warm-listener-unlocked");
        removeAudioUnlockListeners();
      }
    })
    .catch(() => {});
}

function armAudioUnlockListeners() {
  const options = { capture: true };
  window.addEventListener("pointerup", warmAudioPipelineOnce, options);
  window.addEventListener("touchend", warmAudioPipelineOnce, options);
  window.addEventListener("click", warmAudioPipelineOnce, options);
  window.addEventListener("keydown", warmAudioPipelineOnce, options);
}

function removeAudioUnlockListeners() {
  const options = { capture: true };
  window.removeEventListener("pointerup", warmAudioPipelineOnce, options);
  window.removeEventListener("touchend", warmAudioPipelineOnce, options);
  window.removeEventListener("click", warmAudioPipelineOnce, options);
  window.removeEventListener("keydown", warmAudioPipelineOnce, options);
}

async function unlockAudio() {
  pushDebug("Unlock audio");
  setDebugStatus("Unlocking…", true);
  try {
    await ensureAudioContextRunning();
    queueWarmupForCurrentSelection("unlock");
    updateStatus("Audio ready", true);
    setDebugStatus("Audio ready", true);
  } catch (error) {
    if (isAudioUnlockPendingError(error)) {
      setDebugStatus("Audio pending gesture", false);
    } else {
      setDebugStatus("Audio unlock failed", false);
      setDebugError(error?.message || "Audio unlock failed");
      pushDebug(`Unlock error: ${error?.message || "unknown"}`);
    }
  }
  updateDebugInfo();
}

async function ensureBuffer(sampleId) {
  if (STATE.buffers.has(sampleId)) {
    return STATE.buffers.get(sampleId);
  }
  const epoch = STATE.bufferEpoch;
  const inFlight = STATE.bufferPromises.get(sampleId);
  if (inFlight && inFlight.epoch === epoch) {
    return inFlight.promise;
  }
  const sample = STATE.samples.find((item) => item.id === sampleId);
  if (!sample) {
    setDebugError(`Sample not found: ${sampleId}`);
    throw new Error(`Sample not found: ${sampleId}`);
  }
  const loadPromise = (async () => {
    try {
      pushDebug(`Fetch sample: ${sample.id}`);
      const response = await fetch(sample.file);
      if (!response.ok) {
        throw new Error(`Fetch failed: ${response.status}`);
      }
      const data = await response.arrayBuffer();
      const context = getOrCreateAudioContext();
      const buffer = await decodeAudioDataCompat(context, data);
      if (epoch !== STATE.bufferEpoch) {
        pushDebug(`Discard stale sample: ${sample.id}`);
        return null;
      }
      STATE.buffers.set(sampleId, buffer);
      return buffer;
    } catch (error) {
      setDebugStatus("Decode failed", false);
      setDebugError(error?.message || "Decode failed");
      pushDebug(`Decode error: ${error?.message || "unknown"}`);
      throw error;
    }
  })();
  STATE.bufferPromises.set(sampleId, { epoch, promise: loadPromise });
  try {
    return await loadPromise;
  } finally {
    const active = STATE.bufferPromises.get(sampleId);
    if (active?.promise === loadPromise) {
      STATE.bufferPromises.delete(sampleId);
    }
  }
}

function decodeAudioDataCompat(context, encoded) {
  const clonedBuffer = encoded.slice(0);
  return new Promise((resolve, reject) => {
    const promiseLike = context.decodeAudioData(
      clonedBuffer,
      (buffer) => resolve(buffer),
      (error) => reject(error || new Error("Failed to decode audio sample")),
    );
    if (promiseLike && typeof promiseLike.then === "function") {
      promiseLike.then(resolve).catch(reject);
    }
  });
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

function getWarmupSampleIdsForCurrentSelection() {
  if (!STATE.samplesSorted.length) {
    return [];
  }
  const toneContext = getToneContext();
  const seen = new Set();
  const sampleIds = [];
  for (const target of WARMUP_NOTE_TARGETS) {
    const semitone = DEGREE_SEMITONES[target.degree];
    if (semitone === undefined) {
      continue;
    }
    const targetHz =
      toneContext.doFrequency * 2 ** (semitone / 12) * 2 ** target.octaveShift;
    const mapping = mapTargetHzToSample(targetHz);
    if (!mapping || seen.has(mapping.sampleId)) {
      continue;
    }
    seen.add(mapping.sampleId);
    sampleIds.push(mapping.sampleId);
  }
  return sampleIds;
}

function queueWarmupForCurrentSelection(reason = "selection") {
  if (!STATE.samplesSorted.length) {
    return;
  }
  if (!STATE.audioUnlocked) {
    STATE.warmupPendingSelection = true;
    return;
  }
  const sampleIds = getWarmupSampleIdsForCurrentSelection();
  if (!sampleIds.length) {
    return;
  }
  const signature = `${getInstrumentId()}|${UI.genderSelect.value}|${UI.keySelect.value}|${sampleIds.join(",")}`;
  if (!STATE.warmupPendingSelection && signature === STATE.lastWarmupSignature) {
    return;
  }
  STATE.lastWarmupSignature = signature;
  STATE.warmupPendingSelection = false;
  const taskId = STATE.warmupTaskId + 1;
  STATE.warmupTaskId = taskId;
  if (STATE.warmupTimerId) {
    window.clearTimeout(STATE.warmupTimerId);
  }
  STATE.warmupTimerId = window.setTimeout(() => {
    STATE.warmupTimerId = null;
    void warmupSamples(taskId, sampleIds, reason);
  }, 40);
}

async function warmupSamples(taskId, sampleIds, reason) {
  pushDebug(`Warmup start (${reason}): ${sampleIds.length} samples`);
  for (const sampleId of sampleIds) {
    if (taskId !== STATE.warmupTaskId) {
      pushDebug("Warmup canceled");
      return;
    }
    try {
      await ensureBuffer(sampleId);
    } catch (error) {
      pushDebug(`Warmup sample failed (${sampleId}): ${error?.message || "unknown error"}`);
    }
  }
  if (taskId !== STATE.warmupTaskId) {
    return;
  }
  pushDebug("Warmup ready");
}

function syncMobileOverlayExitHold() {
  if (!STATE.mobileOverlayActive) {
    clearMobileExitHoldTimer();
    return;
  }
  if (!STATE.mobileOctHoldUp || !STATE.mobileOctHoldDown) {
    clearMobileExitHoldTimer();
    return;
  }
  if (STATE.mobileExitHoldTimerId) {
    return;
  }
  STATE.mobileExitHoldTimerId = window.setTimeout(() => {
    STATE.mobileExitHoldTimerId = null;
    if (STATE.mobileOverlayActive && STATE.mobileOctHoldUp && STATE.mobileOctHoldDown) {
      closeMobileOverlay();
    }
  }, MOBILE_OVERLAY_EXIT_HOLD_MS);
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
    code === "Digit0" ||
    code === "Numpad8" ||
    code === "Numpad9" ||
    code === "Numpad0" ||
    code === "KeyI" ||
    code === "KeyO" ||
    code === "Space"
  );
}

function setSustainActive(active) {
  const currently = UI.sustainPill.dataset.active === "true";
  if (active === currently) {
    return;
  }
  UI.sustainPill.dataset.active = active ? "true" : "false";
  UI.sustainPill.textContent = active ? "Ped On" : "Ped Off";
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

async function playDegree(degree, octaveShift, keyId, pending) {
  let context;
  try {
    context = await ensureAudioContextRunning();
  } catch (error) {
    if (isAudioUnlockPendingError(error)) {
      pushDebug("Audio unlock pending gesture");
      return null;
    }
    throw error;
  }
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
  if (!buffer) {
    return null;
  }
  if (pending && pending.released) {
    return null;
  }
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = mapping.playbackRate;
  const gainNode = context.createGain();
  gainNode.gain.value = STATE.instrumentGain;
  source.connect(gainNode).connect(STATE.masterGain);
  const voice = { source, gainNode, released: false, instrument: instrumentId };
  if (keyId) {
    voice.keyId = keyId;
    STATE.keyVoices.set(keyId, voice);
  }
  if (pending) {
    pending.voice = voice;
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

function releaseVoice(voice, releaseTime = DEFAULT_RELEASE_SEC) {
  if (!voice || voice.released) {
    return;
  }
  voice.released = true;
  const context = STATE.audioContext || getOrCreateAudioContext();
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

async function startKeyNote(keyId, degree, octaveShift) {
  if (STATE.pendingKeys.has(keyId)) {
    return;
  }
  const pending = { released: false, voice: null };
  STATE.pendingKeys.set(keyId, pending);
  let voice = null;
  try {
    voice = await playDegree(degree, octaveShift, keyId, pending);
  } catch (error) {
    setDebugError(error?.message || "Playback failed");
    pushDebug(`Play error: ${error?.message || "unknown error"}`);
  }
  if (pending.released && voice) {
    releaseVoice(voice, 0.02);
  }
  STATE.pendingKeys.delete(keyId);
}

function releasePendingKey(keyId) {
  const pending = STATE.pendingKeys.get(keyId);
  if (!pending) {
    return false;
  }
  if (!pending.voice) {
    pending.released = true;
    return true;
  }
  return false;
}

async function startPointerNote(pointerId, degree, octaveShift) {
  if (STATE.pendingPointers.has(pointerId)) {
    return;
  }
  const pending = { released: false, voice: null };
  STATE.pendingPointers.set(pointerId, pending);
  let voice = null;
  try {
    voice = await playDegree(degree, octaveShift, null, pending);
  } catch (error) {
    setDebugError(error?.message || "Playback failed");
    pushDebug(`Play error: ${error?.message || "unknown error"}`);
  }
  if (pending.released && voice) {
    releaseVoice(voice, 0.02);
  }
  if (voice) {
    STATE.pointerVoices.set(pointerId, voice);
  }
  STATE.pendingPointers.delete(pointerId);
}

function releasePendingPointer(pointerId) {
  const pending = STATE.pendingPointers.get(pointerId);
  if (!pending) {
    return false;
  }
  if (!pending.voice) {
    pending.released = true;
    return true;
  }
  return false;
}

function updateTouchOctaveLabel() {
  UI.octaveLabel.textContent = String(STATE.touchOctaveShift);
}

function getMobileOctaveModifier() {
  if (STATE.mobileOctHoldUp && STATE.mobileOctHoldDown) {
    return 0;
  }
  if (STATE.mobileOctHoldUp) {
    return 1;
  }
  if (STATE.mobileOctHoldDown) {
    return -1;
  }
  return 0;
}

function bindKeypadOctaveButtons() {
  const updateKeypadModifier = () => {
    if (STATE.keypadOctHoldUp && STATE.keypadOctHoldDown) {
      STATE.touchOctaveModifier = 0;
    } else if (STATE.keypadOctHoldUp) {
      STATE.touchOctaveModifier = 1;
    } else if (STATE.keypadOctHoldDown) {
      STATE.touchOctaveModifier = -1;
    } else {
      STATE.touchOctaveModifier = 0;
    }
  };
  if (UI.keypadOctUp) {
    UI.keypadOctUp.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      UI.keypadOctUp.classList.add("is-active");
      STATE.keypadOctHoldUp = true;
      updateKeypadModifier();
    });
    UI.keypadOctUp.addEventListener("pointerup", () => {
      UI.keypadOctUp.classList.remove("is-active");
      STATE.keypadOctHoldUp = false;
      updateKeypadModifier();
    });
    UI.keypadOctUp.addEventListener("pointercancel", () => {
      UI.keypadOctUp.classList.remove("is-active");
      STATE.keypadOctHoldUp = false;
      updateKeypadModifier();
    });
  }
  if (UI.keypadOctDown) {
    UI.keypadOctDown.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      UI.keypadOctDown.classList.add("is-active");
      STATE.keypadOctHoldDown = true;
      updateKeypadModifier();
    });
    UI.keypadOctDown.addEventListener("pointerup", () => {
      UI.keypadOctDown.classList.remove("is-active");
      STATE.keypadOctHoldDown = false;
      updateKeypadModifier();
    });
    UI.keypadOctDown.addEventListener("pointercancel", () => {
      UI.keypadOctDown.classList.remove("is-active");
      STATE.keypadOctHoldDown = false;
      updateKeypadModifier();
    });
  }
}

function openMobileOverlay() {
  if (!UI.mobileOverlay) {
    return;
  }
  clearMobileExitHoldTimer();
  STATE.mobileOverlayActive = true;
  STATE.mobileOctHoldUp = false;
  STATE.mobileOctHoldDown = false;
  UI.mobileOctUp?.classList.remove("is-active");
  UI.mobileOctDown?.classList.remove("is-active");
  UI.mobileOverlay.classList.remove("hidden");
  UI.mobileOverlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("no-scroll");
}

function closeMobileOverlay() {
  if (!UI.mobileOverlay) {
    return;
  }
  clearMobileExitHoldTimer();
  STATE.mobileOverlayActive = false;
  STATE.mobileOctHoldUp = false;
  STATE.mobileOctHoldDown = false;
  UI.mobileOctUp?.classList.remove("is-active");
  UI.mobileOctDown?.classList.remove("is-active");
  UI.mobilePed?.classList.remove("is-active");
  UI.mobileOverlay.classList.add("hidden");
  UI.mobileOverlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("no-scroll");
}

function handleMobileOctDown(direction) {
  if (direction === "up") {
    STATE.mobileOctHoldUp = true;
  } else {
    STATE.mobileOctHoldDown = true;
  }
  syncMobileOverlayExitHold();
}

function handleMobileOctUp(direction) {
  if (direction === "up") {
    STATE.mobileOctHoldUp = false;
  } else {
    STATE.mobileOctHoldDown = false;
  }
  syncMobileOverlayExitHold();
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
    await startKeyNote(keyId, degree, octaveShift);
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
    if (releasePendingKey(keyId)) {
      return;
    }
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
      const octaveShift = Math.max(-2, Math.min(2, STATE.touchOctaveShift + STATE.touchOctaveModifier));
      await startPointerNote(event.pointerId, degree, octaveShift);
    });
    const releaseHandler = (event) => {
      if (releasePendingPointer(event.pointerId)) {
        button.classList.remove("is-active");
        return;
      }
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

  UI.genderSelect.addEventListener("change", () => {
    queueWarmupForCurrentSelection("gender-change");
  });

  UI.keySelect.addEventListener("change", () => {
    queueWarmupForCurrentSelection("key-change");
  });

  UI.octUpButton.addEventListener("click", () => {
    STATE.touchOctaveShift = Math.min(2, STATE.touchOctaveShift + 1);
    updateTouchOctaveLabel();
  });

  UI.octDownButton.addEventListener("click", () => {
    STATE.touchOctaveShift = Math.max(-2, STATE.touchOctaveShift - 1);
    updateTouchOctaveLabel();
  });

  UI.mobileKeyboardButton?.addEventListener("click", () => {
    openMobileOverlay();
  });

  UI.mobileOverlay?.addEventListener("click", (event) => {
    if (event.target === UI.mobileOverlay) {
      closeMobileOverlay();
    }
  });

  UI.mobileOctUp?.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    UI.mobileOctUp.classList.add("is-active");
    UI.mobileOctUp.setPointerCapture(event.pointerId);
    handleMobileOctDown("up");
  });
  UI.mobileOctUp?.addEventListener("pointerup", (event) => {
    event.preventDefault();
    UI.mobileOctUp.classList.remove("is-active");
    handleMobileOctUp("up");
  });
  UI.mobileOctUp?.addEventListener("pointercancel", (event) => {
    event.preventDefault();
    UI.mobileOctUp.classList.remove("is-active");
    handleMobileOctUp("up");
  });
  UI.mobileOctDown?.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    UI.mobileOctDown.classList.add("is-active");
    UI.mobileOctDown.setPointerCapture(event.pointerId);
    handleMobileOctDown("down");
  });
  UI.mobileOctDown?.addEventListener("pointerup", (event) => {
    event.preventDefault();
    UI.mobileOctDown.classList.remove("is-active");
    handleMobileOctUp("down");
  });
  UI.mobileOctDown?.addEventListener("pointercancel", (event) => {
    event.preventDefault();
    UI.mobileOctDown.classList.remove("is-active");
    handleMobileOctUp("down");
  });

  UI.mobilePed?.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    UI.mobilePed.classList.add("is-active");
    UI.mobilePed.setPointerCapture(event.pointerId);
    STATE.sustainTouch = true;
    updateSustainState();
  });
  UI.mobilePed?.addEventListener("pointerup", () => {
    UI.mobilePed.classList.remove("is-active");
    STATE.sustainTouch = false;
    updateSustainState();
  });
  UI.mobilePed?.addEventListener("pointercancel", () => {
    UI.mobilePed.classList.remove("is-active");
    STATE.sustainTouch = false;
    updateSustainState();
  });

  UI.mobileKeys?.forEach((button) => {
    button.addEventListener("pointerdown", async (event) => {
      const degree = Number(button.dataset.degree);
      if (!degree) {
        return;
      }
      await unlockAudio();
      button.classList.add("is-active");
      button.setPointerCapture(event.pointerId);
      const octaveShift = Math.max(-2, Math.min(2, STATE.touchOctaveShift + getMobileOctaveModifier()));
      await startPointerNote(event.pointerId, degree, octaveShift);
    });
    const releaseHandler = (event) => {
      if (releasePendingPointer(event.pointerId)) {
        button.classList.remove("is-active");
        return;
      }
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
}

async function init() {
  try {
    updateStatus("Loading…", false);
    await loadManifest(UI.instrumentSelect.value);
  } catch (error) {
    updateStatus(error.message || "Failed to load audio", false);
    setDebugStatus("Load failed", false);
    setDebugError(error.message || "Failed to load audio");
  }
  bindKeyboardEvents();
  bindTouchKeys();
  bindControlButtons();
  bindKeypadOctaveButtons();
  updateTouchOctaveLabel();
  updateSustainState();
  updateDebugInfo();
  armAudioUnlockListeners();
}

init();
