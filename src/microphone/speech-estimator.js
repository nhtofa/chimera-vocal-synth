// Estimates acoustic gestures from the newest 32 ms of an AnalyserNode waveform.
// This classic script is also embedded in the downloadable standalone page.
function createSpeechEstimator() {
  const rate = 16000, size = 512, tractSize = size / 2, order = 10;
  const wave = new Float64Array(size), tract = new Float64Array(tractSize);
  const emphasized = new Float64Array(tractSize), correlation = new Float64Array(order + 1);
  const lpc = new Float64Array(order + 1), previous = new Float64Array(order + 1);
  const frequencies = 145, envelope = new Float64Array(frequencies);
  const cosTable = new Float64Array(frequencies * (order + 1)), sinTable = cosTable.slice();
  for (let bin = 0; bin < frequencies; bin++) {
    for (let j = 1; j <= order; j++) {
      const angle = 2 * Math.PI * (bin * 25) * j / (rate / 2);
      cosTable[bin * (order + 1) + j] = Math.cos(angle);
      sinTable[bin * (order + 1) + j] = Math.sin(angle);
    }
  }

  const hann = Float64Array.from({length: tractSize}, (_, i) => .54 - .46 * Math.cos(2 * Math.PI * i / (tractSize - 1)));
  const bands = new Float64Array(5), lagScore = new Float64Array(110);
  // Two short Hann-windowed spectra average local noise fluctuations. The 125-Hz
  // bins cover murmur, vowel, burst and sibilant regions up to the 8-kHz limit.
  const bandWindow = Float64Array.from({length: 128}, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / 127));
  const bandInput = new Float64Array(128);
  const resonator = Float64Array.from({length: 64}, (_, bin) => 2 * Math.cos(2 * Math.PI * bin / 128));
  const vowels = [[740, 1270], [310, 2320], [355, 940], [560, 1900], [505, 1020]];
  const references = [null, null, null];
  const names = ["Open vowel", "Front close vowel", "Back close rounded vowel", "Front mid vowel",
    "Back mid rounded vowel", "Possible front constriction", "Possible back constriction",
    "Possible closure", "Forward hiss", "Rounded hiss", "Diffuse friction", "Possible nasal murmur"];
  const target = new Float64Array(12), weights = new Float64Array(12);
  const result = {weights, active: false, label: "Rest", rms: 0, voicing: 0, f1: 0, f2: 0};
  let previousLevel = 0, previousVoice = 0, closure = 0, pitch = 0;
  const unit = value => Math.max(0, Math.min(1, value));

  function loadFrame(input, inputRate) {
    // Integrate each output sample across its input interval before decimation.
    const ratio = inputRate / rate, start = input.length - size * ratio;
    let sum = 0;
    for (let i = 0; i < size; i++) {
      let left = Math.max(0, start + i * ratio);
      const right = Math.min(input.length, start + (i + 1) * ratio);
      let area = 0;
      while (left < right) {
        const index = Math.floor(left), next = Math.min(right, index + 1);
        area += input[index] * (next - left);
        left = next;
      }
      wave[i] = area / ratio;
      sum += wave[i];
    }
    const mean = sum / size;
    let energy = 0;
    for (let i = 0; i < size; i++) {
      wave[i] -= mean;
      energy += wave[i] * wave[i];
    }
    // Low-pass before the second decimation for LPC and normalized periodicity.
    for (let i = 0; i < tractSize; i++) {
      const j = 2 * i;
      tract[i] = (wave[Math.max(0, j - 1)] + 2 * wave[j] + wave[j + 1]) / 4;
    }
    return Math.sqrt(energy / size);
  }

  function spectralBands() {
    bands.fill(0);
    for (const start of [192, 384]) {
      for (let i = 0; i < 128; i++) bandInput[i] = wave[start + i] * bandWindow[i];
      for (let bin = 1; bin < 64; bin++) {
        let a = 0, b = 0;
        const coefficient = resonator[bin];
        for (let i = 0; i < 128; i++) {
          const next = bandInput[i] + coefficient * a - b;
          b = a;
          a = next;
        }
        const group = bin <= 4 ? 0 : bin <= 12 ? 1 : bin <= 19 ? 2 : bin <= 35 ? 3 : 4;
        bands[group] += a * a + b * b - coefficient * a * b;
      }
    }
    const sum = bands.reduce((a, b) => a + b, 1e-16);
    for (let i = 0; i < bands.length; i++) bands[i] /= sum;
  }

  function periodicity() {
    let best = 0;
    // A full fixed overlap avoids preferring the longest lag by chance.
    for (let lag = 18; lag <= 108; lag++) {
      let xy = 0, xx = 0, yy = 0;
      for (let i = 108; i < tractSize; i += 2) {
        const x = tract[i], y = tract[i - lag];
        xy += x * y;
        xx += x * x;
        yy += y * y;
      }
      lagScore[lag] = xy / Math.sqrt(xx * yy + 1e-20);
      best = Math.max(best, lagScore[lag]);
    }
    pitch = 0;
    for (let lag = 19; lag < 108; lag++) {
      if (lagScore[lag] >= best * .94 && lagScore[lag] >= lagScore[lag - 1] && lagScore[lag] > lagScore[lag + 1]) {
        pitch = rate / (2 * lag);
        break;
      }
    }
    return unit((best - .34) / .56);
  }

  function hasMoreThanOnePartial() {
    // A sinusoid at any frequency satisfies x[t-d] + x[t+d] = k*x[t] + b.
    // Fit that identity at two different delays to allow DC and small noise
    // without mistaking a single spectral line for a vocal-tract envelope.
    for (const lag of [5, 13]) {
      let xSum = 0, ySum = 0, xx = 0, yy = 0, xy = 0;
      const count = size - 2 * lag;
      for (let i = lag; i < size - lag; i++) {
        const x = wave[i], y = wave[i - lag] + wave[i + lag];
        xSum += x;
        ySum += y;
        xx += x * x;
        yy += y * y;
        xy += x * y;
      }
      xx -= xSum * xSum / count;
      yy -= ySum * ySum / count;
      xy -= xSum * ySum / count;
      if (xx > 0 && yy - xy * xy / xx > .004 * xx) return true;
    }
    return false;
  }

  function measureFormants() {
    // Window and pre-emphasize before fitting an all-pole source/filter envelope.
    for (let i = 0; i < tractSize; i++) emphasized[i] = (tract[i] - .94 * tract[Math.max(0, i - 1)]) * hann[i];
    for (let lag = 0; lag <= order; lag++) {
      let sum = 0;
      for (let i = lag; i < tractSize; i++) sum += emphasized[i] * emphasized[i - lag];
      correlation[lag] = sum;
    }
    let error = correlation[0] * 1.0001 + 1e-15;
    lpc[0] = 1;
    for (let i = 1; i <= order; i++) {
      let sum = correlation[i];
      for (let j = 1; j < i; j++) sum += lpc[j] * correlation[i - j];
      const reflection = Math.max(-.995, Math.min(.995, -sum / error));
      previous.set(lpc);
      for (let j = 1; j < i; j++) lpc[j] = previous[j] + reflection * previous[i - j];
      lpc[i] = reflection;
      error *= 1 - reflection * reflection;
    }
    for (let bin = 4; bin < frequencies; bin++) {
      let re = 1, im = 0;
      const offset = bin * (order + 1);
      for (let j = 1; j <= order; j++) {
        re += lpc[j] * cosTable[offset + j];
        im += lpc[j] * sinTable[offset + j];
      }
      envelope[bin] = 1 / (re * re + im * im + 1e-12);
    }
    let f1 = 0, f2 = 0;
    for (let bin = 7; bin < frequencies - 1; bin++) {
      if (envelope[bin] <= envelope[bin - 1] || envelope[bin] <= envelope[bin + 1]) continue;
      // Reject shallow shoulders between resonances, particularly at high F0.
      let left = bin, right = bin;
      while (left > 4 && envelope[left - 1] < envelope[left]) left--;
      while (right < frequencies - 1 && envelope[right + 1] < envelope[right]) right++;
      const prominent = envelope[bin] >= 1.26 * Math.max(envelope[left], envelope[right]);
      const strongShoulder = f1 && envelope[bin] > .22 * envelope[f1 / 25];
      if (!prominent && !strongShoulder) continue;
      const hz = bin * 25;
      if (!f1 && hz <= 1125) f1 = hz;
      else if (f1 && hz > Math.max(570, f1 + 240) && hz < 3300) { f2 = hz; break; }
    }
    result.f1 = f1;
    result.f2 = f2;
  }

  function update(input, inputRate, seconds) {
    const dt = Math.max(.001, Math.min(.15, seconds || 1 / 60));
    const rms = loadFrame(input, inputRate);
    result.rms = rms;
    const gain = unit((20 * Math.log10(rms + 1e-10) + 56) / 19);
    result.active = gain > .06;
    target.fill(0);
    result.f1 = result.f2 = result.voicing = 0;
    const dip = unit(1 - 2.8 * rms / Math.max(previousLevel, .001));
    closure = Math.max(closure * Math.exp(-dt / .026), previousVoice * dip * .45);
    if (result.active) {
      spectralBands();
      const voiced = periodicity();
      result.voicing = voiced;
      if (voiced > .12) {
        if (hasMoreThanOnePartial()) measureFormants();
        const {f1, f2} = result;
        if (f1 && f2) {
          // Allow a modest shift in the vowel space for higher voices. Pitch is
          // only a weak prior; the measured resonance pair still defines place.
          const voiceScale = .98 + .22 * unit((pitch - 125) / 150);
          let sum = 0;
          for (let i = 0; i < vowels.length; i++) {
            const reference = references[i] || vowels[i];
            const scale = references[i] ? 1 : voiceScale;
            const low = Math.log(f1 / (scale * reference[0])) / .22;
            const high = Math.log(f2 / (scale * reference[1])) / .19;
            target[i] = Math.exp(-.5 * (low * low + high * high));
            sum += target[i];
          }
          const scale = gain * voiced / Math.max(.18, sum);
          for (let i = 0; i < vowels.length; i++) target[i] *= scale;
        }
        // When a pair of resonances is unresolved, retain a small jaw opening.
        const oralEvidence = target[0] + target[1] + target[2] + target[3] + target[4];
        target[0] += Math.max(0, .19 * gain * voiced - oralEvidence);
        // A low-frequency murmur with weak oral resonances supports nasalization.
        if (f1) target[11] = gain * voiced * unit((bands[0] - .93) / .065) * unit((450 - f1) / 140) * .7;
      }
      const noise = gain * (1 - voiced * .9);
      // Compare energy per bin across the bands. Diffuse friction spreads its
      // energy more evenly than a narrow sibilant peak, even at the same RMS.
      const widths = [4, 8, 7, 16, 28];
      let logEnergy = 0, meanEnergy = 0;
      for (let i = 0; i < bands.length; i++) {
        const density = bands[i] / widths[i] + 1e-12;
        logEnergy += Math.log(density);
        meanEnergy += density;
      }
      const broad = unit((Math.exp(logEnergy / 5) / (meanEnergy / 5) - .42) / .5);
      const hiss = unit((bands[4] - .14) / .55) * (1 - .79 * broad);
      const sh = unit((bands[2] + bands[3] - .31) / .50) * (1 - hiss) * (1 - .79 * broad);
      target[8] = noise * hiss;
      target[9] = noise * sh;
      target[10] = noise * (1 - Math.max(hiss, sh)) * unit((bands[2] + bands[3] + bands[4]) / .6) * .76;
      // A sudden release and its burst spectrum provide weak place evidence.
      // Steady frication also involves a graded tip/blade constriction.
      const burst = unit((rms - previousLevel * 1.7) / Math.max(rms, .001));
      target[5] = noise * (.38 * hiss + burst * .25 * (sh + hiss));
      target[6] = noise * burst * unit((bands[1] + bands[2] - .24) / .55) * .48;
    }
    target[7] = closure;
    previousLevel += (rms - previousLevel) * (1 - Math.exp(-dt / .035));
    previousVoice = result.voicing * gain;
    let dominant = 0;
    for (let i = 0; i < weights.length; i++) {
      const tau = target[i] > weights[i] ? .021 : .031;
      weights[i] += (target[i] - weights[i]) * (1 - Math.exp(-dt / tau));
      if (weights[i] < .0001) weights[i] = 0;
      if (weights[i] > weights[dominant]) dominant = i;
    }
    result.label = weights[dominant] > .12 ? names[dominant] : result.active ? "Unclear sound" : "Rest";
    if (result.active && result.voicing > .7 && !result.f1 && dominant === 0) result.label = "Unclear voiced sound";
    return result;
  }

  function reset() {
    target.fill(0);
    weights.fill(0);
    previousLevel = previousVoice = closure = 0;
    result.active = false;
    result.label = "Rest";
    result.rms = result.voicing = result.f1 = result.f2 = 0;
  }
  function setVowelReference(index, f1, f2) {
    if (!Number.isInteger(index) || index < 0 || index > 2 ||
        !Number.isFinite(f1) || !Number.isFinite(f2) ||
        f1 < 175 || f1 > 1125 || f2 < 570 || f2 > 3300 || f2 < f1 + 240) return false;
    references[index] = [f1, f2];
    return true;
  }
  function clearReferences() { references.fill(null); }
  return {update, reset, setVowelReference, clearReferences};
}
