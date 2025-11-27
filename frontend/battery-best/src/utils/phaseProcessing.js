import Papa from 'papaparse';

const REQUIRED_COLUMNS = [
  'battery_id',
  'chemistry',
  'cycle_index',
  'sample_index',
  'normalized_time',
  'elapsed_time_s',
  'voltage_v',
  'current_a',
  'c_rate',
  'temperature_k',
];

const DEFAULT_SAMPLE_COUNT = 100;

export const parseCsvFile = (file) =>
  new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results) => {
        if (results.errors.length) {
          reject(new Error(results.errors[0].message));
          return;
        }
        resolve(results.data);
      },
      error: (err) => reject(err),
    });
  });

export const parseCsvString = (text) => {
  const parsed = Papa.parse(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });
  if (parsed.errors.length) {
    throw new Error(parsed.errors[0].message || 'Unable to parse CSV content.');
  }
  return parsed.data;
};

const ensureColumns = (rows) => {
  const missing = REQUIRED_COLUMNS.filter((col) => !(col in (rows[0] || {})));
  if (missing.length) {
    throw new Error(`Missing required columns: ${missing.join(', ')}`);
  }
};

const toNumericRow = (row, fallbackIndex) => ({
  sample_index: Number.isFinite(Number(row.sample_index)) ? Number(row.sample_index) : fallbackIndex,
  normalized_time: Number(row.normalized_time),
  voltage_v: Number(row.voltage_v),
  c_rate: Number(row.c_rate),
  temperature_k: Number(row.temperature_k),
});

const calcStats = (rows, key) => {
  const values = rows
    .map((row) => Number(row[key]))
    .filter((value) => Number.isFinite(value));
  if (!values.length) {
    throw new Error(`Column ${key} does not contain numeric data.`);
  }
  const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
  const variance =
    values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / Math.max(values.length, 1);
  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...values),
    max: Math.max(...values),
  };
};

const summarizePhase = (rows, includeTemperature = true) => {
  ensureColumns(rows);
  const numericRows = rows.map(toNumericRow);

  const clippedRows = numericRows.filter(
    (row) => Number.isFinite(row.voltage_v) && row.voltage_v >= 3.0 && row.voltage_v <= 3.6
  );
  const filteredRows = clippedRows.length ? clippedRows : numericRows;

  const sorted = [...filteredRows].sort((a, b) => a.sample_index - b.sample_index);
  const subset =
    sorted.length > DEFAULT_SAMPLE_COUNT
      ? Array.from({ length: DEFAULT_SAMPLE_COUNT }, (_, idx) => {
          const pointer =
            sorted.length === 1 ? 0 : Math.floor((idx / (DEFAULT_SAMPLE_COUNT - 1)) * (sorted.length - 1));
          return sorted[pointer];
        })
      : sorted;

  const statsSource = subset.length ? subset : sorted;

  const points = statsSource.map((row, idx) => ({
    x: Number.isFinite(row.normalized_time) ? row.normalized_time : row.sample_index ?? idx,
    y: row.voltage_v,
  }));

  return {
    stats: {
      voltage: calcStats(statsSource, 'voltage_v'),
      cRate: calcStats(statsSource, 'c_rate'),
      temperature: includeTemperature ? calcStats(statsSource, 'temperature_k') : null,
    },
    points,
  };
};

const splitRowsByPhase = (rows) => {
  const chargeRows = [];
  const dischargeRows = [];
  rows.forEach((row) => {
    const explicit = (row.phase || '').toString().toLowerCase();
    if (explicit === 'charge') {
      chargeRows.push(row);
      return;
    }
    if (explicit === 'discharge') {
      dischargeRows.push(row);
      return;
    }
    const current = Number(row.current_a);
    if (Number.isFinite(current) && current < 0) {
      dischargeRows.push(row);
    } else {
      chargeRows.push(row);
    }
  });
  return { chargeRows, dischargeRows };
};

export const mergePhaseSummaries = (rows, fileName, currentSummaries = {}) => {
  const { chargeRows, dischargeRows } = splitRowsByPhase(rows);
  if (!chargeRows.length && !dischargeRows.length) {
    throw new Error(`Unable to detect charge/discharge rows in ${fileName}.`);
  }

  let nextSummaries = { ...currentSummaries };
  if (chargeRows.length) {
    const summary = summarizePhase(chargeRows);
    nextSummaries = {
      ...nextSummaries,
      charge: { stats: summary.stats, points: summary.points, fileName },
    };
  }
  if (dischargeRows.length) {
    const summary = summarizePhase(dischargeRows, false);
    nextSummaries = {
      ...nextSummaries,
      discharge: { stats: summary.stats, points: summary.points, fileName },
    };
  }
  return nextSummaries;
};

export const buildFeatureMap = (chargeSummary, dischargeSummary) => {
  if (!chargeSummary || !dischargeSummary) {
    throw new Error('Charge and discharge datasets are both required.');
  }
  return {
    charge_voltage_v_mean: chargeSummary.voltage.mean,
    charge_voltage_v_std: chargeSummary.voltage.std,
    charge_voltage_v_min: chargeSummary.voltage.min,
    charge_voltage_v_max: chargeSummary.voltage.max,
    charge_c_rate_mean: chargeSummary.cRate.mean,
    charge_temperature_k_mean: chargeSummary.temperature.mean,
    discharge_voltage_v_mean: dischargeSummary.voltage.mean,
    discharge_voltage_v_std: dischargeSummary.voltage.std,
    discharge_voltage_v_min: dischargeSummary.voltage.min,
    discharge_voltage_v_max: dischargeSummary.voltage.max,
    discharge_c_rate_mean: dischargeSummary.cRate.mean,
  };
};

