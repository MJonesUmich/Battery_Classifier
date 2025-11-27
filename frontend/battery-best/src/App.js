import BusinessCenterIcon from '@mui/icons-material/BusinessCenter';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SupportAgentIcon from '@mui/icons-material/SupportAgent';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  Grid,
  Link,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import Papa from 'papaparse';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import logo from './logo.svg';
import predictChemistry from './utils/logregPredict';

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

const VoltagePlot = ({ title, color, points = [], axisColor = '#9e9e9e' }) => {
  const width = 350;
  const height = 220;
  const padding = 40;
  const tickCount = 4;

  const processedPoints = points
    .map((point, idx) => ({
      x: Number.isFinite(point.x) ? Number(point.x) : Number(point.sample_index ?? idx),
      y: Number(point.y),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!processedPoints.length) {
    return (
      <Box>
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Not enough data to render this phase.
        </Typography>
      </Box>
    );
  }

  const xValues = processedPoints.map((point) => point.x);
  const yValues = processedPoints.map((point) => point.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const coords = processedPoints.map((point) => {
    const svgX = padding + ((point.x - xMin) / xRange) * (width - padding * 2);
    const svgY = height - padding - ((point.y - yMin) / yRange) * (height - padding * 2);
    return { svgX, svgY, rawX: point.x, rawY: point.y };
  });

  const ticks = (min, max) =>
    Array.from({ length: tickCount }, (_, idx) => min + (idx / (tickCount - 1)) * (max - min));
  const xTicks = ticks(xMin, xMax);
  const yTicks = ticks(yMin, yMax);

  return (
    <Box>
      <Typography variant="subtitle2" fontWeight={600} gutterBottom>
        {title}
      </Typography>
      <Box sx={{ width: '100%', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
          <rect
            x={padding}
            y={padding / 2}
            width={width - padding * 2}
            height={height - padding * 1.5}
            fill="#fafafa"
            stroke="#e0e0e0"
            rx={6}
          />

          {xTicks.map((tick) => {
            const xPos = padding + ((tick - xMin) / xRange) * (width - padding * 2);
            return (
              <line
                key={`grid-x-${tick}`}
                x1={xPos}
                x2={xPos}
                y1={padding / 2}
                y2={height - padding}
                stroke="#eeeeee"
                strokeDasharray="4 4"
              />
            );
          })}
          {yTicks.map((tick) => {
            const yPos = height - padding - ((tick - yMin) / yRange) * (height - padding * 2);
            return (
              <line
                key={`grid-y-${tick}`}
                x1={padding}
                x2={width - padding}
                y1={yPos}
                y2={yPos}
                stroke="#eeeeee"
                strokeDasharray="4 4"
              />
            );
          })}

          <polyline
            points={coords.map((coord) => `${coord.svgX},${coord.svgY}`).join(' ')}
            fill="none"
            stroke={color}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {xTicks.map((tick) => {
            const xPos = padding + ((tick - xMin) / xRange) * (width - padding * 2);
            return (
              <text
                key={`x-${tick}`}
                x={xPos}
                y={height - padding + 16}
                textAnchor="middle"
                fill={axisColor}
                fontSize="11"
              >
                {tick.toFixed(2)}
              </text>
            );
          })}

          {yTicks.map((tick) => {
            const yPos = height - padding - ((tick - yMin) / yRange) * (height - padding * 2);
            return (
              <text
                key={`y-${tick}`}
                x={padding - 8}
                y={yPos + 4}
                textAnchor="end"
                fill={axisColor}
                fontSize="11"
              >
                {tick.toFixed(2)}
              </text>
            );
          })}

          <text
            x={width / 2}
            y={height - 4}
            textAnchor="middle"
            fill={axisColor}
            fontSize="11"
            fontWeight={600}
          >
            Normalized Time / Sample Index
          </text>
          <text
            x={12}
            y={padding / 2 - 6}
            textAnchor="start"
            fill={axisColor}
            fontSize="11"
            fontWeight={600}
          >
            Voltage (V)
          </text>
        </svg>
      </Box>
    </Box>
  );
};

function App() {
  const fileInputRef = useRef(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [phaseSummaries, setPhaseSummaries] = useState({ charge: null, discharge: null });
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const datasetBaseUrl = useMemo(() => `${process.env.PUBLIC_URL || ''}/datasets`, []);
  const [sampleDatasets, setSampleDatasets] = useState([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    const loadManifest = async () => {
      try {
        setDatasetsLoading(true);
        const response = await fetch(`${datasetBaseUrl}/datasets.json`, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Failed to load datasets.json (${response.status})`);
        }
        const payload = await response.json();
        const entries = (payload.datasets || []).map((item) => ({
          ...item,
          url: `${datasetBaseUrl}/${item.file}`,
        }));
        setSampleDatasets(entries);
        setDatasetsError('');
      } catch (err) {
        if (err.name !== 'AbortError') {
          setDatasetsError(err.message || 'Unable to load sample datasets.');
        }
      } finally {
        setDatasetsLoading(false);
      }
    };
    loadManifest();
    return () => controller.abort();
  }, [datasetBaseUrl]);

  const parseCsvFile = (file) =>
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
              sorted.length === 1
                ? 0
                : Math.floor((idx / (DEFAULT_SAMPLE_COUNT - 1)) * (sorted.length - 1));
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

  const mergePhaseSummaries = (rows, fileName, currentSummaries) => {
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

  const buildFeatureMap = useCallback((chargeSummary, dischargeSummary) => {
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
  }, []);

  const handleSelectFile = () => {
    fileInputRef.current?.click();
  };

  const runPrediction = useCallback(
    (summaries) => {
      const featureMap = buildFeatureMap(summaries.charge?.stats, summaries.discharge?.stats);
      const result = predictChemistry(featureMap);
      setPrediction(result.label);
      setProbabilities(result.probabilities);
      setStatusMessage('Prediction generated using uploaded charge & discharge files.');
      setErrorMessage('');
    },
    [buildFeatureMap]
  );

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }
    setSelectedFileName(files.map((file) => file.name).join(', '));
    setIsProcessing(true);
    setErrorMessage('');
    setStatusMessage('Parsing uploaded file(s)...');

    let updatedSummaries = { ...phaseSummaries };
    try {
      for (const file of files) {
        const rows = await parseCsvFile(file);
        if (!rows.length) {
          throw new Error(`File ${file.name} does not contain any rows.`);
        }
        updatedSummaries = mergePhaseSummaries(rows, file.name, updatedSummaries);
      }
      setPhaseSummaries(updatedSummaries);
      if (updatedSummaries.charge && updatedSummaries.discharge) {
        runPrediction(updatedSummaries);
      } else {
        setStatusMessage('Upload both charge and discharge CSV files to run the prediction.');
        setPrediction(null);
        setProbabilities(null);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSamplePredict = async (dataset) => {
    if (!dataset?.url) return;
    try {
      setIsProcessing(true);
      setStatusMessage(`Loading sample "${dataset.title}"...`);
      setErrorMessage('');

      const response = await fetch(dataset.url);
      if (!response.ok) {
        throw new Error(`Failed to fetch sample CSV (${response.status})`);
      }
      const text = await response.text();
      const parsed = Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
      });
      if (parsed.errors.length) {
        throw new Error(parsed.errors[0].message || 'Unable to parse sample CSV.');
      }

      const updatedSummaries = mergePhaseSummaries(parsed.data, dataset.file, {});
      setPhaseSummaries(updatedSummaries);
      setSelectedFileName(`${dataset.file} (sample)`);

      if (updatedSummaries.charge && updatedSummaries.discharge) {
        runPrediction(updatedSummaries);
      } else {
        setStatusMessage('Sample is missing charge or discharge data.');
        setPrediction(null);
        setProbabilities(null);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunModel = () => {
    if (phaseSummaries.charge && phaseSummaries.discharge) {
      try {
        runPrediction(phaseSummaries);
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : String(err));
      }
    }
  };

  const probabilityEntries = useMemo(() => {
    if (!probabilities) return [];
    return Object.entries(probabilities).map(([label, value]) => ({
      label,
      percent: (value * 100).toFixed(2),
    }));
  }, [probabilities]);

  const topProbability = useMemo(() => {
    if (!prediction || !probabilities) return null;
    const value = probabilities[prediction];
    return typeof value === 'number' ? value * 100 : null;
  }, [prediction, probabilities]);

  const chargePoints = phaseSummaries.charge?.points || [];
  const dischargePoints = phaseSummaries.discharge?.points || [];

  return (
    <Box className="app-root">
      <Container maxWidth="md" sx={{ py: { xs: 6, md: 8 } }}>
        <Stack spacing={4}>
          <Box textAlign="center">
            <Box
              component="img"
              src={logo}
              alt="Battery Insight Studio logo"
              sx={{
                width: { xs: 72, sm: 84, md: 600 },
                height: 'auto',
                mx: 'auto',
                mb: 2,
              }}
            />

            <Typography variant="body1" color="text.secondary">
              Upload raw operational or lab data from your batteries. The platform will
              clean the signals, train diagnostic models, and estimate the current health
              to help you evaluate fleet performance in minutes.
            </Typography>
            <Chip
              label="Trial Workspace"
              color="primary"
              variant="outlined"
              sx={{ mt: 2, fontWeight: 600 }}
            />
          </Box>

          <Alert severity="info" sx={{ alignItems: 'center' }}>
            <Typography variant="body2">
              You are exploring the interactive trial experience. Once you validate the
              workflow, switch to our managed batch update pipeline or request direct API
              access for large-scale inference.
            </Typography>
          </Alert>

          <Paper elevation={3} sx={{ p: { xs: 3, md: 4 } }}>
            <Stack spacing={2.5}>
              <Stack direction="row" spacing={1.5} alignItems="flex-start">
                <InfoOutlinedIcon color="primary" fontSize="large" />
                <Stack spacing={0.75}>
                  <Typography variant="h5" fontWeight={600}>
                    How It Works
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    1. Prepare a dataset containing fields such as time, voltage, current,
                    and temperature in CSV or XLSX format.
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    2. Click the upload button below, verify the preview, and start the
                    analysis.
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    3. The model will return health index, remaining life, and other KPIs
                    (placeholder UI in this version).
                  </Typography>
                </Stack>
              </Stack>
            </Stack>
          </Paper>

          <Paper elevation={1} sx={{ p: { xs: 3, md: 4 } }}>
            <Stack spacing={3}>
              <Stack spacing={1}>
                <Typography variant="h5" fontWeight={600}>
                  Upload Dataset
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supports CSV and XLSX files. Parsing and model inference run in the
                  background and typically take 1-2 minutes.
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Trial mode processes a single asset at a time with limited retention.
                </Typography>
              </Stack>

              <Paper
                variant="outlined"
                sx={{
                  p: { xs: 2, sm: 3 },
                  backgroundColor: 'grey.50',
                  borderStyle: 'dashed',
                }}
              >
                <Stack spacing={2}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Need a dataset to try?
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Grab one of the sample CSV files below if you don't have your own data yet, then come back to upload.
                  </Typography>
                  {datasetsLoading ? (
                    <Stack direction="row" spacing={1} alignItems="center">
                      <CircularProgress size={20} />
                      <Typography variant="body2" color="text.secondary">
                        Loading sample datasets...
                      </Typography>
                    </Stack>
                  ) : datasetsError ? (
                    <Alert severity="warning">{datasetsError}</Alert>
                  ) : sampleDatasets.length ? (
                    <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ width: '100%' }}>
                      {sampleDatasets.map((dataset) => (
                        <Paper key={dataset.id} variant="outlined" sx={{ p: 2, flex: 1 }}>
                          <Stack spacing={1.5}>
                            <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                              <Typography variant="subtitle2" fontWeight={600}>
                                {dataset.title}
                              </Typography>
                              <Chip label={dataset.size || 'CSV'} size="small" color="primary" variant="outlined" />
                            </Stack>
                            <Typography variant="body2" color="text.secondary">
                              {dataset.description}
                            </Typography>
                            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                              <Button
                                component="a"
                                href={dataset.url}
                                download
                                size="small"
                                variant="outlined"
                                color="primary"
                                fullWidth
                              >
                                Download
                              </Button>
                              <Button
                                size="small"
                                variant="contained"
                                color="primary"
                                fullWidth
                                onClick={() => handleSamplePredict(dataset)}
                                disabled={isProcessing}
                              >
                                Predict Now
                              </Button>
                            </Stack>
                          </Stack>
                        </Paper>
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No sample datasets available. Please regenerate them via `create_demo_datasets.py`.
                    </Typography>
                  )}
                </Stack>
              </Paper>

              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems="center">
                {['charge', 'discharge'].map((phase) => {
                  const phaseData = phaseSummaries[phase];
                  return (
                    <Chip
                      key={phase}
                      color={phaseData ? 'success' : 'default'}
                      variant={phaseData ? 'filled' : 'outlined'}
                      label={
                        phaseData ? `${phase.toUpperCase()} · ${phaseData.fileName}` : `${phase.toUpperCase()} pending`
                      }
                    />
                  );
                })}
              </Stack>

              {errorMessage && (
                <Alert severity="error" onClose={() => setErrorMessage('')}>
                  {errorMessage}
                </Alert>
              )}
              {statusMessage && !errorMessage && (
                <Alert severity="info" onClose={() => setStatusMessage('')}>
                  {statusMessage}
                </Alert>
              )}

              <input
                hidden
                ref={fileInputRef}
                type="file"
                multiple
                accept=".csv"
                onChange={handleFileChange}
              />

              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={2}
                alignItems={{ xs: 'stretch', sm: 'center' }}
              >
                <Button
                  variant="contained"
                  startIcon={<CloudUploadIcon />}
                  onClick={handleSelectFile}
                  size="large"
                  disabled={isProcessing}
                >
                  {isProcessing ? 'Processing...' : 'Choose File(s)'}
                </Button>
                {isProcessing && <CircularProgress size={24} />}
                <Typography
                  variant="body2"
                  color={selectedFileName ? 'text.primary' : 'text.secondary'}
                  sx={{ flexGrow: 1 }}
                >
                  {selectedFileName || 'No file selected'}
                </Typography>
              </Stack>

              <Divider />

              <Stack spacing={1.5}>
                <Button
                  variant="contained"
                  color="secondary"
                  startIcon={<PlayArrowIcon />}
                  size="large"
                  onClick={handleRunModel}
                  disabled={!(phaseSummaries.charge && phaseSummaries.discharge) || isProcessing}
                >
                  Run Model Analysis
                </Button>
                <Typography variant="caption" color="text.secondary">
                  Model inference will run on the backend. This page is not yet connected
                  to the processing pipeline.
                </Typography>
              </Stack>
            </Stack>
          </Paper>

          <Paper
            variant="outlined"
            sx={{
              p: { xs: 3, md: 4 },
              borderStyle: 'dashed',
              borderColor: 'primary.light',
              backgroundColor: 'rgba(25, 118, 210, 0.04)',
            }}
          >
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              Prediction Preview
            </Typography>
            {prediction ? (
              <Stack spacing={2}>
                <Typography variant="h4" fontWeight={700}>
                  {prediction}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Top prediction with probability {topProbability ? topProbability.toFixed(2) : '--'}%
                </Typography>
                <Stack spacing={1}>
                  {probabilityEntries.map((entry) => (
                    <Stack
                      key={entry.label}
                      direction="row"
                      spacing={1}
                      alignItems="center"
                      justifyContent="space-between"
                    >
                      <Typography variant="body2" fontWeight={600}>
                        {entry.label}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {entry.percent}%
                      </Typography>
                    </Stack>
                  ))}
                </Stack>
                <Box>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <VoltagePlot
                        title="Charge Voltage Profile"
                        color="#1976d2"
                        points={chargePoints}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <VoltagePlot
                        title="Discharge Voltage Profile"
                        color="#ff7043"
                        points={dischargePoints}
                      />
                    </Grid>
                  </Grid>
                </Box>
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Upload both charge and discharge CSV files from the same cycle to preview the chemistry
                classification probabilities.
              </Typography>
            )}
          </Paper>

          <Paper elevation={0} sx={{ p: { xs: 3, md: 4 }, border: '1px solid', borderColor: 'divider' }}>
            <Stack spacing={3}>
              <Stack direction="row" spacing={1.5} alignItems="center">
                <BusinessCenterIcon color="primary" fontSize="large" />
                <Typography variant="h5" fontWeight={600}>
                  Ready for Batch Update?
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                Keep your production fleets synchronized with nightly batch updates,
                automated health reports, and seamless API integration. We tailor the
                deployment plan to your data cadence and infrastructure requirements.
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2.5, height: '100%' }}>
                    <Stack spacing={1.5}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip label="Batch Predict" color="primary" size="small" />
                        <Typography variant="subtitle2" fontWeight={600}>
                          High-volume scoring
                        </Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        Stream ingest-ready files and receive inference results through the
                        batch update queue or via REST endpoints with secure credentials.
                      </Typography>
                    </Stack>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2.5, height: '100%' }}>
                    <Stack spacing={1.5}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip label="Enterprise Support" color="secondary" size="small" />
                        <Typography variant="subtitle2" fontWeight={600}>
                          Co-design the rollout
                        </Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        Collaborate with our engineering team to define monitoring SLAs,
                        deployment guardrails, and integration checkpoints tailored to your
                        roadmap.
                      </Typography>
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
                <Button
                  variant="contained"
                  startIcon={<SupportAgentIcon />}
                  href="mailto:solutions@batteryinsight.ai?subject=Battery%20Insight%20Studio%20-%20Batch%20Predict"
                >
                  Contact Solutions Team
                </Button>
                <Button
                  variant="outlined"
                  color="primary"
                  href=""
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Schedule a Planning Call
                </Button>
                <Typography variant="caption" color="text.secondary">
                  Prefer another channel? Reach us via{' '}
                  <Link href="https://batteryinsight.ai" target="_blank" rel="noopener noreferrer">
                    batteryinsight.ai
                  </Link>
                  .
                </Typography>
              </Stack>
            </Stack>
          </Paper>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
