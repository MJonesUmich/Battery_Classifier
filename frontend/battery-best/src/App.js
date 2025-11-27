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
import { useCallback, useMemo, useRef, useState } from 'react';
import './App.css';
import predictChemistry from './utils/logregPredict';
import logo from './logo.svg';

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

function App() {
  const fileInputRef = useRef(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [phaseSummaries, setPhaseSummaries] = useState({ charge: null, discharge: null });
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const datasetBaseUrl = `${process.env.PUBLIC_URL || ''}/datasets`;
  const sampleDatasets = [
    {
      id: 'lco-charge',
      title: 'LCO · Charge Cycle',
      size: '~5 KB CSV',
      url: `${datasetBaseUrl}/LCO_sample_charge.csv`,
      description: '100-point constant-current charge slice from the Capacity_25C cell.',
    },
    {
      id: 'lco-discharge',
      title: 'LCO · Discharge Cycle',
      size: '~5 KB CSV',
      url: `${datasetBaseUrl}/LCO_sample_discharge.csv`,
      description: 'Matching discharge curve covering the same cycle for reference.',
    },
    {
      id: 'lfp-charge',
      title: 'LFP · Charge Cycle',
      size: '~5 KB CSV',
      url: `${datasetBaseUrl}/LFP_sample_charge.csv`,
      description: 'LFP chemistry example highlighting a slower voltage ramp.',
    },
    {
      id: 'lfp-discharge',
      title: 'LFP · Discharge Cycle',
      size: '~5 KB CSV',
      url: `${datasetBaseUrl}/LFP_sample_discharge.csv`,
      description: 'Companion discharge sample showcasing the flat voltage plateau.',
    },
  ];

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

  const detectPhase = (fileName, rows) => {
    const lower = fileName.toLowerCase();
    if (lower.includes('discharge')) return 'discharge';
    if (lower.includes('charge')) return 'charge';
    const avgCurrent =
      rows.reduce((acc, row) => acc + (Number(row.current_a) || 0), 0) / Math.max(rows.length, 1);
    return avgCurrent < 0 ? 'discharge' : 'charge';
  };

  const ensureColumns = (rows) => {
    const missing = REQUIRED_COLUMNS.filter((col) => !(col in (rows[0] || {})));
    if (missing.length) {
      throw new Error(`Missing required columns: ${missing.join(', ')}`);
    }
  };

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

  const summarizePhase = (rows) => {
    ensureColumns(rows);
    const sorted = [...rows].sort((a, b) => Number(a.sample_index) - Number(b.sample_index));
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

    return {
      voltage: calcStats(subset, 'voltage_v'),
      cRate: calcStats(subset, 'c_rate'),
      temperature: calcStats(subset, 'temperature_k'),
    };
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
        ensureColumns(rows);
        const phase = detectPhase(file.name, rows);
        const stats = summarizePhase(rows);
        updatedSummaries = {
          ...updatedSummaries,
          [phase]: { stats, fileName: file.name },
        };
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
                  <Stack
                    direction={{ xs: 'column', md: 'row' }}
                    spacing={2}
                    sx={{ width: '100%' }}
                  >
                    {sampleDatasets.map((dataset) => (
                      <Paper key={dataset.id} variant="outlined" sx={{ p: 2, flex: 1 }}>
                        <Stack spacing={1.5}>
                          <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                            <Typography variant="subtitle2" fontWeight={600}>
                              {dataset.title}
                            </Typography>
                            <Chip label={dataset.size} size="small" color="primary" variant="outlined" />
                          </Stack>
                          <Typography variant="body2" color="text.secondary">
                            {dataset.description}
                          </Typography>
                          <Button
                            component="a"
                            href={dataset.url}
                            download
                            size="small"
                            variant="contained"
                            color="primary"
                            sx={{ alignSelf: 'flex-start' }}
                          >
                            Download CSV
                          </Button>
                        </Stack>
                      </Paper>
                    ))}
                  </Stack>
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
