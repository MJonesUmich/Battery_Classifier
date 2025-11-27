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
  Container,
  Divider,
  Grid,
  Link,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import { useRef, useState } from 'react';
import './App.css';
import logo from './logo.svg';

function App() {
  const fileInputRef = useRef(null);
  const [selectedFileName, setSelectedFileName] = useState('');

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

  const handleSelectFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setSelectedFileName(file ? file.name : '');
  };

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

              <input
                hidden
                ref={fileInputRef}
                type="file"
                accept=".csv,.xls,.xlsx"
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
                >
                  Choose File
                </Button>
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
                  disabled={!selectedFileName}
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
            <Typography variant="body2" color="text.secondary">
              Once the model integration is complete, this section will display health
              indices, remaining life curves, and other performance insights.
            </Typography>
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
